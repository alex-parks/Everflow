"""
Inference Pipeline for Crowd Enhancement V2V Model
Handles real-time inference with camera matching and temporal consistency
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import json
import time
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import OpenImageIO as oiio

from .v2v_model import CrowdEnhancementNetwork, create_model
from .training_pipeline import TrainingDataFormat

logger = logging.getLogger(__name__)


@dataclass
class CameraTransform:
    """Camera transformation data for exact reference matching"""
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]  # Euler angles (pitch, yaw, roll)
    fov: float
    near_clip: float
    far_clip: float
    resolution: Tuple[int, int]
    
    # Camera matrices for precise matching
    view_matrix: np.ndarray
    projection_matrix: np.ndarray
    world_to_screen_matrix: np.ndarray
    
    @classmethod
    def from_blender_camera(cls, camera_data: Dict) -> 'CameraTransform':
        """Create from Blender camera export data"""
        return cls(
            position=tuple(camera_data['location']),
            rotation=tuple(camera_data['rotation_euler']),
            fov=camera_data['angle'],
            near_clip=camera_data['clip_start'],
            far_clip=camera_data['clip_end'],
            resolution=tuple(camera_data['resolution']),
            view_matrix=np.array(camera_data['matrix_world']).reshape(4, 4),
            projection_matrix=np.array(camera_data['projection_matrix']).reshape(4, 4),
            world_to_screen_matrix=np.array(camera_data['world_to_screen']).reshape(4, 4)
        )
    
    def matches_camera(self, other: 'CameraTransform', tolerance: float = 0.001) -> bool:
        """Check if this camera transform matches another within tolerance"""
        pos_diff = np.linalg.norm(np.array(self.position) - np.array(other.position))
        rot_diff = np.linalg.norm(np.array(self.rotation) - np.array(other.rotation))
        fov_diff = abs(self.fov - other.fov)
        
        return (pos_diff < tolerance and 
                rot_diff < tolerance and 
                fov_diff < tolerance and
                self.resolution == other.resolution)


@dataclass
class InferenceFrame:
    """Single frame data for inference"""
    frame_number: int
    timestamp: float
    camera_transform: CameraTransform
    
    # Multi-pass input data
    beauty_pass: np.ndarray
    depth_pass: np.ndarray
    emission_pass: Optional[np.ndarray] = None
    normal_pass: Optional[np.ndarray] = None
    motion_vector_pass: Optional[np.ndarray] = None
    
    # Processed data for model input
    input_tensor: Optional[torch.Tensor] = None
    
    def to_model_input(self, device: str = 'cuda') -> torch.Tensor:
        """Convert frame data to model input tensor"""
        if self.input_tensor is not None:
            return self.input_tensor.to(device)
        
        # Process each pass
        height, width = self.beauty_pass.shape[:2]
        
        # Beauty pass (RGB)
        beauty_rgb = self.beauty_pass[:, :, :3] if self.beauty_pass.shape[2] >= 3 else self.beauty_pass
        
        # Depth pass (single channel)
        depth_normalized = self._normalize_depth(self.depth_pass)
        
        # Emission pass (RGB)
        emission_rgb = self.emission_pass if self.emission_pass is not None else np.zeros_like(beauty_rgb)
        
        # Normal pass (RGB)
        normal_rgb = self.normal_pass if self.normal_pass is not None else np.zeros_like(beauty_rgb)
        if self.normal_pass is not None:
            normal_rgb = (normal_rgb - 0.5) * 2.0  # Normalize to [-1, 1]
        
        # Stack all channels [H, W, 10]
        input_data = np.dstack([
            beauty_rgb,                           # 3 channels
            depth_normalized[..., np.newaxis],    # 1 channel  
            emission_rgb,                         # 3 channels
            normal_rgb                            # 3 channels
        ])
        
        # Convert to tensor and normalize
        input_tensor = torch.from_numpy(input_data.transpose(2, 0, 1)).float()
        input_tensor = input_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        
        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)
        
        self.input_tensor = input_tensor
        return input_tensor.to(device)
    
    def _normalize_depth(self, depth_data: np.ndarray) -> np.ndarray:
        """Normalize depth data to [0, 1] range"""
        if depth_data.max() == depth_data.min():
            return np.zeros_like(depth_data)
        
        # Use camera clip planes for normalization if available
        if hasattr(self, 'camera_transform'):
            near = self.camera_transform.near_clip
            far = self.camera_transform.far_clip
            depth_normalized = (depth_data - near) / (far - near)
            depth_normalized = np.clip(depth_normalized, 0, 1)
        else:
            # Fallback normalization
            depth_clipped = np.clip(depth_data, np.percentile(depth_data, 1), np.percentile(depth_data, 99))
            depth_normalized = (depth_clipped - depth_clipped.min()) / (depth_clipped.max() - depth_clipped.min())
        
        return depth_normalized.astype(np.float32)


class TemporalConsistencyManager:
    """Manages temporal consistency across frame sequence"""
    
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.frame_history: List[Dict] = []
        self.feature_cache: Dict[int, torch.Tensor] = {}
    
    def add_frame_result(self, frame_number: int, features: torch.Tensor, output: torch.Tensor):
        """Add frame result to history"""
        frame_data = {
            'frame_number': frame_number,
            'timestamp': time.time(),
            'features': features.detach().cpu(),
            'output': output.detach().cpu()
        }
        
        self.frame_history.append(frame_data)
        
        # Keep only recent history
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
        
        # Cache features for temporal consistency
        self.feature_cache[frame_number] = features.detach()
    
    def get_temporal_context(self, current_frame: int) -> Optional[torch.Tensor]:
        """Get temporal context for current frame"""
        if not self.frame_history:
            return None
        
        # Find most recent previous frame
        previous_frame = None
        for frame_data in reversed(self.frame_history):
            if frame_data['frame_number'] < current_frame:
                previous_frame = frame_data
                break
        
        if previous_frame is None:
            return None
        
        return previous_frame['features']
    
    def apply_temporal_smoothing(self, 
                                current_output: torch.Tensor, 
                                frame_number: int,
                                smoothing_factor: float = 0.1) -> torch.Tensor:
        """Apply temporal smoothing to reduce flickering"""
        if not self.frame_history:
            return current_output
        
        # Get previous frame output
        previous_frame = None
        for frame_data in reversed(self.frame_history):
            if frame_data['frame_number'] == frame_number - 1:
                previous_frame = frame_data
                break
        
        if previous_frame is None:
            return current_output
        
        previous_output = previous_frame['output'].to(current_output.device)
        
        # Blend current and previous outputs
        smoothed_output = (1 - smoothing_factor) * current_output + smoothing_factor * previous_output
        
        return smoothed_output


class CrowdEnhancementInference:
    """Real-time inference engine for crowd enhancement"""
    
    def __init__(self, 
                 model_path: Path,
                 device: str = 'cuda',
                 batch_size: int = 1,
                 use_temporal_consistency: bool = True):
        
        self.device = device
        self.batch_size = batch_size
        self.use_temporal_consistency = use_temporal_consistency
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Temporal consistency manager
        self.temporal_manager = TemporalConsistencyManager() if use_temporal_consistency else None
        
        # Performance monitoring
        self.inference_times = []
        self.total_frames_processed = 0
        
        # Threading for async processing
        self.processing_queue = Queue(maxsize=10)
        self.result_queue = Queue()
        self.is_processing = False
        
        logger.info(f"Inference engine initialized on {device}")
    
    def _load_model(self, model_path: Path) -> CrowdEnhancementNetwork:
        """Load trained model from checkpoint"""
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model (assumes medium config - could be made configurable)
        model = create_model('medium', self.device)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def preprocess_frame(self, frame_data: InferenceFrame) -> torch.Tensor:
        """Preprocess frame for inference"""
        return frame_data.to_model_input(self.device)
    
    @torch.no_grad()
    def enhance_frame(self, frame_data: InferenceFrame) -> Dict[str, Any]:
        """Enhance a single frame"""
        start_time = time.time()
        
        # Preprocess input
        input_tensor = self.preprocess_frame(frame_data)
        
        # Get temporal context if available
        temporal_context = None
        if self.temporal_manager:
            temporal_context = self.temporal_manager.get_temporal_context(frame_data.frame_number)
        
        # Run inference
        with torch.cuda.amp.autocast():  # Use mixed precision for speed
            results = self.model(input_tensor, temporal_context)
        
        # Get enhanced image
        enhanced_image = results['enhanced_image']
        
        # Apply temporal smoothing if enabled
        if self.temporal_manager:
            enhanced_image = self.temporal_manager.apply_temporal_smoothing(
                enhanced_image, 
                frame_data.frame_number
            )
            
            # Update temporal history
            self.temporal_manager.add_frame_result(
                frame_data.frame_number,
                results.get('features', enhanced_image),
                enhanced_image
            )
        
        # Convert output to displayable format
        output_image = self._tensor_to_image(enhanced_image)
        
        # Performance tracking
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.total_frames_processed += 1
        
        # Keep only recent timing data
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        
        return {
            'enhanced_image': output_image,
            'person_masks': results.get('person_masks'),
            'attention_weights': results.get('attention_weights'),
            'inference_time': inference_time,
            'frame_number': frame_data.frame_number,
            'camera_transform': frame_data.camera_transform
        }
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert model output tensor to image array"""
        # Remove batch dimension and move to CPU
        image = tensor.squeeze(0).cpu().numpy()
        
        # Convert from [-1, 1] to [0, 1]
        image = (image + 1.0) / 2.0
        
        # Transpose from CHW to HWC
        image = image.transpose(1, 2, 0)
        
        # Clip values and convert to uint8
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        return image
    
    def enhance_sequence(self, 
                        frames: List[InferenceFrame],
                        progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """Enhance a sequence of frames with temporal consistency"""
        results = []
        
        for i, frame in enumerate(frames):
            result = self.enhance_frame(frame)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(frames), result['inference_time'])
        
        return results
    
    def start_async_processing(self):
        """Start asynchronous frame processing thread"""
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._async_processing_loop)
        self.processing_thread.start()
        logger.info("Async processing started")
    
    def stop_async_processing(self):
        """Stop asynchronous processing"""
        self.is_processing = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        logger.info("Async processing stopped")
    
    def _async_processing_loop(self):
        """Asynchronous processing loop"""
        while self.is_processing:
            try:
                frame_data = self.processing_queue.get(timeout=1.0)
                result = self.enhance_frame(frame_data)
                self.result_queue.put(result)
            except:
                continue
    
    def submit_frame_async(self, frame_data: InferenceFrame) -> bool:
        """Submit frame for asynchronous processing"""
        try:
            self.processing_queue.put_nowait(frame_data)
            return True
        except:
            return False
    
    def get_result_async(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get result from asynchronous processing"""
        try:
            return self.result_queue.get(timeout=timeout)
        except:
            return None
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        avg_time = np.mean(self.inference_times)
        min_time = np.min(self.inference_times)
        max_time = np.max(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'average_inference_time': avg_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'average_fps': fps,
            'total_frames_processed': self.total_frames_processed
        }


class SequenceProcessor:
    """High-level processor for crowd enhancement sequences"""
    
    def __init__(self, 
                 model_path: Path,
                 output_dir: Path,
                 device: str = 'cuda'):
        
        self.inference_engine = CrowdEnhancementInference(model_path, device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process_exr_sequence(self, 
                           sequence_dir: Path, 
                           output_format: str = 'exr',
                           progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Process an EXR sequence directory"""
        
        # Load sequence metadata
        metadata_path = sequence_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Sequence metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            sequence_metadata = json.load(f)
        
        # Create frames list
        frames = []
        frame_count = sequence_metadata['frame_count']
        
        for frame_num in range(frame_count):
            try:
                frame = self._load_frame_from_sequence(sequence_dir, frame_num, sequence_metadata)
                frames.append(frame)
            except Exception as e:
                logger.warning(f"Failed to load frame {frame_num}: {e}")
                continue
        
        logger.info(f"Loaded {len(frames)} frames for processing")
        
        # Process sequence
        results = self.inference_engine.enhance_sequence(frames, progress_callback)
        
        # Save results
        output_sequence_dir = self.output_dir / f"enhanced_{sequence_dir.name}"
        output_sequence_dir.mkdir(parents=True, exist_ok=True)
        
        saved_frames = []
        for i, result in enumerate(results):
            try:
                output_path = self._save_enhanced_frame(
                    result, 
                    output_sequence_dir, 
                    i, 
                    output_format
                )
                saved_frames.append(str(output_path))
            except Exception as e:
                logger.error(f"Failed to save frame {i}: {e}")
        
        # Save processing metadata
        processing_metadata = {
            'source_sequence': str(sequence_dir),
            'output_directory': str(output_sequence_dir),
            'frames_processed': len(results),
            'performance_stats': self.inference_engine.get_performance_stats(),
            'processing_timestamp': time.time()
        }
        
        metadata_output_path = output_sequence_dir / "processing_metadata.json"
        with open(metadata_output_path, 'w') as f:
            json.dump(processing_metadata, f, indent=2)
        
        return {
            'output_directory': output_sequence_dir,
            'frames_saved': saved_frames,
            'processing_metadata': processing_metadata
        }
    
    def _load_frame_from_sequence(self, 
                                 sequence_dir: Path, 
                                 frame_num: int, 
                                 metadata: Dict) -> InferenceFrame:
        """Load a single frame from sequence directory"""
        
        # Load camera data
        camera_transform = CameraTransform.from_blender_camera(metadata['camera_data'])
        
        # Load beauty pass
        beauty_file = sequence_dir / f"beauty_{frame_num:04d}.exr"
        beauty_data = self._load_exr_as_array(beauty_file)
        
        # Load depth pass
        depth_file = sequence_dir / f"depth_{frame_num:04d}.exr"
        depth_data = self._load_exr_as_array(depth_file)
        
        # Load optional passes
        emission_data = None
        emission_file = sequence_dir / f"emission_{frame_num:04d}.exr"
        if emission_file.exists():
            emission_data = self._load_exr_as_array(emission_file)
        
        normal_data = None
        normal_file = sequence_dir / f"normal_{frame_num:04d}.exr"
        if normal_file.exists():
            normal_data = self._load_exr_as_array(normal_file)
        
        return InferenceFrame(
            frame_number=frame_num,
            timestamp=frame_num / metadata.get('frame_rate', 24.0),
            camera_transform=camera_transform,
            beauty_pass=beauty_data,
            depth_pass=depth_data,
            emission_pass=emission_data,
            normal_pass=normal_data
        )
    
    def _load_exr_as_array(self, file_path: Path) -> np.ndarray:
        """Load EXR file as numpy array"""
        try:
            img_input = oiio.ImageInput.open(str(file_path))
            if not img_input:
                raise Exception(f"Could not open {file_path}")
            
            spec = img_input.spec()
            width, height = spec.width, spec.height
            channels = spec.nchannels
            
            # Read image data
            image_data = img_input.read_image()
            img_input.close()
            
            if image_data is None:
                raise Exception("Failed to read image data")
            
            # Convert to numpy array
            image_array = np.array(image_data)
            
            # Ensure proper shape [H, W, C]
            if len(image_array.shape) == 2:
                image_array = image_array[..., np.newaxis]
            elif len(image_array.shape) == 3 and image_array.shape[0] == channels:
                image_array = image_array.transpose(1, 2, 0)
            
            return image_array.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error loading EXR {file_path}: {e}")
            raise
    
    def _save_enhanced_frame(self, 
                           result: Dict[str, Any], 
                           output_dir: Path, 
                           frame_num: int, 
                           format: str = 'exr') -> Path:
        """Save enhanced frame to disk"""
        
        enhanced_image = result['enhanced_image']
        
        if format.lower() == 'exr':
            # Save as EXR for HDR pipeline compatibility
            output_path = output_dir / f"enhanced_{frame_num:04d}.exr"
            
            # Convert back to float32 HDR range
            hdr_image = enhanced_image.astype(np.float32) / 255.0
            
            # Use OpenImageIO to write EXR
            img_output = oiio.ImageOutput.create(str(output_path))
            if not img_output:
                raise Exception(f"Could not create output file: {output_path}")
            
            spec = oiio.ImageSpec(hdr_image.shape[1], hdr_image.shape[0], hdr_image.shape[2], oiio.FLOAT)
            img_output.open(str(output_path), spec)
            img_output.write_image(hdr_image)
            img_output.close()
            
        else:
            # Save as standard image format
            output_path = output_dir / f"enhanced_{frame_num:04d}.{format}"
            cv2.imwrite(str(output_path), cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
        
        return output_path


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Crowd Enhancement Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--input", type=str, required=True, help="Input sequence directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--format", type=str, default="exr", help="Output format")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    processor = SequenceProcessor(
        model_path=Path(args.model),
        output_dir=Path(args.output),
        device=args.device
    )
    
    def progress_callback(current, total, inference_time):
        print(f"Processing frame {current}/{total} ({inference_time:.3f}s)")
    
    result = processor.process_exr_sequence(
        sequence_dir=Path(args.input),
        output_format=args.format,
        progress_callback=progress_callback
    )
    
    print(f"Processing completed. Output: {result['output_directory']}")
    print(f"Performance: {result['processing_metadata']['performance_stats']}")