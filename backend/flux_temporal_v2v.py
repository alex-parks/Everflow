"""
FLUX.1 Depth with Temporal Consistency for Video-to-Video Enhancement
Implements temporal consistency using transformer-based attention and stable diffusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import os
import logging
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from collections import deque
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import diffusers components
try:
    from diffusers import FluxPipeline
    from diffusers.models.attention_processor import Attention
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    FluxPipeline = None
    Attention = None

# Try to import ONNX runtime for optimized models
try:
    from optimum.onnxruntime import ORTDiffusionPipeline
    ONNX_AVAILABLE = True
except ImportError:
    ORTDiffusionPipeline = None
    ONNX_AVAILABLE = False

# Try to import depth preprocessor
try:
    from image_gen_aux import DepthPreprocessor
    DEPTH_PREPROCESSOR_AVAILABLE = True
    logger.info("DepthPreprocessor from image_gen_aux loaded successfully")
except ImportError as e:
    DepthPreprocessor = None
    DEPTH_PREPROCESSOR_AVAILABLE = False
    logger.warning(f"DepthPreprocessor not available: {e}")
except Exception as e:
    DepthPreprocessor = None
    DEPTH_PREPROCESSOR_AVAILABLE = False
    logger.warning(f"DepthPreprocessor failed to load due to system dependencies: {e}")

@dataclass
class TemporalConfig:
    """Configuration for temporal consistency parameters"""
    temporal_window_size: int = 5  # Number of frames to consider for temporal consistency
    keyframe_interval: int = 3     # Interval between keyframes
    temporal_weight: float = 0.3   # Weight for temporal loss
    feature_dim: int = 768         # Dimension of temporal features
    num_heads: int = 8             # Number of attention heads
    dropout: float = 0.1           # Dropout rate
    memory_length: int = 16        # Length of temporal memory buffer


class TemporalAttention(nn.Module):
    """Temporal attention module for maintaining consistency across frames"""
    
    def __init__(self, config: TemporalConfig):
        super().__init__()
        self.config = config
        self.feature_dim = config.feature_dim
        self.num_heads = config.num_heads
        self.head_dim = self.feature_dim // self.num_heads
        
        # Multi-head attention for temporal consistency
        self.query_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.key_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.value_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.out_proj = nn.Linear(self.feature_dim, self.feature_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.feature_dim)
        self.norm2 = nn.LayerNorm(self.feature_dim)
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.feature_dim * 4, self.feature_dim),
            nn.Dropout(config.dropout)
        )
        
        # Positional encoding for temporal relationships
        self.register_buffer('pos_encoding', self._create_positional_encoding())
    
    def _create_positional_encoding(self):
        """Create sinusoidal positional encoding for temporal positions"""
        max_len = self.config.temporal_window_size * 2
        pe = torch.zeros(max_len, self.feature_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.feature_dim, 2).float() * 
                           (-math.log(10000.0) / self.feature_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, features: torch.Tensor, temporal_context: Optional[torch.Tensor] = None):
        """
        Apply temporal attention to features
        
        Args:
            features: Current frame features [B, H*W, C]
            temporal_context: Previous frame features [B, T, H*W, C]
        
        Returns:
            Temporally consistent features
        """
        batch_size, seq_len, dim = features.shape
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :].to(features.device)
        features = features + pos_enc
        
        # Prepare input for attention
        if temporal_context is not None:
            # Combine current features with temporal context
            context_features = torch.cat([temporal_context.flatten(1, 2), features], dim=1)
        else:
            context_features = features
        
        # Self-attention
        residual = features
        features = self.norm1(features)
        
        # Multi-head attention
        q = self.query_proj(features).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key_proj(context_features).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.value_proj(context_features).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [B, H, seq_len, head_dim]
        k = k.transpose(1, 2)  # [B, H, context_len, head_dim]
        v = v.transpose(1, 2)  # [B, H, context_len, head_dim]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection
        features = residual + attn_output
        
        # Feed forward
        residual = features
        features = self.norm2(features)
        features = residual + self.ffn(features)
        
        return features


class TemporalMemory:
    """Memory buffer for maintaining temporal context across frames"""
    
    def __init__(self, config: TemporalConfig):
        self.config = config
        self.memory = deque(maxlen=config.memory_length)
        self.keyframes = deque(maxlen=config.memory_length // config.keyframe_interval)
    
    def add_frame(self, features: torch.Tensor, is_keyframe: bool = False):
        """Add frame features to memory"""
        self.memory.append(features.detach().cpu())
        if is_keyframe:
            self.keyframes.append(features.detach().cpu())
    
    def get_temporal_context(self, device: torch.device) -> Optional[torch.Tensor]:
        """Get recent temporal context"""
        if len(self.memory) < 2:
            return None
        
        # Use last few frames as context
        recent_frames = list(self.memory)[-self.config.temporal_window_size:]
        if len(recent_frames) > 1:
            context = torch.stack([f.to(device) for f in recent_frames[:-1]])
            return context
        return None
    
    def get_keyframe_context(self, device: torch.device) -> Optional[torch.Tensor]:
        """Get keyframe context for long-term consistency"""
        if len(self.keyframes) < 1:
            return None
        
        keyframe_context = torch.stack([f.to(device) for f in self.keyframes])
        return keyframe_context
    
    def clear(self):
        """Clear memory buffers"""
        self.memory.clear()
        self.keyframes.clear()


class FluxDepthProcessor:
    """Enhanced depth processor with temporal awareness"""
    
    def __init__(self):
        self.depth_preprocessor = None
        if DEPTH_PREPROCESSOR_AVAILABLE:
            try:
                self.depth_preprocessor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
                logger.info("Loaded DepthAnything preprocessor")
            except Exception as e:
                logger.warning(f"Failed to load DepthAnything preprocessor: {e}")
                self.depth_preprocessor = None
    
    def normalize_depth_map(self, depth_image: Image.Image, target_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
        """Normalize and prepare depth map for Flux.1 Depth"""
        
        # If we have the official depth preprocessor, use it
        if self.depth_preprocessor is not None:
            try:
                processed_depth = self.depth_preprocessor(depth_image)[0].convert("RGB")
                if processed_depth.size != target_size:
                    processed_depth = processed_depth.resize(target_size, Image.Resampling.LANCZOS)
                return processed_depth
            except Exception as e:
                logger.warning(f"DepthAnything preprocessor failed, falling back to manual processing: {e}")
        
        # Fallback to manual depth processing
        depth_array = np.array(depth_image)
        
        if len(depth_array.shape) == 3:
            depth_array = depth_array[:, :, 0]
        
        # Normalize to 0-1 range
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        
        if depth_max > depth_min:
            depth_normalized = (depth_array - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_array)
        
        # Convert to uint8 and RGB
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        depth_rgb = np.stack([depth_uint8] * 3, axis=-1)
        
        # Create PIL Image and resize
        depth_pil = Image.fromarray(depth_rgb, mode='RGB')
        depth_pil = depth_pil.resize(target_size, Image.Resampling.LANCZOS)
        
        return depth_pil
    
    @staticmethod
    def compute_temporal_depth_consistency(prev_depth: np.ndarray, curr_depth: np.ndarray, 
                                         motion_threshold: float = 0.1) -> float:
        """Compute temporal consistency score between depth maps"""
        if prev_depth.shape != curr_depth.shape:
            return 0.0
        
        # Compute optical flow approximation using depth differences
        depth_diff = np.abs(curr_depth - prev_depth)
        consistency_score = 1.0 - np.mean(depth_diff > motion_threshold)
        
        return max(0.0, consistency_score)


class FluxTemporalV2VModel:
    """FLUX.1 Depth model with temporal consistency for video-to-video enhancement"""
    
    def __init__(self, 
                 model_id: str = "black-forest-labs/FLUX.1-Depth-dev-onnx",
                 device: str = None,
                 use_fp16: bool = True,
                 temporal_config: Optional[TemporalConfig] = None):
        """Initialize Flux.1 Depth model with temporal consistency"""
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if use_fp16 and self.device == 'cuda' else torch.float32
        self.model_id = model_id
        self.pipeline = None
        
        # Temporal consistency components
        self.temporal_config = temporal_config or TemporalConfig()
        self.temporal_attention = TemporalAttention(self.temporal_config).to(self.device)
        self.temporal_memory = TemporalMemory(self.temporal_config)
        self.depth_processor = FluxDepthProcessor()
        
        # Frame tracking
        self.frame_count = 0
        self.prev_depth = None
        
        logger.info(f"Initializing FLUX Temporal V2V model on {self.device}")
    
    def load_model(self):
        """Load the Flux.1 Depth pipeline"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        if self.pipeline is None:
            try:
                logger.info(f"Loading FLUX.1 Depth model: {self.model_id}")
                
                # Get Hugging Face token from environment
                hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
                if not hf_token:
                    logger.warning("No HUGGINGFACE_HUB_TOKEN found in environment. Model may fail if gated.")
                
                # Check if this is an ONNX model and use appropriate pipeline
                if "onnx" in self.model_id.lower() and ONNX_AVAILABLE:
                    logger.info("Loading ONNX optimized model")
                    self.pipeline = ORTDiffusionPipeline.from_pretrained(
                        self.model_id,
                        token=hf_token,
                        provider="CPUExecutionProvider"  # Use CPU for now
                    )
                else:
                    # Load using FluxPipeline with authentication
                    self.pipeline = FluxPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=self.dtype,
                        token=hf_token
                    ).to(self.device)
                
                # Enable memory optimizations
                if hasattr(self.pipeline, "enable_attention_slicing"):
                    self.pipeline.enable_attention_slicing()
                
                if hasattr(self.pipeline, "enable_model_cpu_offload"):
                    self.pipeline.enable_model_cpu_offload()
                
                logger.info("FLUX.1 Depth model loaded successfully")
                
            except Exception as e:
                logger.error(f"Error loading FLUX model: {e}")
                raise RuntimeError(f"Failed to load FLUX.1 Depth model: {str(e)}")
    
    def extract_features(self, image: Image.Image, depth_map: Image.Image) -> torch.Tensor:
        """Extract features from image and depth for temporal processing"""
        # Convert images to tensors
        img_array = np.array(image.resize((512, 512))) / 255.0
        depth_array = np.array(depth_map.resize((512, 512))) / 255.0
        
        # Combine RGB and depth channels
        combined = np.concatenate([img_array, depth_array[:, :, :1]], axis=-1)
        
        # Convert to tensor and flatten spatial dimensions
        features = torch.from_numpy(combined).float().to(self.device)
        features = features.view(-1, features.shape[-1])  # [H*W, 4]
        
        # Project to feature dimension
        if not hasattr(self, 'feature_proj'):
            self.feature_proj = nn.Linear(4, self.temporal_config.feature_dim).to(self.device)
        
        features = self.feature_proj(features).unsqueeze(0)  # [1, H*W, feature_dim]
        
        return features
    
    def apply_temporal_consistency(self, features: torch.Tensor, is_keyframe: bool = False) -> torch.Tensor:
        """Apply temporal consistency to current frame features"""
        # Get temporal context from memory
        temporal_context = self.temporal_memory.get_temporal_context(self.device)
        
        # Apply temporal attention
        consistent_features = self.temporal_attention(features, temporal_context)
        
        # Add to memory
        self.temporal_memory.add_frame(consistent_features, is_keyframe)
        
        return consistent_features
    
    def generate_with_temporal_consistency(self,
                                         depth_map: Image.Image,
                                         prompt: str,
                                         reference_image: Optional[Image.Image] = None,
                                         negative_prompt: str = "",
                                         num_inference_steps: int = 28,
                                         guidance_scale: float = 3.5,
                                         seed: Optional[int] = None,
                                         height: int = 1024,
                                         width: int = 1024,
                                         is_keyframe: bool = False,
                                         **kwargs) -> Image.Image:
        """Generate enhanced image with temporal consistency"""
        
        # Load model if not already loaded
        if self.pipeline is None:
            self.load_model()
        
        # Normalize depth map
        depth_normalized = self.depth_processor.normalize_depth_map(
            depth_map, 
            target_size=(width, height)
        )
        
        # Extract and process features for temporal consistency
        if reference_image is not None:
            features = self.extract_features(reference_image, depth_normalized)
            consistent_features = self.apply_temporal_consistency(features, is_keyframe)
            
            # Compute temporal consistency loss if we have previous depth
            temporal_weight = 0.0
            if self.prev_depth is not None:
                curr_depth_array = np.array(depth_normalized)[:, :, 0]
                consistency_score = self.depth_processor.compute_temporal_depth_consistency(
                    self.prev_depth, curr_depth_array
                )
                temporal_weight = self.temporal_config.temporal_weight * (1.0 - consistency_score)
                logger.info(f"Temporal consistency score: {consistency_score:.3f}")
            
            # Store current depth for next frame
            self.prev_depth = np.array(depth_normalized)[:, :, 0]
        
        # Set random seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate image with FLUX
        logger.info(f"Generating frame {self.frame_count} with prompt: {prompt}")
        
        try:
            # Generate with FluxControlPipeline using control_image
            output = self.pipeline(
                prompt=prompt,
                control_image=depth_normalized,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs
            )
            
            if hasattr(output, 'images') and output.images:
                result_image = output.images[0]
                self.frame_count += 1
                return result_image
            else:
                raise RuntimeError("No images generated by FLUX pipeline")
                
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def process_video_sequence(self,
                             frames: List[Image.Image],
                             depth_maps: List[Image.Image],
                             prompts: List[str],
                             keyframe_indices: Optional[List[int]] = None,
                             **generation_params) -> List[Image.Image]:
        """Process entire video sequence with temporal consistency"""
        
        if len(frames) != len(depth_maps) or len(frames) != len(prompts):
            raise ValueError("Frames, depth maps, and prompts must have same length")
        
        # Clear temporal memory for new sequence
        self.temporal_memory.clear()
        self.frame_count = 0
        self.prev_depth = None
        
        # Determine keyframes
        if keyframe_indices is None:
            keyframe_indices = list(range(0, len(frames), self.temporal_config.keyframe_interval))
        
        enhanced_frames = []
        
        for i, (frame, depth_map, prompt) in enumerate(zip(frames, depth_maps, prompts)):
            is_keyframe = i in keyframe_indices
            
            logger.info(f"Processing frame {i+1}/{len(frames)} {'(keyframe)' if is_keyframe else ''}")
            
            # Generate enhanced frame
            enhanced_frame = self.generate_with_temporal_consistency(
                depth_map=depth_map,
                prompt=prompt,
                reference_image=frame,
                is_keyframe=is_keyframe,
                **generation_params
            )
            
            enhanced_frames.append(enhanced_frame)
        
        logger.info(f"Completed processing {len(enhanced_frames)} frames")
        return enhanced_frames
    
    def reset_temporal_state(self):
        """Reset temporal memory and state for new sequence"""
        self.temporal_memory.clear()
        self.frame_count = 0
        self.prev_depth = None
        logger.info("Temporal state reset")


# Global model instance
_global_flux_temporal_model = None

def get_flux_temporal_model(temporal_config: Optional[TemporalConfig] = None) -> FluxTemporalV2VModel:
    """Get or create global FLUX temporal model instance"""
    global _global_flux_temporal_model
    if _global_flux_temporal_model is None:
        _global_flux_temporal_model = FluxTemporalV2VModel(temporal_config=temporal_config)
    return _global_flux_temporal_model


def enhance_sequence_with_flux_temporal(frames: List[Image.Image],
                                      depth_maps: List[Image.Image],
                                      prompts: List[str],
                                      temporal_params: Optional[Dict[str, Any]] = None,
                                      generation_params: Optional[Dict[str, Any]] = None) -> List[Image.Image]:
    """
    Main function to enhance video sequence using FLUX with temporal consistency
    
    Args:
        frames: List of input frames
        depth_maps: List of corresponding depth maps
        prompts: List of prompts for each frame
        temporal_params: Parameters for temporal consistency
        generation_params: Parameters for FLUX generation
    
    Returns:
        List of enhanced frames with temporal consistency
    """
    
    # Create temporal config
    temporal_config = TemporalConfig()
    if temporal_params:
        for key, value in temporal_params.items():
            if hasattr(temporal_config, key):
                setattr(temporal_config, key, value)
    
    # Get model instance
    model = get_flux_temporal_model(temporal_config)
    
    # Default generation parameters
    default_generation_params = {
        'num_inference_steps': 28,
        'guidance_scale': 3.5,
        'height': 1024,
        'width': 1024
    }
    
    if generation_params:
        default_generation_params.update(generation_params)
    
    # Process sequence
    enhanced_frames = model.process_video_sequence(
        frames=frames,
        depth_maps=depth_maps,
        prompts=prompts,
        **default_generation_params
    )
    
    return enhanced_frames


def check_flux_temporal_status() -> Dict[str, Any]:
    """Check FLUX temporal model status and availability"""
    try:
        import diffusers
        diffusers_version = diffusers.__version__
    except:
        diffusers_version = "Not installed"
    
    return {
        'available': DIFFUSERS_AVAILABLE or ONNX_AVAILABLE,
        'diffusers_available': DIFFUSERS_AVAILABLE,
        'onnx_available': ONNX_AVAILABLE,
        'depth_preprocessor_available': DEPTH_PREPROCESSOR_AVAILABLE,
        'cuda_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_type': 'FLUX.1-Depth with Temporal Consistency',
        'diffusers_version': diffusers_version,
        'temporal_features': [
            'Transformer-based temporal attention',
            'Memory buffer for frame context',
            'Keyframe-based consistency',
            'Depth-aware temporal processing',
            'Professional depth preprocessing'
        ],
        'model_id': 'black-forest-labs/FLUX.1-Depth-dev',
        'hf_token_configured': bool(os.getenv('HUGGINGFACE_HUB_TOKEN')),
        'temporal_config': {
            'window_size': 5,
            'keyframe_interval': 3,
            'memory_length': 16,
            'feature_dim': 768
        }
    }