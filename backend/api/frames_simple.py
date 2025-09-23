from fastapi import APIRouter, HTTPException, Response, BackgroundTasks
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import sys
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel

# Add parent directory to Python path to import models
sys.path.append(str(Path(__file__).parent.parent))
from simple_v2v import check_v2v_status
from sd_controlnet_v2v import check_sd_controlnet_status, enhance_sequence_with_sd_controlnet, TemporalConfig

router = APIRouter()

UPLOAD_DIR = Path("/app/uploads/crowd_sequences")

class SDControlNetParams(BaseModel):
    """Parameters for Stable Diffusion + ControlNet Depth enhancement"""
    prompt: str = "Enhanced cinematic quality, professional VFX, high detail, photorealistic"
    negative_prompt: str = "blurry, low quality, distorted, deformed"
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 1.0
    seed: Optional[int] = None
    width: int = 768
    height: int = 768
    use_depth: bool = True
    # Temporal consistency parameters
    temporal_window_size: int = 5
    keyframe_interval: int = 3
    temporal_weight: float = 0.3
    memory_length: int = 16
    enable_temporal: bool = True

@router.get("/{sequence_id}/frame/{frame_num}")
async def get_frame(sequence_id: str, frame_num: int, channel: str = "beauty", exposure: float = 0.0):
    """Get a specific frame as JPEG with support for beauty and depth channels"""
    
    sequence_dir = UPLOAD_DIR / sequence_id
    if not sequence_dir.exists():
        raise HTTPException(status_code=404, detail="Sequence not found")
    
    # Find frame files
    frame_files = sorted(sequence_dir.glob("frame_*"))
    
    if frame_num >= len(frame_files):
        raise HTTPException(status_code=404, detail="Frame not found")
    
    frame_path = frame_files[frame_num]
    
    # Handle EXR files with beauty and depth channel extraction
    if frame_path.suffix.lower() == '.exr':
        try:
            # Try to load and process EXR file
            from simple_v2v import EXRProcessor
            
            if channel == "beauty":
                # Load beauty pass from EXR
                exr_data = load_exr_beauty_pass(frame_path)
                img_array = (np.clip(exr_data, 0, 1) * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                
            elif channel == "depth":
                # Load depth map from EXR
                depth_data = load_exr_depth_channel(frame_path)
                # Normalize depth to 0-1 range and convert to grayscale image
                depth_normalized = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min() + 1e-8)
                depth_array = (depth_normalized * 255).astype(np.uint8)
                img = Image.fromarray(depth_array, mode='L').convert('RGB')
                
            else:
                raise HTTPException(status_code=400, detail=f"Unknown channel: {channel}")
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=90)
            return Response(content=output.getvalue(), media_type="image/jpeg")
            
        except Exception as e:
            print(f"Error loading EXR frame: {e}")
            # Fallback to placeholder
            img = Image.new('RGB', (1920, 1080), color=(50, 50, 50))
            draw = ImageDraw.Draw(img)
            draw.text((960, 540), f"Error loading {channel} channel", fill=(255, 255, 255), anchor="mm")
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=90)
            return Response(content=output.getvalue(), media_type="image/jpeg")
    
    else:
        # Handle regular image files
        try:
            with Image.open(frame_path) as img:
                # Apply exposure adjustment if requested
                if exposure != 0.0:
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_array = np.clip(img_array * (2 ** exposure), 0, 1)
                    img_array = (img_array * 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
                
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=90)
                return Response(content=output.getvalue(), media_type="image/jpeg")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading frame: {str(e)}")

def load_exr_beauty_pass(exr_path):
    """Load beauty pass from EXR file"""
    try:
        import OpenEXR
        import Imath
        
        exr_file = OpenEXR.InputFile(str(exr_path))
        header = exr_file.header()
        
        # Get image dimensions
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # Read RGB channels
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        
        rgb_channels = ['R', 'G', 'B']
        channels = []
        
        for channel_name in rgb_channels:
            try:
                channel_data = exr_file.channel(channel_name, FLOAT)
                channel_array = np.frombuffer(channel_data, dtype=np.float32).reshape(height, width)
                channels.append(channel_array)
            except:
                # Create dummy channel if not found
                channels.append(np.ones((height, width), dtype=np.float32) * 0.5)
        
        exr_file.close()
        
        # Stack channels and return
        rgb_array = np.stack(channels, axis=-1)
        return rgb_array
        
    except Exception as e:
        # Fallback: create a placeholder
        print(f"Error loading beauty pass: {e}")
        return np.full((1080, 1920, 3), 0.3, dtype=np.float32)

def load_exr_depth_channel(exr_path):
    """Load depth channel from EXR file with proper preprocessing for ControlNet"""
    try:
        import OpenEXR
        import Imath

        exr_file = OpenEXR.InputFile(str(exr_path))
        header = exr_file.header()

        # Get image dimensions
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # Try different depth channel names (Z is most common for depth)
        depth_names = ['Z', 'depth', 'Depth', 'DEPTH']
        depth = None

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        for depth_name in depth_names:
            try:
                depth_channel = exr_file.channel(depth_name, FLOAT)
                depth = np.frombuffer(depth_channel, dtype=np.float32).reshape(height, width)
                print(f"Successfully loaded depth channel: {depth_name}")
                break
            except:
                continue

        exr_file.close()

        if depth is None:
            raise ValueError("No depth channel found in EXR file")

        # Proper depth preprocessing for ControlNet (following Manus guide)

        # 1. Handle infinite and NaN values
        finite_mask = np.isfinite(depth)
        if not np.any(finite_mask):
            raise ValueError("No finite depth values found")

        # Replace inf/nan with max finite value
        max_finite_depth = np.max(depth[finite_mask])
        depth = np.where(finite_mask, depth, max_finite_depth)

        # 2. Normalize to 0-1 range (critical for ControlNet)
        min_depth = np.min(depth[finite_mask])
        max_depth = np.max(depth[finite_mask])

        if max_depth > min_depth:
            normalized_depth = (depth - min_depth) / (max_depth - min_depth)
        else:
            # If all depth values are the same, create a flat depth map
            normalized_depth = np.full_like(depth, 0.5)

        # 3. Clip to ensure [0, 1] range
        normalized_depth = np.clip(normalized_depth, 0.0, 1.0)

        print(f"Depth range: {min_depth:.3f} to {max_depth:.3f}, normalized to [0, 1]")

        return normalized_depth

    except Exception as e:
        # Fallback: create a proper normalized gradient for testing
        print(f"Error loading depth channel: {e}")
        height, width = 1080, 1920
        y, x = np.ogrid[:height, :width]
        depth = (y + x) / (height + width)
        # Normalize to 0-1 range
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        return depth.astype(np.float32)

@router.get("/{sequence_id}/info")
async def get_sequence_info(sequence_id: str):
    """Get information about a sequence"""
    
    sequence_dir = UPLOAD_DIR / sequence_id
    if not sequence_dir.exists():
        raise HTTPException(status_code=404, detail="Sequence not found")
    
    # Find frame files
    frame_files = sorted(sequence_dir.glob("frame_*"))
    
    if not frame_files:
        raise HTTPException(status_code=404, detail="No frames found in sequence")
    
    # Get info from first frame
    first_frame = frame_files[0]
    
    try:
        if first_frame.suffix.lower() == '.exr':
            # Get EXR metadata
            import OpenEXR
            exr_file = OpenEXR.InputFile(str(first_frame))
            header = exr_file.header()
            
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            channels = list(header['channels'].keys())
            exr_file.close()
            
            return {
                "sequence_id": sequence_id,
                "frame_count": len(frame_files),
                "width": width,
                "height": height,
                "format": "EXR",
                "channels": channels,
                "has_depth": any(ch in channels for ch in ['Z', 'depth', 'Depth', 'DEPTH'])
            }
        else:
            # Regular image file
            with Image.open(first_frame) as img:
                return {
                    "sequence_id": sequence_id,
                    "frame_count": len(frame_files),
                    "width": img.width,
                    "height": img.height,
                    "format": first_frame.suffix.upper()[1:],
                    "channels": ["R", "G", "B"] if img.mode == "RGB" else [img.mode],
                    "has_depth": False
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading sequence info: {str(e)}")

# ===== STATUS ENDPOINTS =====

@router.get("/v2v/status")
async def get_v2v_status_endpoint():
    """Get Simple V2V model status and capabilities"""
    return check_v2v_status()

@router.get("/sd-controlnet/status")
async def get_sd_controlnet_status_endpoint():
    """Get SD + ControlNet model status and capabilities"""
    return check_sd_controlnet_status()

@router.get("/sd-controlnet-temporal/status")
async def get_sd_controlnet_temporal_status_endpoint():
    """Get SD + ControlNet Temporal model status and capabilities"""
    return check_sd_controlnet_status()

# ===== SD CONTROLNET ENHANCEMENT ENDPOINTS =====

@router.post("/{sequence_id}/enhance-sd-controlnet")
async def enhance_sequence_sd_controlnet(
    sequence_id: str, 
    params: SDControlNetParams,
    background_tasks: BackgroundTasks
):
    """Enhance an EXR sequence using SD + ControlNet Depth with temporal consistency"""
    
    sequence_dir = UPLOAD_DIR / sequence_id
    if not sequence_dir.exists():
        raise HTTPException(status_code=404, detail="Sequence not found")
    
    # Check SD ControlNet model availability
    status = check_sd_controlnet_status()
    if not status.get('available', False):
        raise HTTPException(
            status_code=503, 
            detail="SD ControlNet model not available."
        )
    
    # Find frame files
    frame_files = sorted(sequence_dir.glob("frame_*"))
    if not frame_files:
        raise HTTPException(status_code=400, detail="No frames found in sequence")
    
    # Create output directory
    output_dir = sequence_dir / "enhanced_sd_controlnet"
    output_dir.mkdir(exist_ok=True)
    
    try:
        frames = []
        depth_maps = []
        prompts = []
        enhanced_files = []
        
        # Load all frames and depth maps
        for i, frame_file in enumerate(frame_files):
            if frame_file.suffix.lower() == '.exr':
                # Load beauty pass
                beauty_data = load_exr_beauty_pass(frame_file)
                img_array = (np.clip(beauty_data, 0, 1) * 255).astype(np.uint8)
                beauty_img = Image.fromarray(img_array)
                frames.append(beauty_img)
                
                # Load depth if requested
                if params.use_depth:
                    # Load properly normalized depth data (already 0-1 range)
                    depth_normalized = load_exr_depth_channel(frame_file)

                    # Convert to uint8 for ControlNet (0-255 range)
                    depth_uint8 = (depth_normalized * 255).astype(np.uint8)

                    # ControlNet depth expects RGB format (duplicate grayscale across channels)
                    depth_rgb = np.stack([depth_uint8] * 3, axis=-1)
                    depth_img = Image.fromarray(depth_rgb, mode='RGB')
                    depth_maps.append(depth_img)
                else:
                    # Create dummy depth map (RGB format, mid-gray = medium depth)
                    dummy_depth = np.full((beauty_img.height, beauty_img.width, 3), 128, dtype=np.uint8)
                    depth_maps.append(Image.fromarray(dummy_depth, mode='RGB'))
            else:
                # Load regular image files
                beauty_img = Image.open(frame_file).convert('RGB')
                frames.append(beauty_img)
                
                # Create dummy depth map for non-EXR files (RGB format)
                dummy_depth = np.full((beauty_img.height, beauty_img.width, 3), 128, dtype=np.uint8)
                depth_maps.append(Image.fromarray(dummy_depth, mode='RGB'))
            
            # Use the same prompt for all frames (could be customized per frame)
            prompts.append(params.prompt)
        
        # Temporal config
        temporal_config = {
            'temporal_window_size': params.temporal_window_size,
            'keyframe_interval': params.keyframe_interval,
            'temporal_weight': params.temporal_weight,
            'memory_length': params.memory_length
        }

        # Use actual image dimensions for exact resolution matching (per Manus guide)
        if frames:
            actual_height = frames[0].height
            actual_width = frames[0].width
            print(f"Using actual image dimensions: {actual_width}x{actual_height}")
        else:
            actual_height = params.height
            actual_width = params.width

        # Generation parameters
        generation_params = {
            'num_inference_steps': params.num_inference_steps,
            'guidance_scale': params.guidance_scale,
            'controlnet_conditioning_scale': params.controlnet_conditioning_scale,
            'negative_prompt': params.negative_prompt,
            'height': actual_height,
            'width': actual_width,
            'seed': params.seed
        }
        
        # Process sequence with temporal consistency
        enhanced_frames = enhance_sequence_with_sd_controlnet(
            frames=frames,
            depth_maps=depth_maps,
            prompts=prompts,
            temporal_params=temporal_config if params.enable_temporal else None,
            generation_params=generation_params
        )
        
        # Save enhanced frames
        for i, enhanced_frame in enumerate(enhanced_frames):
            output_file = output_dir / f"sd_controlnet_enhanced_frame_{i:04d}.jpg"
            enhanced_frame.save(output_file, quality=95)
            enhanced_files.append(str(output_file))
        
        return {
            "message": "SD + ControlNet enhancement completed",
            "input_frames": len(frame_files),
            "enhanced_frames": len(enhanced_files),
            "output_dir": str(output_dir),
            "enhanced_files": enhanced_files[:10],
            "parameters": params.dict(),
            "temporal_enabled": params.enable_temporal
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"SD ControlNet enhancement failed: {str(e)}"
        )

@router.get("/{sequence_id}/enhanced-sd-controlnet/{frame_num}")
async def get_sd_controlnet_enhanced_frame(sequence_id: str, frame_num: int):
    """Get an SD ControlNet enhanced frame"""
    
    sequence_dir = UPLOAD_DIR / sequence_id
    enhanced_dir = sequence_dir / "enhanced_sd_controlnet"
    
    if not enhanced_dir.exists():
        raise HTTPException(status_code=404, detail="SD ControlNet enhanced sequence not found")
    
    # Find enhanced frame files
    enhanced_files = sorted(enhanced_dir.glob("sd_controlnet_enhanced_frame_*.jpg"))
    
    if frame_num >= len(enhanced_files):
        raise HTTPException(status_code=404, detail="SD ControlNet enhanced frame not found")
    
    enhanced_path = enhanced_files[frame_num]
    
    try:
        with Image.open(enhanced_path) as img:
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=90)
            return Response(
                content=output.getvalue(), 
                media_type="image/jpeg",
                headers={"X-Enhancement-Method": "sd-controlnet"}
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading SD ControlNet enhanced frame: {str(e)}")

@router.post("/{sequence_id}/enhance-sd-controlnet-single")
async def enhance_single_frame_sd_controlnet(
    sequence_id: str,
    frame_num: int,
    params: SDControlNetParams
):
    """Enhance a single frame using SD + ControlNet for testing"""
    
    sequence_dir = UPLOAD_DIR / sequence_id
    if not sequence_dir.exists():
        raise HTTPException(status_code=404, detail="Sequence not found")
    
    # Find frame files
    frame_files = sorted(sequence_dir.glob("frame_*"))
    if frame_num >= len(frame_files):
        raise HTTPException(status_code=404, detail="Frame not found")
    
    frame_file = frame_files[frame_num]
    
    try:
        # Load beauty and depth
        if frame_file.suffix.lower() == '.exr':
            beauty_data = load_exr_beauty_pass(frame_file)
            img_array = (np.clip(beauty_data, 0, 1) * 255).astype(np.uint8)
            beauty_img = Image.fromarray(img_array)
            
            if params.use_depth:
                # Load properly normalized depth data (already 0-1 range)
                depth_normalized = load_exr_depth_channel(frame_file)

                # Convert to uint8 for ControlNet (0-255 range)
                depth_uint8 = (depth_normalized * 255).astype(np.uint8)

                # ControlNet depth expects RGB format (duplicate grayscale across channels)
                depth_rgb = np.stack([depth_uint8] * 3, axis=-1)
                depth_img = Image.fromarray(depth_rgb, mode='RGB')
            else:
                dummy_depth = np.full((beauty_img.height, beauty_img.width, 3), 128, dtype=np.uint8)
                depth_img = Image.fromarray(dummy_depth, mode='RGB')
        else:
            beauty_img = Image.open(frame_file).convert('RGB')
            dummy_depth = np.full((beauty_img.height, beauty_img.width, 3), 128, dtype=np.uint8)
            depth_img = Image.fromarray(dummy_depth, mode='RGB')
        
        # Create temporal config
        temporal_config = {
            'temporal_window_size': params.temporal_window_size,
            'keyframe_interval': params.keyframe_interval,
            'temporal_weight': params.temporal_weight,
            'memory_length': params.memory_length
        }

        # Use actual image dimensions for exact resolution matching (per Manus guide)
        actual_height = beauty_img.height
        actual_width = beauty_img.width
        print(f"Single frame using actual image dimensions: {actual_width}x{actual_height}")

        # Generation parameters
        generation_params = {
            'num_inference_steps': params.num_inference_steps,
            'guidance_scale': params.guidance_scale,
            'controlnet_conditioning_scale': params.controlnet_conditioning_scale,
            'negative_prompt': params.negative_prompt,
            'height': actual_height,
            'width': actual_width,
            'seed': params.seed
        }
        
        # Process single frame (as sequence of 1)
        enhanced_frames = enhance_sequence_with_sd_controlnet(
            frames=[beauty_img],
            depth_maps=[depth_img],
            prompts=[params.prompt],
            temporal_params=temporal_config,
            generation_params=generation_params
        )
        
        enhanced_img = enhanced_frames[0]
        
        # Return as JPEG
        output = io.BytesIO()
        enhanced_img.save(output, format='JPEG', quality=95)
        return Response(
            content=output.getvalue(), 
            media_type="image/jpeg",
            headers={"X-Enhancement-Method": "sd-controlnet"}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Single frame SD ControlNet enhancement failed: {str(e)}"
        )