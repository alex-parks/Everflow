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
    width: int = 512
    height: int = 512
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
            # Fallback to placeholder images
            if channel == "beauty":
                img = Image.new('RGB', (1920, 1080), color=(60, 60, 80))
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 48)
                except:
                    font = ImageFont.load_default()
                draw.text((50, 50), f"Beauty Pass\nFrame {frame_num + 1}\nEXR Loading...", fill=(255, 255, 255), font=font)
            else:  # depth
                img = Image.new('RGB', (1920, 1080), color=(30, 30, 30))
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 48)
                except:
                    font = ImageFont.load_default()
                draw.text((50, 50), f"Depth Map\nFrame {frame_num + 1}\nEXR Loading...", fill=(200, 200, 200), font=font)
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=90)
            return Response(content=output.getvalue(), media_type="image/jpeg")
    
    # Handle regular image formats
    try:
        with Image.open(frame_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=90)
            return Response(content=output.getvalue(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading image: {str(e)}")


def load_exr_beauty_pass(exr_path):
    """Load RGB beauty pass from EXR file"""
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
        rgb_channels = exr_file.channels(['R', 'G', 'B'], FLOAT)
        
        # Convert to numpy arrays
        r = np.frombuffer(rgb_channels[0], dtype=np.float32).reshape(height, width)
        g = np.frombuffer(rgb_channels[1], dtype=np.float32).reshape(height, width)
        b = np.frombuffer(rgb_channels[2], dtype=np.float32).reshape(height, width)
        
        beauty = np.stack([r, g, b], axis=2)
        exr_file.close()
        
        # Apply simple tone mapping
        beauty = beauty / (1.0 + beauty)  # Reinhard tone mapping
        beauty = np.power(np.clip(beauty, 0, 1), 1.0 / 2.2)  # Gamma correction
        
        return beauty
        
    except Exception as e:
        # Fallback: create a placeholder
        print(f"Error loading beauty pass: {e}")
        return np.full((1080, 1920, 3), 0.3, dtype=np.float32)


def load_exr_depth_channel(exr_path):
    """Load depth channel from EXR file"""
    try:
        import OpenEXR
        import Imath
        
        exr_file = OpenEXR.InputFile(str(exr_path))
        header = exr_file.header()
        
        # Get image dimensions
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # Try different depth channel names
        depth_names = ['Z', 'depth', 'Depth', 'DEPTH']
        depth = None
        
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        for depth_name in depth_names:
            try:
                depth_channel = exr_file.channel(depth_name, FLOAT)
                depth = np.frombuffer(depth_channel, dtype=np.float32).reshape(height, width)
                break
            except:
                continue
        
        exr_file.close()
        
        if depth is None:
            raise ValueError("No depth channel found")
        
        # Handle infinite values
        depth = np.where(np.isinf(depth), depth[np.isfinite(depth)].max(), depth)
        
        return depth
        
    except Exception as e:
        # Fallback: create a placeholder gradient
        print(f"Error loading depth channel: {e}")
        height, width = 1080, 1920
        y, x = np.ogrid[:height, :width]
        depth = (y + x) / (height + width)
        return depth.astype(np.float32)

@router.get("/{sequence_id}/info")
async def get_sequence_info(sequence_id: str):
    """Get information about a sequence"""
    
    sequence_dir = UPLOAD_DIR / sequence_id
    if not sequence_dir.exists():
        raise HTTPException(status_code=404, detail="Sequence not found")
    
    frame_files = sorted(sequence_dir.glob("frame_*"))
    
    return {
        "frame_count": len(frame_files),
        "frames": [f.name for f in frame_files[:10]]  # First 10 frame names
    }

@router.post("/{sequence_id}/enhance")
async def enhance_sequence(sequence_id: str, background_tasks: BackgroundTasks):
    """Enhance an EXR sequence using Simple V2V model"""
    
    sequence_dir = UPLOAD_DIR / sequence_id
    if not sequence_dir.exists():
        raise HTTPException(status_code=404, detail="Sequence not found")
    
    # Check V2V model availability
    status = check_v2v_status()
    if not status.get('available', False):
        raise HTTPException(
            status_code=503, 
            detail="V2V model not available."
        )
    
    # Find frame files (EXR and other formats)
    frame_files = sorted(sequence_dir.glob("frame_*"))
    if not frame_files:
        raise HTTPException(status_code=400, detail="No frames found in sequence")
    
    # Create output directory
    output_dir = sequence_dir / "enhanced_v2v"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Simple enhancement - just create enhanced versions
        from simple_v2v import enhance_image
        enhanced_files = []
        
        for i, frame_file in enumerate(frame_files):
            if frame_file.suffix.lower() == '.exr':
                # Load beauty pass for EXR files
                beauty_data = load_exr_beauty_pass(frame_file)
                img_array = (np.clip(beauty_data, 0, 1) * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
            else:
                # Load regular image files
                img = Image.open(frame_file).convert('RGB')
            
            # Enhance the image
            enhanced_img = enhance_image(img)
            
            # Save enhanced image
            output_file = output_dir / f"enhanced_frame_{i:04d}.jpg"
            enhanced_img.save(output_file, quality=95)
            enhanced_files.append(str(output_file))
        
        return {
            "message": "V2V enhancement completed",
            "input_frames": len(frame_files),
            "enhanced_frames": len(enhanced_files),
            "output_dir": str(output_dir),
            "enhanced_files": enhanced_files[:10]  # First 10 for preview
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Enhancement failed: {str(e)}"
        )

@router.get("/{sequence_id}/enhanced/{frame_num}")
async def get_enhanced_frame(sequence_id: str, frame_num: int):
    """Get an enhanced frame from V2V processing"""
    
    sequence_dir = UPLOAD_DIR / sequence_id
    enhanced_dir = sequence_dir / "enhanced_v2v"
    
    if not enhanced_dir.exists():
        raise HTTPException(status_code=404, detail="Enhanced sequence not found")
    
    # Find enhanced frame files
    enhanced_files = sorted(enhanced_dir.glob("enhanced_frame_*.jpg"))
    
    if frame_num >= len(enhanced_files):
        raise HTTPException(status_code=404, detail="Enhanced frame not found")
    
    enhanced_path = enhanced_files[frame_num]
    
    try:
        # Load and return enhanced image
        with Image.open(enhanced_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=90)
            return Response(content=output.getvalue(), media_type="image/jpeg")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading enhanced frame: {str(e)}")

@router.get("/v2v/status")
async def get_v2v_status_endpoint():
    """Get Simple V2V model status and capabilities"""
    return check_v2v_status()

@router.get("/sd-controlnet/status")
async def get_sd_controlnet_status_endpoint():
    """Get SD + ControlNet model status and capabilities"""
    return check_sd_controlnet_status()


@router.post("/{sequence_id}/enhance-flux")
async def enhance_sequence_flux(
    sequence_id: str, 
    params: SDControlNetParams,
    background_tasks: BackgroundTasks
):
    """Enhance an EXR sequence using Flux.1 Depth model"""
    
    sequence_dir = UPLOAD_DIR / sequence_id
    if not sequence_dir.exists():
        raise HTTPException(status_code=404, detail="Sequence not found")
    
    # Check Flux model availability
    status = check_flux_status()
    if not status.get('available', False):
        raise HTTPException(
            status_code=503, 
            detail="Flux model not available."
        )
    
    # Find frame files
    frame_files = sorted(sequence_dir.glob("frame_*"))
    if not frame_files:
        raise HTTPException(status_code=400, detail="No frames found in sequence")
    
    # Create output directory
    output_dir = sequence_dir / "enhanced_flux"
    output_dir.mkdir(exist_ok=True)
    
    try:
        enhanced_files = []
        
        for i, frame_file in enumerate(frame_files):
            # Load beauty pass
            if frame_file.suffix.lower() == '.exr':
                beauty_data = load_exr_beauty_pass(frame_file)
                img_array = (np.clip(beauty_data, 0, 1) * 255).astype(np.uint8)
                beauty_img = Image.fromarray(img_array)
                
                # Load depth if requested
                depth_img = None
                if params.use_depth:
                    depth_data = load_exr_depth_channel(frame_file)
                    depth_processor = FluxDepthProcessor()
                    depth_img = depth_processor.normalize_depth_map(
                        Image.fromarray((depth_data * 255).astype(np.uint8)),
                        target_size=(params.width, params.height)
                    )
            else:
                beauty_img = Image.open(frame_file).convert('RGB')
                depth_img = None
            
            # Enhance with Flux
            enhanced_img = enhance_with_flux(
                input_image=beauty_img,
                depth_map=depth_img,
                params=params.dict(exclude={'use_depth'})
            )
            
            # Save enhanced image
            output_file = output_dir / f"flux_enhanced_frame_{i:04d}.jpg"
            enhanced_img.save(output_file, quality=95)
            enhanced_files.append(str(output_file))
        
        return {
            "message": "Flux.1 Depth enhancement completed",
            "input_frames": len(frame_files),
            "enhanced_frames": len(enhanced_files),
            "output_dir": str(output_dir),
            "enhanced_files": enhanced_files[:10],
            "parameters": params.dict()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Flux enhancement failed: {str(e)}"
        )

@router.get("/{sequence_id}/enhanced-flux/{frame_num}")
async def get_flux_enhanced_frame(sequence_id: str, frame_num: int):
    """Get a Flux-enhanced frame"""
    
    sequence_dir = UPLOAD_DIR / sequence_id
    enhanced_dir = sequence_dir / "enhanced_flux"
    
    if not enhanced_dir.exists():
        raise HTTPException(status_code=404, detail="Flux enhanced sequence not found")
    
    # Find enhanced frame files
    enhanced_files = sorted(enhanced_dir.glob("flux_enhanced_frame_*.jpg"))
    
    if frame_num >= len(enhanced_files):
        raise HTTPException(status_code=404, detail="Flux enhanced frame not found")
    
    enhanced_path = enhanced_files[frame_num]
    
    try:
        with Image.open(enhanced_path) as img:
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=90)
            return Response(content=output.getvalue(), media_type="image/jpeg")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading flux enhanced frame: {str(e)}")

@router.post("/{sequence_id}/enhance-flux-single")
async def enhance_single_frame_flux(
    sequence_id: str,
    frame_num: int,
    params: SDControlNetParams
):
    """Enhance a single frame using Flux.1 Depth model for testing"""
    
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
            
            depth_img = None
            if params.use_depth:
                depth_data = load_exr_depth_channel(frame_file)
                depth_processor = FluxDepthProcessor()
                depth_img = depth_processor.normalize_depth_map(
                    Image.fromarray((depth_data * 255).astype(np.uint8)),
                    target_size=(params.width, params.height)
                )
        else:
            beauty_img = Image.open(frame_file).convert('RGB')
            depth_img = None
        
        # Enhance with Flux
        enhanced_img = enhance_with_flux(
            input_image=beauty_img,
            depth_map=depth_img,
            params=params.dict(exclude={'use_depth'})
        )
        
        # Return as JPEG
        output = io.BytesIO()
        enhanced_img.save(output, format='JPEG', quality=95)
        return Response(
            content=output.getvalue(), 
            media_type="image/jpeg",
            headers={"X-Enhancement-Method": "flux"}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Single frame Flux enhancement failed: {str(e)}"
        )

# ===== FLUX TEMPORAL CONSISTENCY ENDPOINTS =====

@router.get("/sd-controlnet-temporal/status")
async def get_sd_controlnet_temporal_status_endpoint():
    """Get SD + ControlNet Temporal model status and capabilities"""
    return check_sd_controlnet_status()

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
                    depth_data = load_exr_depth_channel(frame_file)
                    depth_img = Image.fromarray((depth_data * 255).astype(np.uint8))
                    depth_maps.append(depth_img)
                else:
                    # Create dummy depth map
                    dummy_depth = np.full((beauty_img.height, beauty_img.width), 128, dtype=np.uint8)
                    depth_maps.append(Image.fromarray(dummy_depth))
            else:
                # Load regular image files
                beauty_img = Image.open(frame_file).convert('RGB')
                frames.append(beauty_img)
                
                # Create dummy depth map for non-EXR files
                dummy_depth = np.full((beauty_img.height, beauty_img.width), 128, dtype=np.uint8)
                depth_maps.append(Image.fromarray(dummy_depth))
            
            # Use same prompt for all frames (could be made frame-specific)
            prompts.append(params.prompt)
        
        # Create temporal config
        temporal_config = {
            'temporal_window_size': params.temporal_window_size,
            'keyframe_interval': params.keyframe_interval,
            'temporal_weight': params.temporal_weight,
            'memory_length': params.memory_length
        }
        
        # Generation parameters
        generation_params = {
            'num_inference_steps': params.num_inference_steps,
            'guidance_scale': params.guidance_scale,
            'controlnet_conditioning_scale': params.controlnet_conditioning_scale,
            'negative_prompt': params.negative_prompt,
            'height': params.height,
            'width': params.width,
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
            detail=f"Flux temporal enhancement failed: {str(e)}"
        )

@router.get("/{sequence_id}/enhanced-flux-temporal/{frame_num}")
async def get_flux_temporal_enhanced_frame(sequence_id: str, frame_num: int):
    """Get a Flux temporal enhanced frame"""
    
    sequence_dir = UPLOAD_DIR / sequence_id
    enhanced_dir = sequence_dir / "enhanced_flux_temporal"
    
    if not enhanced_dir.exists():
        raise HTTPException(status_code=404, detail="Flux temporal enhanced sequence not found")
    
    # Find enhanced frame files
    enhanced_files = sorted(enhanced_dir.glob("flux_temporal_enhanced_frame_*.jpg"))
    
    if frame_num >= len(enhanced_files):
        raise HTTPException(status_code=404, detail="Flux temporal enhanced frame not found")
    
    enhanced_path = enhanced_files[frame_num]
    
    try:
        with Image.open(enhanced_path) as img:
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=90)
            return Response(
                content=output.getvalue(), 
                media_type="image/jpeg",
                headers={"X-Enhancement-Method": "flux-temporal"}
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading flux temporal enhanced frame: {str(e)}")

@router.post("/{sequence_id}/enhance-flux-temporal-single")
async def enhance_single_frame_flux_temporal(
    sequence_id: str,
    frame_num: int,
    params: SDControlNetParams
):
    """Enhance a single frame using Flux.1 Temporal for testing"""
    
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
                depth_data = load_exr_depth_channel(frame_file)
                depth_img = Image.fromarray((depth_data * 255).astype(np.uint8))
            else:
                dummy_depth = np.full((beauty_img.height, beauty_img.width), 128, dtype=np.uint8)
                depth_img = Image.fromarray(dummy_depth)
        else:
            beauty_img = Image.open(frame_file).convert('RGB')
            dummy_depth = np.full((beauty_img.height, beauty_img.width), 128, dtype=np.uint8)
            depth_img = Image.fromarray(dummy_depth)
        
        # Create temporal config
        temporal_config = {
            'temporal_window_size': params.temporal_window_size,
            'keyframe_interval': params.keyframe_interval,
            'temporal_weight': params.temporal_weight,
            'memory_length': params.memory_length
        } if params.enable_temporal else None
        
        # Generation parameters
        generation_params = {
            'num_inference_steps': params.num_inference_steps,
            'guidance_scale': params.guidance_scale,
            'controlnet_conditioning_scale': params.controlnet_conditioning_scale,
            'negative_prompt': params.negative_prompt,
            'height': params.height,
            'width': params.width,
            'seed': params.seed
        }
        
        # Process single frame (as sequence of 1)
        enhanced_frames = enhance_sequence_with_flux_temporal(
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
            headers={"X-Enhancement-Method": "flux-temporal"}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Single frame Flux temporal enhancement failed: {str(e)}"
        )