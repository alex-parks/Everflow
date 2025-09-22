from fastapi import APIRouter, HTTPException, Response, BackgroundTasks
from pathlib import Path
from PIL import Image
import io
import numpy as np
import OpenImageIO as oiio
import asyncio
from cache_manager import frame_cache

router = APIRouter()

UPLOAD_DIR = Path("/app/uploads/sequences")

def read_exr_with_oiio(image_path: Path) -> np.ndarray:
    """Read EXR file using OpenImageIO and return as numpy array"""
    try:
        img_input = oiio.ImageInput.open(str(image_path))
        if not img_input:
            raise Exception(f"Could not open {image_path}")
        
        spec = img_input.spec()
        width, height = spec.width, spec.height
        channels = spec.nchannels
        
        pixel_data = img_input.read_image()
        img_input.close()
        
        if pixel_data is None:
            raise Exception("Failed to read pixel data")
        
        return np.array(pixel_data)
    except Exception as e:
        raise Exception(f"Error reading EXR with OpenImageIO: {str(e)}")

def tone_map_exr(hdr_data: np.ndarray, exposure: float = 0.0, gamma: float = 2.2) -> np.ndarray:
    """Apply simple tone mapping to HDR data"""
    # Apply exposure adjustment
    exposed = hdr_data * (2.0 ** exposure)
    
    # Simple Reinhard tone mapping
    tone_mapped = exposed / (1.0 + exposed)
    
    # Apply gamma correction
    gamma_corrected = np.power(np.clip(tone_mapped, 0, 1), 1.0 / gamma)
    
    # Convert to 8-bit
    return (gamma_corrected * 255).astype(np.uint8)

def convert_to_jpeg(image_path: Path, exposure: float = 0.0) -> bytes:
    """Convert any image format to JPEG bytes, with special handling for EXR"""
    try:
        file_ext = image_path.suffix.lower()
        
        if file_ext == '.exr':
            # Handle EXR files with OpenImageIO
            hdr_data = read_exr_with_oiio(image_path)
            
            # Ensure we have RGB data (take first 3 channels if more exist)
            if len(hdr_data.shape) == 3 and hdr_data.shape[2] >= 3:
                rgb_data = hdr_data[:, :, :3]
            else:
                rgb_data = hdr_data
            
            # Apply tone mapping
            ldr_data = tone_map_exr(rgb_data, exposure)
            
            # Convert to PIL Image
            img = Image.fromarray(ldr_data, 'RGB')
        else:
            # Handle regular image formats with PIL
            with Image.open(image_path) as img:
                if img.mode in ('RGBA', 'LA', 'P'):
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = rgb_img
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
        
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=90)
        return output.getvalue()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting image: {str(e)}")

def preload_adjacent_frames(sequence_id: str, frame_num: int, exposure: float, total_frames: int):
    """Background task to preload adjacent frames"""
    try:
        sequence_dir = UPLOAD_DIR / sequence_id
        frame_files = sorted(sequence_dir.glob("frame_*"))
        
        # Preload next 5 frames and previous 2 frames
        frames_to_preload = []
        for offset in [-2, -1, 1, 2, 3, 4, 5]:
            target_frame = frame_num + offset
            if 0 <= target_frame < len(frame_files):
                frames_to_preload.append(target_frame)
        
        for target_frame in frames_to_preload:
            # Check if already cached
            cached_data = frame_cache.get(sequence_id, target_frame, exposure)
            if cached_data is None:
                try:
                    frame_path = frame_files[target_frame]
                    jpeg_data = convert_to_jpeg(frame_path, exposure)
                    frame_cache.put(sequence_id, target_frame, exposure, jpeg_data)
                except Exception as e:
                    print(f"Error preloading frame {target_frame}: {e}")
    except Exception as e:
        print(f"Error in preload task: {e}")

@router.get("/{sequence_id}/frame/{frame_num}")
async def get_frame(sequence_id: str, frame_num: int, exposure: float = 0.0, background_tasks: BackgroundTasks = None):
    """Get a specific frame as JPEG with optional exposure adjustment"""
    # Check cache first
    cached_data = frame_cache.get(sequence_id, frame_num, exposure)
    if cached_data:
        # Start preloading adjacent frames in background
        if background_tasks:
            sequence_dir = UPLOAD_DIR / sequence_id
            if sequence_dir.exists():
                frame_files = sorted(sequence_dir.glob("frame_*"))
                background_tasks.add_task(preload_adjacent_frames, sequence_id, frame_num, exposure, len(frame_files))
        
        return Response(content=cached_data, media_type="image/jpeg")
    
    # Cache miss - generate frame
    sequence_dir = UPLOAD_DIR / sequence_id
    
    if not sequence_dir.exists():
        raise HTTPException(status_code=404, detail="Sequence not found")
    
    frame_files = sorted(sequence_dir.glob("frame_*"))
    if frame_num < 0 or frame_num >= len(frame_files):
        raise HTTPException(status_code=404, detail="Frame not found")
    
    frame_path = frame_files[frame_num]
    
    jpeg_data = convert_to_jpeg(frame_path, exposure)
    
    # Cache the result
    frame_cache.put(sequence_id, frame_num, exposure, jpeg_data)
    
    # Start preloading adjacent frames in background
    if background_tasks:
        background_tasks.add_task(preload_adjacent_frames, sequence_id, frame_num, exposure, len(frame_files))
    
    return Response(content=jpeg_data, media_type="image/jpeg")

@router.get("/{sequence_id}/frame/{frame_num}/thumbnail")
async def get_frame_thumbnail(sequence_id: str, frame_num: int, size: int = 256, exposure: float = 0.0):
    """Get a thumbnail of a specific frame"""
    # Check cache first
    cached_data = frame_cache.get(sequence_id, frame_num, exposure, size)
    if cached_data:
        return Response(content=cached_data, media_type="image/jpeg")
    
    sequence_dir = UPLOAD_DIR / sequence_id
    
    if not sequence_dir.exists():
        raise HTTPException(status_code=404, detail="Sequence not found")
    
    frame_files = sorted(sequence_dir.glob("frame_*"))
    if frame_num < 0 or frame_num >= len(frame_files):
        raise HTTPException(status_code=404, detail="Frame not found")
    
    frame_path = frame_files[frame_num]
    
    try:
        jpeg_data = convert_to_jpeg(frame_path, exposure)
        
        # Convert JPEG bytes back to PIL Image for thumbnail creation
        with Image.open(io.BytesIO(jpeg_data)) as img:
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=85)
            thumbnail_data = output.getvalue()
            
            # Cache the thumbnail
            frame_cache.put(sequence_id, frame_num, exposure, thumbnail_data, size)
            
            return Response(content=thumbnail_data, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating thumbnail: {str(e)}")

@router.post("/{sequence_id}/preload")
async def preload_sequence_range(sequence_id: str, start_frame: int = 0, end_frame: int = None, exposure: float = 0.0, background_tasks: BackgroundTasks = None):
    """Preload a range of frames for smooth playback"""
    sequence_dir = UPLOAD_DIR / sequence_id
    
    if not sequence_dir.exists():
        raise HTTPException(status_code=404, detail="Sequence not found")
    
    frame_files = sorted(sequence_dir.glob("frame_*"))
    if end_frame is None:
        end_frame = len(frame_files) - 1
    
    end_frame = min(end_frame, len(frame_files) - 1)
    start_frame = max(0, start_frame)
    
    if background_tasks:
        background_tasks.add_task(preload_frames_range, sequence_id, start_frame, end_frame, exposure)
    
    return {"message": f"Preloading frames {start_frame} to {end_frame} for sequence {sequence_id}"}

def preload_frames_range(sequence_id: str, start_frame: int, end_frame: int, exposure: float):
    """Background task to preload a range of frames"""
    try:
        sequence_dir = UPLOAD_DIR / sequence_id
        frame_files = sorted(sequence_dir.glob("frame_*"))
        
        for frame_num in range(start_frame, end_frame + 1):
            if frame_num >= len(frame_files):
                break
                
            # Check if already cached
            cached_data = frame_cache.get(sequence_id, frame_num, exposure)
            if cached_data is None:
                try:
                    frame_path = frame_files[frame_num]
                    jpeg_data = convert_to_jpeg(frame_path, exposure)
                    frame_cache.put(sequence_id, frame_num, exposure, jpeg_data)
                except Exception as e:
                    print(f"Error preloading frame {frame_num}: {e}")
    except Exception as e:
        print(f"Error in preload range task: {e}")

@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    return frame_cache.get_cache_stats()

@router.delete("/cache/{sequence_id}")
async def clear_sequence_cache(sequence_id: str):
    """Clear cache for a specific sequence"""
    frame_cache.clear_sequence_cache(sequence_id)
    return {"message": f"Cache cleared for sequence {sequence_id}"}

@router.delete("/cache")
async def clear_all_cache():
    """Clear entire cache"""
    frame_cache.clear_all_cache()
    return {"message": "All cache cleared"}