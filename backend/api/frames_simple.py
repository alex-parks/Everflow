from fastapi import APIRouter, HTTPException, Response, BackgroundTasks
from pathlib import Path
from PIL import Image
import io
import numpy as np

router = APIRouter()

UPLOAD_DIR = Path("/app/uploads/sequences")

@router.get("/{sequence_id}/frame/{frame_num}")
async def get_frame(sequence_id: str, frame_num: int, exposure: float = 0.0):
    """Get a specific frame as JPEG"""
    
    sequence_dir = UPLOAD_DIR / sequence_id
    if not sequence_dir.exists():
        raise HTTPException(status_code=404, detail="Sequence not found")
    
    # Find frame files
    frame_files = sorted(sequence_dir.glob("frame_*"))
    
    if frame_num >= len(frame_files):
        raise HTTPException(status_code=404, detail="Frame not found")
    
    frame_path = frame_files[frame_num]
    
    # For now, just return a placeholder image for EXR files
    if frame_path.suffix.lower() == '.exr':
        # Create a placeholder image
        img = Image.new('RGB', (1920, 1080), color=(100, 100, 100))
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