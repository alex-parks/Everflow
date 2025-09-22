from fastapi import APIRouter, UploadFile, HTTPException
from typing import List
import os
import json
from pathlib import Path
import aiofiles
import uuid

router = APIRouter()

UPLOAD_DIR = Path("/app/uploads/sequences")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_sequence(files: List[UploadFile]):
    """Upload an image sequence"""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    sequence_id = str(uuid.uuid4())
    sequence_dir = UPLOAD_DIR / sequence_id
    sequence_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    for idx, file in enumerate(files):
        file_extension = Path(file.filename).suffix
        new_filename = f"frame_{idx:04d}{file_extension}"
        file_path = sequence_dir / new_filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        saved_files.append(new_filename)
    
    metadata = {
        "id": sequence_id,
        "frame_count": len(saved_files),
        "frames": saved_files,
        "format": Path(files[0].filename).suffix if files else ""
    }
    
    async with aiofiles.open(sequence_dir / "metadata.json", 'w') as f:
        await f.write(json.dumps(metadata, indent=2))
    
    return {"sequence_id": sequence_id, "frames_uploaded": len(saved_files)}

@router.get("/")
async def list_sequences():
    """List all uploaded sequences"""
    sequences = []
    
    if UPLOAD_DIR.exists():
        for seq_dir in UPLOAD_DIR.iterdir():
            if seq_dir.is_dir():
                metadata_path = seq_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        sequences.append(metadata)
    
    return sequences

@router.get("/{sequence_id}/frames")
async def get_sequence_frames(sequence_id: str):
    """Get list of frames in a sequence"""
    sequence_dir = UPLOAD_DIR / sequence_id
    metadata_path = sequence_dir / "metadata.json"
    
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Sequence not found")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata