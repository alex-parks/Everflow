from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import List, Optional, Dict, Any
import os
import json
import numpy as np
from pathlib import Path
import asyncio
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

router = APIRouter()

UPLOAD_DIR = Path("/app/uploads/crowd_sequences")
TRAINING_DIR = Path("/app/training_data")
MODELS_DIR = Path("/app/models")

# Create directories
for dir_path in [UPLOAD_DIR, TRAINING_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

@router.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify API is working"""
    return {"status": "crowd API is working", "timestamp": str(uuid.uuid4())}

@router.post("/upload-crowd-sequence")
async def upload_crowd_sequence(
    name: str = Form(...),
    frame_rate: float = Form(24.0),
    camera_data: str = Form("{}"),
    files: List[UploadFile] = File(...)
):
    """Upload a multi-pass EXR crowd sequence"""
    
    sequence_id = str(uuid.uuid4())
    sequence_dir = UPLOAD_DIR / sequence_id
    sequence_dir.mkdir(parents=True, exist_ok=True)
    
    # Save files
    saved_files = []
    for file in files:
        filename = file.filename
        file_path = sequence_dir / filename
        
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        saved_files.append(filename)
    
    # Create simple metadata
    metadata = {
        "sequence_id": sequence_id,
        "name": name,
        "frame_count": len(saved_files),
        "frame_rate": frame_rate,
        "camera_data": json.loads(camera_data),
        "files": saved_files,
        "created_at": str(uuid.uuid4())
    }
    
    # Save metadata
    metadata_path = sequence_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        "sequence_id": sequence_id,
        "message": f"Uploaded {len(saved_files)} files",
        "passes": ["beauty", "depth", "emission", "normal"],
        "crowd_analysis": {
            "people_detected": 0,
            "resolution": [1920, 1080]
        }
    }

@router.get("/models")
async def list_available_models():
    """List available trained models"""
    # Return empty list for now
    return {"models": []}

@router.get("/training/status")
async def get_training_status():
    """Get current training status"""
    return {
        "active_training": False,
        "training_processes": [],
        "latest_checkpoint": None,
        "models_dir": str(MODELS_DIR)
    }