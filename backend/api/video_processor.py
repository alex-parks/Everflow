"""
Video Processing API for converting videos to multi-pass EXR sequences
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import Optional
import os
import json
import numpy as np
from pathlib import Path
import uuid
import tempfile
import subprocess
import shutil
import OpenImageIO as oiio
from PIL import Image
from scipy import ndimage

router = APIRouter()

UPLOAD_DIR = Path("/app/uploads/crowd_sequences")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload-video-sequence")
async def upload_video_sequence(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    target_fps: float = Form(24.0),
    extract_proxy_passes: bool = Form(True)
):
    """Upload video file and extract frames for crowd enhancement"""
    
    processing_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Save uploaded video
        video_path = temp_dir / f"input_video.{video.filename.split('.')[-1]}"
        with open(video_path, 'wb') as f:
            content = await video.read()
            f.write(content)
        
        # Start background processing
        background_tasks.add_task(process_video_to_sequence, processing_id, video_path, target_fps, extract_proxy_passes)
        
        return {
            "processing_id": processing_id,
            "message": "Video upload successful, processing started",
            "estimated_time": "5-15 minutes depending on video length"
        }
        
    except Exception as e:
        # Cleanup temp directory on error
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Video upload failed: {str(e)}")

@router.get("/video-processing-status/{processing_id}")
async def get_video_processing_status(processing_id: str):
    """Get status of video processing"""
    status_file = UPLOAD_DIR / f"processing_{processing_id}.json"
    
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="Processing job not found")
    
    try:
        with open(status_file, 'r') as f:
            status = json.load(f)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading processing status: {str(e)}")

def process_video_to_sequence(processing_id: str, video_path: Path, target_fps: float, extract_proxy_passes: bool):
    """Background task to process video into EXR sequence"""
    
    status_file = UPLOAD_DIR / f"processing_{processing_id}.json"
    
    def update_status(status: str, progress: float = 0, current_step: str = "", error: str = None, result: dict = None):
        status_data = {
            "processing_id": processing_id,
            "status": status,
            "progress": progress,
            "current_step": current_step,
            "timestamp": str(np.datetime64('now'))
        }
        if error:
            status_data["error"] = error
        if result:
            status_data["result"] = result
            
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
    
    try:
        update_status("processing", 0, "Analyzing video...")
        
        # Create output directory
        sequence_dir = UPLOAD_DIR / processing_id
        sequence_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract frames using FFmpeg
        update_status("processing", 10, "Extracting frames from video...")
        
        frames_dir = sequence_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        # Use FFmpeg to extract frames
        ffmpeg_cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', f'fps={target_fps}',
            '-y',  # Overwrite output files
            str(frames_dir / 'frame_%04d.png')
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg failed: {result.stderr}")
        
        update_status("processing", 30, "Converting frames to proxy render format...")
        
        # Count extracted frames
        frame_files = list(frames_dir.glob("frame_*.png"))
        frame_count = len(frame_files)
        
        if frame_count == 0:
            raise Exception("No frames extracted from video")
        
        # Get frame dimensions from first frame
        first_frame = Image.open(frame_files[0]).convert('RGB')
        frame_size = first_frame.size
        
        # Convert to multi-pass EXR format
        for i, frame_file in enumerate(frame_files):
            progress = 30 + (i / frame_count) * 50  # 30-80% progress for frame processing
            update_status("processing", progress, f"Processing frame {i+1}/{frame_count}...")
            
            # Load frame
            frame = Image.open(frame_file).convert('RGB')
            frame_array = np.array(frame).astype(np.float32) / 255.0
            
            # Create proxy passes
            beauty_pass = frame_array
            
            # Generate synthetic depth pass (simple gradient based on luminance)
            luminance = np.dot(frame_array, [0.299, 0.587, 0.114])
            depth_pass = 1.0 - luminance  # Darker areas = closer
            
            # Generate synthetic emission pass (detect bright clothing areas)
            emission_pass = np.zeros_like(frame_array)
            bright_mask = luminance > 0.7
            if np.any(bright_mask):
                # Add random emissive colors to bright areas (simulating shirt colors)
                emission_colors = np.random.rand(bright_mask.sum(), 3) * 0.8 + 0.2
                emission_pass[bright_mask] = emission_colors
            
            # Generate synthetic normal pass (simple edge detection)
            grad_x = ndimage.sobel(luminance, axis=1)
            grad_y = ndimage.sobel(luminance, axis=0)
            normal_pass = np.dstack([grad_x, grad_y, np.ones_like(grad_x)])
            normal_pass = (normal_pass + 1.0) / 2.0  # Normalize to [0,1]
            
            # Save as EXR files
            frame_num = i
            
            # Save beauty pass
            beauty_path = sequence_dir / f"beauty_{frame_num:04d}.exr"
            save_array_as_exr(beauty_pass, beauty_path)
            
            # Save depth pass
            depth_path = sequence_dir / f"depth_{frame_num:04d}.exr"
            save_array_as_exr(depth_pass, depth_path)
            
            # Save emission pass
            emission_path = sequence_dir / f"emission_{frame_num:04d}.exr"
            save_array_as_exr(emission_pass, emission_path)
            
            # Save normal pass
            normal_path = sequence_dir / f"normal_{frame_num:04d}.exr"
            save_array_as_exr(normal_pass, normal_path)
        
        update_status("processing", 80, "Creating sequence metadata...")
        
        # Create metadata
        metadata = {
            "sequence_id": processing_id,
            "name": f"Video Sequence {processing_id[:8]}",
            "frame_count": frame_count,
            "resolution": list(frame_size),
            "frame_rate": target_fps,
            "camera_data": {
                "location": [0, 0, 10],
                "rotation_euler": [0, 0, 0],
                "angle": 45,
                "clip_start": 0.1,
                "clip_end": 1000,
                "resolution": list(frame_size),
                "matrix_world": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
                "projection_matrix": [0]*16,
                "world_to_screen": [0]*16
            },
            "passes": {
                "beauty": {"available_frames": list(range(frame_count)), "total_frames": frame_count},
                "depth": {"available_frames": list(range(frame_count)), "total_frames": frame_count},
                "emission": {"available_frames": list(range(frame_count)), "total_frames": frame_count},
                "normal": {"available_frames": list(range(frame_count)), "total_frames": frame_count}
            },
            "crowd_config": {
                "people": [],  # Would be populated by crowd analysis
                "total_people": 0
            },
            "created_at": str(np.datetime64('now')),
            "source": "video_conversion"
        }
        
        # Save metadata
        metadata_path = sequence_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        update_status("processing", 90, "Finalizing sequence...")
        
        # Cleanup temp files
        shutil.rmtree(video_path.parent)
        
        # Prepare result
        result = {
            "sequence_id": processing_id,
            "message": f"Video processed successfully into {frame_count} frames",
            "passes": ["beauty", "depth", "emission", "normal"],
            "crowd_analysis": {
                "people_detected": 0,  # Would be updated by actual analysis
                "resolution": list(frame_size)
            }
        }
        
        update_status("completed", 100, "Video processing complete!", result=result)
        
    except Exception as e:
        update_status("failed", 0, "Processing failed", error=str(e))
        # Cleanup on error
        if 'sequence_dir' in locals() and sequence_dir.exists():
            shutil.rmtree(sequence_dir)

def save_array_as_exr(array: np.ndarray, output_path: Path):
    """Save numpy array as EXR file using OpenImageIO"""
    try:
        # Ensure array is in correct format
        if len(array.shape) == 2:
            # Single channel (depth)
            height, width = array.shape
            channels = 1
            spec = oiio.ImageSpec(width, height, channels, oiio.FLOAT)
            spec.channelnames = ["Z"]
        elif len(array.shape) == 3:
            # Multi-channel (beauty, emission, normal)
            height, width, channels = array.shape
            spec = oiio.ImageSpec(width, height, channels, oiio.FLOAT)
            if channels == 3:
                spec.channelnames = ["R", "G", "B"]
            elif channels == 4:
                spec.channelnames = ["R", "G", "B", "A"]
        
        # Create output
        img_output = oiio.ImageOutput.create(str(output_path))
        if not img_output:
            raise Exception(f"Could not create output file: {output_path}")
        
        img_output.open(str(output_path), spec)
        img_output.write_image(array.astype(np.float32))
        img_output.close()
        
    except Exception as e:
        print(f"Error saving EXR {output_path}: {e}")
        raise