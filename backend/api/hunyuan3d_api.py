"""
FastAPI endpoints for Hunyuan3D image-to-3D processing
"""

import os
import uuid
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import aiofiles

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hunyuan3d import get_processor, get_status

router = APIRouter()

# Storage for processing jobs
PROCESSING_JOBS = {}
UPLOAD_DIR = Path("uploads/3d_generation")
OUTPUT_DIR = Path("outputs/3d_models")

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@router.get("/status")
async def get_hunyuan3d_status():
    """Get the current status of Hunyuan3D models"""
    try:
        status = get_status()
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.post("/upload-image")
async def upload_image_for_3d(file: UploadFile = File(...)):
    """
    Upload an image for 3D generation
    Returns an image_id for tracking
    """
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp", "image/tiff", "image/bmp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Allowed: {allowed_types}"
            )

        # Generate unique ID for this upload
        image_id = str(uuid.uuid4())

        # Determine file extension
        file_ext = Path(file.filename).suffix.lower()
        if not file_ext:
            file_ext = ".png"  # Default extension

        # Save uploaded file
        upload_path = UPLOAD_DIR / f"{image_id}{file_ext}"

        async with aiofiles.open(upload_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        return {
            "success": True,
            "image_id": image_id,
            "filename": file.filename,
            "file_size": len(content),
            "upload_path": str(upload_path)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/generate-3d/{image_id}")
async def generate_3d_model(image_id: str, background_tasks: BackgroundTasks):
    """
    Start 3D generation process for an uploaded image
    Returns a job_id for tracking progress
    """
    try:
        # Find the uploaded image
        image_files = list(UPLOAD_DIR.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

        image_path = str(image_files[0])

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Create output directory for this job
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(exist_ok=True)

        # Initialize job status
        PROCESSING_JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "image_id": image_id,
            "image_path": image_path,
            "output_dir": str(job_output_dir),
            "result": None,
            "error": None
        }

        # Start background processing
        background_tasks.add_task(process_3d_generation, job_id, image_path, str(job_output_dir))

        return {
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "message": "3D generation started"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start 3D generation: {str(e)}")

@router.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a 3D generation job"""
    if job_id not in PROCESSING_JOBS:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = PROCESSING_JOBS[job_id]
    return {
        "success": True,
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "result": job["result"],
        "error": job["error"]
    }

@router.get("/download/{job_id}/{file_type}")
async def download_3d_file(job_id: str, file_type: str):
    """
    Download generated 3D files
    file_type can be: fbx, obj, glb, ply
    """
    try:
        if job_id not in PROCESSING_JOBS:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        job = PROCESSING_JOBS[job_id]
        if job["status"] != "completed":
            raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job['status']}")

        if not job["result"] or "exports" not in job["result"]:
            raise HTTPException(status_code=404, detail="No export files found")

        exports = job["result"]["exports"]
        if file_type not in exports:
            available_types = list(exports.keys())
            raise HTTPException(
                status_code=404,
                detail=f"File type '{file_type}' not available. Available: {available_types}"
            )

        file_path = exports[file_type]
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        return FileResponse(
            path=file_path,
            filename=f"model_{job_id}.{file_type}",
            media_type="application/octet-stream"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@router.delete("/job/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up job files and remove from memory"""
    try:
        if job_id not in PROCESSING_JOBS:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        job = PROCESSING_JOBS[job_id]

        # Remove output directory
        output_dir = Path(job["output_dir"])
        if output_dir.exists():
            shutil.rmtree(output_dir)

        # Remove from memory
        del PROCESSING_JOBS[job_id]

        return {
            "success": True,
            "message": f"Job {job_id} cleaned up successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.get("/jobs")
async def list_jobs():
    """List all current jobs"""
    jobs = []
    for job_id, job_data in PROCESSING_JOBS.items():
        jobs.append({
            "job_id": job_id,
            "status": job_data["status"],
            "progress": job_data["progress"],
            "image_id": job_data["image_id"]
        })

    return {
        "success": True,
        "jobs": jobs,
        "total": len(jobs)
    }

async def process_3d_generation(job_id: str, image_path: str, output_dir: str):
    """
    Background task for processing 3D generation
    """
    try:
        # Update job status
        PROCESSING_JOBS[job_id]["status"] = "processing"
        PROCESSING_JOBS[job_id]["progress"] = 10

        # Get processor and run generation
        processor = get_processor()

        # Update progress
        PROCESSING_JOBS[job_id]["progress"] = 30

        # Run the actual 3D generation
        result = processor.generate_3d_from_image(image_path, output_dir)

        if result["success"]:
            PROCESSING_JOBS[job_id]["status"] = "completed"
            PROCESSING_JOBS[job_id]["progress"] = 100
            PROCESSING_JOBS[job_id]["result"] = result
        else:
            PROCESSING_JOBS[job_id]["status"] = "failed"
            PROCESSING_JOBS[job_id]["error"] = result.get("error", "Unknown error")

    except Exception as e:
        PROCESSING_JOBS[job_id]["status"] = "failed"
        PROCESSING_JOBS[job_id]["error"] = str(e)

    finally:
        # Ensure progress is set to 100 if completed or failed
        if PROCESSING_JOBS[job_id]["status"] in ["completed", "failed"]:
            PROCESSING_JOBS[job_id]["progress"] = 100