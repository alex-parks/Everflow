"""
Standalone FastAPI service for Hunyuan3D 2.1
Runs on port 4007 with PyTorch 2.1.2
"""

import os
import sys
import torch
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
from typing import Dict, Any, Optional
import shutil
import aiofiles
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hunyuan3D 2.1 Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for processing jobs
PROCESSING_JOBS = {}
UPLOAD_DIR = Path("/app/uploads/3d")
OUTPUT_DIR = Path("/app/outputs/3d")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global model instances
shape_pipeline = None
paint_pipeline = None
rembg = None

def load_hunyuan3d_models():
    """Load Hunyuan3D 2.1 models"""
    global shape_pipeline, paint_pipeline

    if shape_pipeline is not None:
        return True

    try:
        logger.info("Loading Hunyuan3D 2.1 models...")

        # Add model path to sys.path
        model_path = "/app/Hunyuan3D-2.1"
        sys.path.insert(0, os.path.join(model_path, 'hy3dshape'))
        sys.path.insert(0, os.path.join(model_path, 'hy3dpaint'))
        sys.path.insert(0, model_path)

        # Import shape generation pipeline only (no paint/texture)
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dshape.rembg import BackgroundRemover

        # Initialize background remover
        global rembg
        rembg = BackgroundRemover()

        # Load shape generation pipeline
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
        logger.info("✅ Shape pipeline loaded successfully")

        # Skip paint pipeline (requires bpy/Blender)
        paint_pipeline = None
        logger.info("ℹ️  Paint pipeline disabled - will generate untextured 3D shapes")

        logger.info("✅ Successfully loaded Hunyuan3D 2.1 shape generation")
        return True

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting up - loading Hunyuan3D models...")
    load_hunyuan3d_models()
    logger.info("Startup complete")

@app.get("/")
def read_root():
    return {"service": "Hunyuan3D 2.1", "port": 4007, "status": "running"}

@app.get("/status")
def get_status():
    """Get service status"""
    return {
        "service": "Hunyuan3D 2.1",
        "model_loaded": shape_pipeline is not None,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "xformers_available": True  # We have xformers 0.0.23.post1
    }

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload image for 3D generation"""
    # Validate file type
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

    # Generate unique ID
    image_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix or ".png"
    upload_path = UPLOAD_DIR / f"{image_id}{file_ext}"

    # Save file
    async with aiofiles.open(upload_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    return {
        "image_id": image_id,
        "path": str(upload_path)
    }

@app.post("/generate/{image_id}")
async def generate_3d(image_id: str):
    """Generate 3D model from uploaded image"""

    # Find uploaded image
    image_files = list(UPLOAD_DIR.glob(f"{image_id}.*"))
    if not image_files:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = str(image_files[0])

    # Generate job ID
    job_id = str(uuid.uuid4())
    job_output_dir = OUTPUT_DIR / job_id
    job_output_dir.mkdir(exist_ok=True)

    # Initialize job
    PROCESSING_JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "image_id": image_id,
        "output_dir": str(job_output_dir),
        "result": None,
        "error": None
    }

    # Start truly async background task (doesn't block response)
    asyncio.create_task(
        process_3d_generation(
            job_id,
            image_path,
            str(job_output_dir)
        )
    )

    return {
        "job_id": job_id,
        "status": "queued"
    }

@app.get("/job/{job_id}")
def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in PROCESSING_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    return PROCESSING_JOBS[job_id]

@app.get("/download/{job_id}/{file_type}")
def download_3d_file(job_id: str, file_type: str):
    """Download 3D file (obj, ply, glb, fbx)"""
    if job_id not in PROCESSING_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = PROCESSING_JOBS[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    if not job["result"] or "exports" not in job["result"]:
        raise HTTPException(status_code=404, detail="No exports found")

    exports = job["result"]["exports"]
    if file_type not in exports:
        raise HTTPException(status_code=404, detail=f"File type {file_type} not available")

    file_path = exports[file_type]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)

async def process_3d_generation(job_id: str, image_path: str, output_dir: str):
    """Process 3D generation in background"""
    try:
        PROCESSING_JOBS[job_id]["status"] = "processing"
        PROCESSING_JOBS[job_id]["progress"] = 10

        # Models are already loaded at startup
        if shape_pipeline is None:
            raise Exception("Shape pipeline not loaded")

        PROCESSING_JOBS[job_id]["progress"] = 30

        # Preprocess image
        from PIL import Image
        import numpy as np

        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Remove background if needed
        try:
            from rembg import remove
            image = remove(image)
            logger.info("Background removed")
        except:
            logger.warning("Background removal failed, using original image")

        PROCESSING_JOBS[job_id]["progress"] = 50

        # Generate 3D shape
        logger.info(f"Generating 3D shape for job {job_id}")

        with torch.no_grad():
            shape_output = shape_pipeline(
                image,
                num_inference_steps=30,
                guidance_scale=7.5
            )

        # Extract mesh (handle different output formats)
        if isinstance(shape_output, list):
            mesh = shape_output[0]  # Get first mesh from list
        elif hasattr(shape_output, 'meshes'):
            mesh = shape_output.meshes[0]
        else:
            mesh = shape_output

        logger.info(f"Mesh type: {type(mesh)}")

        PROCESSING_JOBS[job_id]["progress"] = 70

        # Apply texture/paint if available
        if paint_pipeline is not None:
            logger.info("Applying texture...")
            with torch.no_grad():
                painted_mesh = paint_pipeline(
                    mesh,
                    image,
                    num_views=6
                )
            final_mesh = painted_mesh
        else:
            logger.info("Skipping texture (paint pipeline not available)")
            final_mesh = mesh

        PROCESSING_JOBS[job_id]["progress"] = 90

        # Export to different formats
        import trimesh

        # Convert to trimesh object
        if isinstance(final_mesh, trimesh.Trimesh):
            tri_mesh = final_mesh
        elif hasattr(final_mesh, 'vertices') and hasattr(final_mesh, 'faces'):
            tri_mesh = trimesh.Trimesh(
                vertices=final_mesh.vertices,
                faces=final_mesh.faces
            )
        else:
            # Already a trimesh object or compatible format
            tri_mesh = final_mesh

        # Export formats
        exports = {}

        # PLY format
        ply_path = os.path.join(output_dir, "model.ply")
        tri_mesh.export(ply_path)
        exports["ply"] = ply_path

        # OBJ format
        obj_path = os.path.join(output_dir, "model.obj")
        tri_mesh.export(obj_path)
        exports["obj"] = obj_path

        # GLB format
        glb_path = os.path.join(output_dir, "model.glb")
        tri_mesh.export(glb_path)
        exports["glb"] = glb_path

        # Save result
        PROCESSING_JOBS[job_id]["result"] = {
            "exports": exports,
            "vertices": len(tri_mesh.vertices),
            "faces": len(tri_mesh.faces)
        }
        PROCESSING_JOBS[job_id]["status"] = "completed"
        PROCESSING_JOBS[job_id]["progress"] = 100

        logger.info(f"✅ Completed job {job_id}")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        PROCESSING_JOBS[job_id]["status"] = "failed"
        PROCESSING_JOBS[job_id]["error"] = str(e)
        PROCESSING_JOBS[job_id]["progress"] = 100

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4007)