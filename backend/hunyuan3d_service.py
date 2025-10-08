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

# Preload models at startup since we have a dedicated GPU
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Hunyuan3D service with dedicated GPU 1 (RTX 4090)")
    logger.info("Loading models at startup for instant availability...")
    if load_hunyuan3d_models():
        logger.info("✅ Models preloaded and ready")
    else:
        logger.error("❌ Failed to preload models")

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
        "stage": "Queued for processing",
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

def unload_hunyuan3d_models():
    """Unload Hunyuan3D models to free GPU memory"""
    global shape_pipeline, paint_pipeline, rembg

    if shape_pipeline is not None or paint_pipeline is not None:
        logger.info("🔄 Unloading Hunyuan3D models to free GPU memory...")

        # Clear models
        shape_pipeline = None
        paint_pipeline = None
        rembg = None

        # Force GPU memory cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("✅ GPU memory freed")

async def process_3d_generation(job_id: str, image_path: str, output_dir: str):
    """Process 3D generation in background"""

    def update_progress(progress: int, message: str):
        """Update job progress and status message"""
        PROCESSING_JOBS[job_id]["progress"] = progress
        PROCESSING_JOBS[job_id]["stage"] = message
        logger.info(f"[{progress}%] {message}")

    try:
        PROCESSING_JOBS[job_id]["status"] = "processing"
        update_progress(0, "Initializing 3D generation pipeline")

        # Ensure models are loaded (should already be loaded from startup)
        update_progress(5, "Loading AI models...")
        if shape_pipeline is None and not load_hunyuan3d_models():
            raise Exception("Failed to load models")
        update_progress(10, "AI models loaded successfully")

        # Preprocess image
        from PIL import Image
        import numpy as np

        update_progress(12, "Loading and preprocessing input image")
        image = Image.open(image_path)
        original_size = image.size
        logger.info(f"Input image size: {original_size[0]}x{original_size[1]}")

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Remove background with high quality settings
        update_progress(15, "Removing background with AI (alpha matting)")
        try:
            from rembg import remove
            # Use U2Net model with alpha matting for better quality
            image = remove(image, alpha_matting=True, alpha_matting_foreground_threshold=240,
                          alpha_matting_background_threshold=10, alpha_matting_erode_size=10)
            update_progress(20, "Background removed successfully")
        except Exception as e:
            logger.warning(f"Background removal failed: {e}, using original image")
            update_progress(20, "Using original image (background removal skipped)")

        # Ensure image is high quality RGB
        if image.mode == 'RGBA':
            # Convert RGBA to RGB with white background for better results
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
            logger.info("✓ Converted RGBA to RGB with white background")

        # Generate 3D shape with MAXIMUM QUALITY settings
        update_progress(22, "Starting diffusion sampling (100 steps)")
        logger.info(f"Generating 3D shape for job {job_id} with maximum quality")

        # Wrap the pipeline call to track progress
        import tqdm
        from functools import partial

        # Create a custom progress tracker
        class ProgressTracker:
            def __init__(self, total_steps, start_pct, end_pct):
                self.total_steps = total_steps
                self.start_pct = start_pct
                self.end_pct = end_pct
                self.current_step = 0

            def update(self, step=None):
                if step is not None:
                    self.current_step = step
                else:
                    self.current_step += 1

                progress = int(self.start_pct + (self.current_step / self.total_steps) * (self.end_pct - self.start_pct))
                update_progress(progress, f"Diffusion sampling: step {self.current_step}/{self.total_steps}")

        # Monkey-patch tqdm for this generation
        original_tqdm = tqdm.tqdm
        diffusion_tracker = ProgressTracker(100, 22, 55)

        def custom_tqdm(iterable=None, *args, **kwargs):
            if iterable is not None and hasattr(iterable, '__len__'):
                total = len(iterable)
                if total == 100:  # This is our diffusion sampling
                    class TqdmWrapper:
                        def __init__(self, iterable):
                            self.iterable = iterable
                            self.n = 0

                        def __iter__(self):
                            for i, item in enumerate(self.iterable):
                                diffusion_tracker.update(i + 1)
                                yield item

                        def update(self, n=1):
                            pass

                        def close(self):
                            pass

                        def set_description(self, desc):
                            pass

                    return TqdmWrapper(iterable)
            return original_tqdm(iterable, *args, **kwargs)

        tqdm.tqdm = custom_tqdm

        with torch.no_grad():
            shape_output = shape_pipeline(
                image,
                num_inference_steps=100,  # MAXIMUM quality (was 30, then 50)
                guidance_scale=7.5,
                dual_guidance=True,  # Enable dual guidance for better quality
                dual_guidance_scale=10.5,  # Dual guidance strength
                octree_resolution=512,  # MAXIMUM mesh detail resolution (was 384)
                mc_level=-1 / 1024  # Higher precision marching cubes (was -1/512)
            )

        # Restore original tqdm
        tqdm.tqdm = original_tqdm

        update_progress(55, "Diffusion sampling complete, extracting 3D mesh")

        # Extract mesh (handle different output formats)
        if isinstance(shape_output, list):
            mesh = shape_output[0]  # Get first mesh from list
        elif hasattr(shape_output, 'meshes'):
            mesh = shape_output.meshes[0]
        else:
            mesh = shape_output

        logger.info(f"Mesh type: {type(mesh)}")

        update_progress(70, "3D mesh extracted, starting cleanup")

        # Apply texture/paint if available
        if paint_pipeline is not None:
            update_progress(72, "Applying texture to 3D model")
            with torch.no_grad():
                painted_mesh = paint_pipeline(
                    mesh,
                    image,
                    num_views=6
                )
            final_mesh = painted_mesh
            update_progress(75, "Texture applied successfully")
        else:
            final_mesh = mesh

        # COMPREHENSIVE MESH CLEANUP AND OPTIMIZATION
        import trimesh
        update_progress(75, "Converting to trimesh format")

        # Convert to trimesh object
        if isinstance(final_mesh, trimesh.Trimesh):
            tri_mesh = final_mesh
        elif hasattr(final_mesh, 'vertices') and hasattr(final_mesh, 'faces'):
            tri_mesh = trimesh.Trimesh(
                vertices=final_mesh.vertices,
                faces=final_mesh.faces
            )
        else:
            tri_mesh = final_mesh

        # 1. Remove degenerate and duplicate faces
        update_progress(77, "Cleanup: Removing degenerate faces")
        tri_mesh.remove_degenerate_faces()
        tri_mesh.remove_duplicate_faces()

        # 2. Remove unreferenced vertices
        update_progress(79, "Cleanup: Removing unreferenced vertices")
        tri_mesh.remove_unreferenced_vertices()

        # 3. Fix mesh normals for proper lighting
        update_progress(81, "Cleanup: Fixing mesh normals")
        tri_mesh.fix_normals()

        # 4. Fill small holes in the mesh
        update_progress(83, "Cleanup: Filling mesh holes")
        try:
            tri_mesh.fill_holes()
        except:
            logger.warning("Could not fill holes (mesh may be watertight already)")

        # 5. Remove infinite/NaN values
        update_progress(85, "Cleanup: Removing invalid values")
        tri_mesh.remove_infinite_values()

        # 6. Subdivide mesh for smoother appearance (optional but recommended)
        update_progress(87, "Optimization: Subdividing mesh for smoothness")
        try:
            # Only subdivide if mesh is not already too dense
            if len(tri_mesh.vertices) < 500000:
                logger.info(f"Subdividing mesh from {len(tri_mesh.vertices)} vertices...")
                tri_mesh = tri_mesh.subdivide()
                logger.info(f"✓ Subdivided to {len(tri_mesh.vertices)} vertices")
            else:
                logger.info("Skipping subdivision (mesh already dense)")
        except Exception as e:
            logger.warning(f"Subdivision failed: {e}")

        # Export formats with quality logging
        exports = {}

        logger.info(f"Final mesh stats: {len(tri_mesh.vertices)} vertices, {len(tri_mesh.faces)} faces")
        logger.info(f"Mesh is watertight: {tri_mesh.is_watertight}")
        logger.info(f"Mesh bounds: {tri_mesh.bounds}")

        # PLY format (high precision)
        update_progress(90, f"Exporting PLY format ({len(tri_mesh.vertices):,} vertices)")
        ply_path = os.path.join(output_dir, "model.ply")
        tri_mesh.export(ply_path, encoding='binary')
        exports["ply"] = ply_path
        logger.info(f"✓ Exported PLY: {os.path.getsize(ply_path) / 1024 / 1024:.2f} MB")

        # OBJ format (with normals)
        update_progress(93, "Exporting OBJ format (with normals)")
        obj_path = os.path.join(output_dir, "model.obj")
        tri_mesh.export(obj_path, include_normals=True)
        exports["obj"] = obj_path
        logger.info(f"✓ Exported OBJ: {os.path.getsize(obj_path) / 1024 / 1024:.2f} MB")

        # GLB format (for web viewing)
        update_progress(96, "Exporting GLB format (for web viewing)")
        glb_path = os.path.join(output_dir, "model.glb")
        tri_mesh.export(glb_path)
        exports["glb"] = glb_path
        logger.info(f"✓ Exported GLB: {os.path.getsize(glb_path) / 1024 / 1024:.2f} MB")

        # Save result
        update_progress(98, "Finalizing and saving metadata")
        PROCESSING_JOBS[job_id]["result"] = {
            "exports": exports,
            "vertices": len(tri_mesh.vertices),
            "faces": len(tri_mesh.faces),
            "watertight": tri_mesh.is_watertight,
            "volume": float(tri_mesh.volume) if tri_mesh.is_watertight else None
        }

        update_progress(100, f"Complete! Generated {len(tri_mesh.vertices):,} vertices, {len(tri_mesh.faces):,} faces")
        PROCESSING_JOBS[job_id]["status"] = "completed"

        logger.info(f"✅ Completed job {job_id} - MAXIMUM QUALITY MODE")
        logger.info(f"Models remain loaded on dedicated GPU for instant reuse")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        PROCESSING_JOBS[job_id]["status"] = "failed"
        PROCESSING_JOBS[job_id]["error"] = str(e)
        PROCESSING_JOBS[job_id]["stage"] = f"Failed: {str(e)}"
        PROCESSING_JOBS[job_id]["progress"] = 100

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4007)