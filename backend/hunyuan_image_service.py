"""
Standalone FastAPI service for Hunyuan Image 2.1
Much more efficient than 3.0 - works great on consumer GPUs!
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import torch
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hunyuan Image 2.1 Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 2048
    height: int = 2048
    num_inference_steps: int = 8  # 8 for distilled, 50 for full quality
    guidance_scale: float = 3.25  # 3.25 for distilled, 3.5 for full
    shift: int = 4  # 4 for distilled, 5 for full
    seed: Optional[int] = None
    use_refiner: bool = True
    use_reprompt: bool = False

# Storage for processing jobs
IMAGE_JOBS = {}
OUTPUT_DIR = Path("/app/outputs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global pipeline instance
pipeline = None

def load_hunyuan_pipeline():
    """Load Hunyuan Image 2.1 pipeline with FP8 quantization"""
    global pipeline

    if pipeline is not None:
        return True

    try:
        logger.info("Loading Hunyuan Image 2.1 pipeline...")

        # Change to repo directory so relative paths work
        os.chdir("/app/HunyuanImage-2.1")

        # Add repo to path
        sys.path.insert(0, "/app/HunyuanImage-2.1")

        from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline

        # Load with FP8 quantization - models are in ./ckpts
        logger.info("Loading models from local ckpts directory...")
        pipeline = HunyuanImagePipeline.from_pretrained(
            model_name="hunyuanimage-v2.1-distilled",  # Use distilled version for speed
            use_fp8=True  # FP8 quantization for efficiency
        )
        pipeline = pipeline.to("cuda")

        logger.info("✅ Successfully loaded Hunyuan Image 2.1 pipeline")
        return True

    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

@app.post("/generate")
async def generate_image(request: ImageGenerationRequest, background_tasks: BackgroundTasks):
    """Initiate image generation"""
    job_id = str(uuid.uuid4())

    IMAGE_JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "image_path": None,
        "error": None,
        "result": {
            "metadata": {
                "width": request.width,
                "height": request.height,
                "inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale
            }
        }
    }

    # Start generation in background
    background_tasks.add_task(process_image_generation, job_id, request)

    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get generation status"""
    if job_id not in IMAGE_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    return IMAGE_JOBS[job_id]

@app.get("/image/{job_id}")
async def get_image(job_id: str):
    """Download generated image"""
    if job_id not in IMAGE_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = IMAGE_JOBS[job_id]
    if job["status"] != "completed" or not job["image_path"]:
        raise HTTPException(status_code=400, detail="Image not ready")

    image_path = Path(job["image_path"])
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(image_path)

# Gateway-compatible endpoint aliases
@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get generation status (alias for gateway compatibility)"""
    return await get_status(job_id)

@app.get("/download/{job_id}")
async def download_image(job_id: str):
    """Download generated image (alias for gateway compatibility)"""
    return await get_image(job_id)

async def process_image_generation(job_id: str, request: ImageGenerationRequest):
    """Process image generation in background"""
    try:
        IMAGE_JOBS[job_id]["status"] = "processing"
        IMAGE_JOBS[job_id]["progress"] = 10

        # Load pipeline if needed
        if not load_hunyuan_pipeline():
            raise Exception("Failed to load pipeline")

        IMAGE_JOBS[job_id]["progress"] = 30

        # Generate image
        logger.info(f"Generating image for job {job_id}")
        logger.info(f"Prompt: {request.prompt}")
        logger.info(f"Size: {request.width}x{request.height}")

        output_path = OUTPUT_DIR / f"{job_id}.png"

        # Create generator with seed if provided
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(request.seed)

        # Generate
        image = pipeline(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            use_reprompt=request.use_reprompt,
            use_refiner=request.use_refiner,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            shift=request.shift,
            generator=generator,
        )

        IMAGE_JOBS[job_id]["progress"] = 80

        # Save image
        image.save(str(output_path))
        logger.info(f"Image saved to {output_path}")

        IMAGE_JOBS[job_id]["progress"] = 100
        IMAGE_JOBS[job_id]["status"] = "completed"
        IMAGE_JOBS[job_id]["image_path"] = str(output_path)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(traceback.format_exc())

        IMAGE_JOBS[job_id]["status"] = "failed"
        IMAGE_JOBS[job_id]["error"] = str(e)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "Hunyuan Image 2.1"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4006)
