"""
Standalone FastAPI service for Hunyuan Image 2.1
Much more efficient than 3.0 - works great on consumer GPUs!
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import torch
import logging
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

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

# Global pipeline and refiner instances
pipeline = None
refiner = None
model_loading = False
model_loaded = False

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=2)

def load_hunyuan_pipeline_sync():
    """Load Hunyuan Image 2.1 pipeline and refiner with FP8 quantization"""
    global pipeline, refiner, model_loaded, model_loading

    if pipeline is not None and refiner is not None:
        logger.info("✅ Models already loaded")
        return True

    try:
        model_loading = True
        logger.info("🔄 Loading Hunyuan Image 2.1 pipeline...")

        # Change to repo directory so relative paths work
        os.chdir("/app/HunyuanImage-2.1")

        # Add repo to path
        sys.path.insert(0, "/app/HunyuanImage-2.1")

        from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline

        # Load main pipeline with FP8 quantization
        logger.info("📦 Loading main generation pipeline from local ckpts...")
        pipeline = HunyuanImagePipeline.from_pretrained(
            model_name="hunyuanimage-v2.1-distilled",  # Use distilled version for speed
            use_fp8=True  # FP8 quantization for efficiency
        )
        pipeline = pipeline.to("cuda")
        logger.info("✅ Main pipeline loaded successfully")

        # Load refiner pipeline separately to avoid reloading during generation
        logger.info("📦 Loading refiner pipeline from local ckpts...")
        refiner = HunyuanImagePipeline.from_pretrained(
            model_name="hunyuanimage-v2.1-refiner",
            use_fp8=True
        )
        refiner = refiner.to("cuda")
        logger.info("✅ Refiner pipeline loaded successfully")

        logger.info("🎉 All models loaded and ready for generation!")
        model_loaded = True
        model_loading = False
        return True

    except Exception as e:
        logger.error(f"❌ Failed to load pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        model_loading = False
        model_loaded = False
        return False

async def load_hunyuan_pipeline():
    """Async wrapper for loading pipeline"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, load_hunyuan_pipeline_sync)

def unload_hunyuan_pipeline():
    """Unload Hunyuan Image pipelines to free GPU memory"""
    global pipeline, refiner, model_loaded

    if pipeline is not None or refiner is not None:
        logger.info("🔄 Unloading Hunyuan Image pipelines to free GPU memory...")

        # Clear pipelines
        pipeline = None
        refiner = None
        model_loaded = False

        # Force GPU memory cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("✅ GPU memory freed")
        return True
    return False

@app.on_event("startup")
async def startup_event():
    """Pre-load models on startup"""
    logger.info("🚀 Starting Hunyuan Image 2.1 Service...")
    logger.info("⏳ Pre-loading AI models in background...")
    # Load models in background so startup doesn't block
    asyncio.create_task(load_hunyuan_pipeline())

@app.get("/model-status")
async def get_model_status():
    """Get current model loading status"""
    return {
        "loaded": model_loaded,
        "loading": model_loading,
        "ready": model_loaded and not model_loading
    }

@app.post("/generate")
async def generate_image(request: ImageGenerationRequest, background_tasks: BackgroundTasks):
    """Initiate image generation"""
    job_id = str(uuid.uuid4())

    # Determine initial status based on model state
    if model_loading:
        initial_status = "queued"
        initial_stage = "Waiting for AI models to finish loading..."
        initial_progress = 0
    elif not model_loaded:
        initial_status = "queued"
        initial_stage = "Preparing to load AI models..."
        initial_progress = 0
    else:
        initial_status = "queued"
        initial_stage = "Ready to generate - queued for processing"
        initial_progress = 0

    IMAGE_JOBS[job_id] = {
        "status": initial_status,
        "progress": initial_progress,
        "stage": initial_stage,
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

    return {"job_id": job_id, "status": initial_status}

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

def run_generation_sync(job_id: str, request: ImageGenerationRequest, update_callback):
    """Synchronous image generation to run in executor"""
    global pipeline, refiner

    try:
        output_path = OUTPUT_DIR / f"{job_id}.png"

        # Create generator with seed if provided
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(request.seed)

        # Generate base image with main pipeline
        logger.info(f"🎨 Generating base image with main pipeline...")
        image = pipeline(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            use_reprompt=request.use_reprompt,
            use_refiner=False,  # Don't use built-in refiner, we'll use ours
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            shift=request.shift,
            generator=generator,
        )

        # Apply refiner if requested
        if request.use_refiner and refiner is not None:
            logger.info(f"✨ Applying refiner for enhanced quality...")
            update_callback(65, "Applying refiner for enhanced quality...")

            # Use the pre-loaded refiner
            image = refiner(
                prompt=request.prompt,
                image=image,  # Pass the base image
                num_inference_steps=4,  # Refiner uses fewer steps
                guidance_scale=request.guidance_scale,
                shift=1,  # Refiner uses shift=1
                generator=generator,
            )

        # Save image
        update_callback(85, "Saving generated image...")
        image.save(str(output_path))
        logger.info(f"💾 Image saved to {output_path}")

        return str(output_path)

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise

async def process_image_generation(job_id: str, request: ImageGenerationRequest):
    """Process image generation in background"""

    def update_progress(progress: int, message: str):
        """Update job progress and status message"""
        IMAGE_JOBS[job_id]["progress"] = progress
        IMAGE_JOBS[job_id]["stage"] = message
        logger.info(f"Job {job_id} [{progress}%] {message}")

    try:
        IMAGE_JOBS[job_id]["status"] = "processing"
        update_progress(0, "Starting image generation pipeline...")

        # Wait for models to load if they're not ready
        if model_loading:
            update_progress(2, "⏳ Waiting for AI models to load (this may take 2-3 minutes on first run)...")

            # Poll until models are loaded
            while model_loading:
                await asyncio.sleep(2)
                update_progress(5, "⏳ Still loading AI models... (loading transformer, VAE, text encoders)")

        elif not model_loaded:
            update_progress(5, "📦 Loading AI models for the first time...")
            success = await load_hunyuan_pipeline()
            if not success:
                raise Exception("Failed to load AI models")
            update_progress(15, "✅ AI models loaded successfully")
        else:
            update_progress(15, "✅ AI models ready - using pre-loaded pipeline")

        # Prepare for generation
        logger.info(f"🎨 Generating image for job {job_id}")
        logger.info(f"📝 Prompt: {request.prompt}")
        logger.info(f"📐 Size: {request.width}x{request.height}")
        logger.info(f"🔧 Steps: {request.num_inference_steps}, Guidance: {request.guidance_scale}")

        update_progress(20, "Preparing generation parameters...")

        if request.seed is not None:
            update_progress(22, f"Using seed {request.seed} for reproducibility")

        # Generate image (run in executor to avoid blocking)
        update_progress(25, f"🎨 Generating base image ({request.num_inference_steps} diffusion steps)...")

        loop = asyncio.get_event_loop()
        output_path = await loop.run_in_executor(
            executor,
            run_generation_sync,
            job_id,
            request,
            update_progress
        )

        update_progress(95, f"✅ Image generated successfully ({request.width}x{request.height})")

        IMAGE_JOBS[job_id]["status"] = "completed"
        IMAGE_JOBS[job_id]["image_path"] = output_path
        update_progress(100, f"🎉 Complete! Image ready for download")

        logger.info(f"✨ Job {job_id} completed successfully")
        logger.info(f"💡 Pipeline remains loaded in memory for instant reuse")

    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(traceback.format_exc())

        IMAGE_JOBS[job_id]["status"] = "failed"
        IMAGE_JOBS[job_id]["error"] = str(e)
        IMAGE_JOBS[job_id]["stage"] = f"❌ Failed: {str(e)}"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "Hunyuan Image 2.1"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4006)
