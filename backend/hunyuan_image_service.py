"""
Standalone FastAPI service for Hunyuan Image 3.0
Using the OFFICIAL implementation approach
"""

import os
import sys
import torch
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from typing import Dict, Any, Optional
from fastapi.responses import FileResponse
from fastapi import BackgroundTasks

# Add the official code path FIRST
sys.path.insert(0, "/app/HunyuanImage-3-Official")
# Add model path for tokenizer files
sys.path.insert(0, "/app/HunyuanImage-3-Model")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hunyuan Image 3.0 Service")

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
    width: int = 256  # Reduced default size
    height: int = 256  # Reduced default size
    num_inference_steps: int = 10  # Reduced steps for faster testing
    guidance_scale: float = 7.5
    seed: Optional[int] = None

# Storage for processing jobs
IMAGE_JOBS = {}
OUTPUT_DIR = Path("/app/outputs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global model instance
model = None
MODEL_PATH = "/app/HunyuanImage-3-Model"  # This has the actual model weights

def load_hunyuan_image_model():
    """Load Hunyuan Image 3.0 model using OFFICIAL approach"""
    global model

    if model is not None:
        return True

    try:
        logger.info("Loading Hunyuan Image 3.0 model using official implementation...")
        logger.info(f"Model path: {MODEL_PATH}")

        # Import the OFFICIAL model class from the official code
        from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM

        # Load model with settings from official script
        # This 160GB model is too large for 16GB GPU
        # Force CPU execution with minimal GPU use
        kwargs = dict(
            attn_implementation="sdpa",  # Use sdpa as default
            torch_dtype=torch.float16,  # Use float16 to save memory
            device_map="sequential",  # Load layers sequentially, offload to CPU
            moe_impl="eager",  # Use eager mode for MoE
            offload_folder="/tmp",  # Offload to disk if needed
            low_cpu_mem_usage=True,  # Reduce memory usage during loading
            max_memory={0: "8GB"},  # Only use 8GB of GPU
        )

        logger.info("Loading model from local files...")
        model = HunyuanImage3ForCausalMM.from_pretrained(
            MODEL_PATH,
            **kwargs
        )

        # Load tokenizer - REQUIRED!
        logger.info("Loading tokenizer...")
        model.load_tokenizer(MODEL_PATH)

        logger.info("✅ Successfully loaded Hunyuan Image 3.0 model with tokenizer")
        return True

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(f"Error details: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

@app.get("/")
def read_root():
    return {"service": "Hunyuan Image 3.0", "port": 4006, "status": "running"}

@app.get("/status")
def get_status():
    """Get service status"""
    return {
        "service": "Hunyuan Image 3.0",
        "model_loaded": model is not None,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_path": MODEL_PATH
    }

@app.post("/generate")
async def generate_image(request: ImageGenerationRequest, background_tasks: BackgroundTasks):
    """Generate image using Hunyuan Image 3.0"""

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Initialize job
    IMAGE_JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "result": None,
        "error": None
    }

    # Start background task
    background_tasks.add_task(
        process_image_generation,
        job_id,
        request
    )

    return {
        "job_id": job_id,
        "status": "queued"
    }

@app.get("/job/{job_id}")
def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in IMAGE_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    return IMAGE_JOBS[job_id]

@app.get("/download/{job_id}")
def download_image(job_id: str):
    """Download generated image"""
    if job_id not in IMAGE_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = IMAGE_JOBS[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    if not job["result"] or "image_path" not in job["result"]:
        raise HTTPException(status_code=404, detail="No image found")

    image_path = job["result"]["image_path"]
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(image_path)

async def process_image_generation(job_id: str, request: ImageGenerationRequest):
    """Process image generation in background"""
    try:
        IMAGE_JOBS[job_id]["status"] = "processing"
        IMAGE_JOBS[job_id]["progress"] = 10

        # Load model if needed
        if not load_hunyuan_image_model():
            raise Exception("Failed to load model")

        IMAGE_JOBS[job_id]["progress"] = 30

        # Generate image
        logger.info(f"Generating image for job {job_id}")
        logger.info(f"Prompt: {request.prompt}")

        # Prepare output path
        output_path = OUTPUT_DIR / f"{job_id}.png"

        # Format size as "WxH" - the model expects this format
        image_size = f"{request.width}x{request.height}"

        # Generate image using the OFFICIAL method signature
        logger.info("Calling model.generate_image...")

        # Use the exact parameters from the official script
        generated_image = model.generate_image(
            prompt=request.prompt,
            seed=request.seed,
            image_size=image_size,
            use_system_prompt=None,  # Let it use defaults
            system_prompt=None,
            bot_task=None,  # Let it use default from config
            diff_infer_steps=request.num_inference_steps,
            verbose=0,  # Minimal output
            stream=False  # Don't stream
        )

        IMAGE_JOBS[job_id]["progress"] = 80

        # Save the image
        generated_image.save(str(output_path))
        logger.info(f"Image saved to {output_path}")

        IMAGE_JOBS[job_id]["progress"] = 90

        # Save result
        IMAGE_JOBS[job_id]["result"] = {
            "image_path": str(output_path),
            "width": request.width,
            "height": request.height
        }
        IMAGE_JOBS[job_id]["status"] = "completed"
        IMAGE_JOBS[job_id]["progress"] = 100

        logger.info(f"✅ Completed job {job_id}")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(traceback.format_exc())
        IMAGE_JOBS[job_id]["status"] = "failed"
        IMAGE_JOBS[job_id]["error"] = str(e)
        IMAGE_JOBS[job_id]["progress"] = 100

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4006)