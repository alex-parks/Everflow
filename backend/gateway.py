"""
API Gateway - Routes requests to appropriate backend services
"""

import os
import httpx
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VFX Enhancement Platform Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs from environment
HUNYUAN_IMAGE_SERVICE = os.getenv("HUNYUAN_IMAGE_SERVICE", "http://localhost:4006")
HUNYUAN3D_SERVICE = os.getenv("HUNYUAN3D_SERVICE", "http://localhost:4007")

# Request models
class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None

@app.get("/")
def read_root():
    return {
        "service": "VFX Enhancement Platform Gateway",
        "version": "1.0.0",
        "services": {
            "hunyuan_image": HUNYUAN_IMAGE_SERVICE,
            "hunyuan3d": HUNYUAN3D_SERVICE
        }
    }

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# ============ Hunyuan3D API Endpoints (Route to :4007) ============

@app.get("/api/hunyuan3d/status")
async def hunyuan3d_status():
    """Get Hunyuan3D service status"""
    try:
        # Get both service statuses
        async with httpx.AsyncClient() as client:
            # Get Hunyuan3D status
            hunyuan3d_resp = await client.get(f"{HUNYUAN3D_SERVICE}/status")
            hunyuan3d_data = hunyuan3d_resp.json()

            # Get Hunyuan Image status
            hunyuan_image_resp = await client.get(f"{HUNYUAN_IMAGE_SERVICE}/status")
            hunyuan_image_data = hunyuan_image_resp.json()

        return {
            "success": True,
            "hunyuan3d_status": hunyuan3d_data,
            "hunyuan_image_status": hunyuan_image_data
        }
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/hunyuan3d/upload-image")
async def upload_image_for_3d(file: UploadFile = File(...)):
    """Upload image for 3D generation"""
    try:
        async with httpx.AsyncClient() as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            response = await client.post(f"{HUNYUAN3D_SERVICE}/upload", files=files)
            data = response.json()

        return {
            "success": True,
            "image_id": data["image_id"],
            "filename": file.filename,
            "upload_path": data.get("path")
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/hunyuan3d/generate-3d/{image_id}")
async def generate_3d_model(image_id: str):
    """Generate 3D model from uploaded image"""
    try:
        # Use longer timeout since model loading can take ~30 seconds
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{HUNYUAN3D_SERVICE}/generate/{image_id}")
            data = response.json()

        return {
            "success": True,
            "job_id": data["job_id"],
            "status": data["status"],
            "message": "3D generation started"
        }
    except Exception as e:
        logger.error(f"3D generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hunyuan3d/job-status/{job_id}")
async def get_3d_job_status(job_id: str):
    """Get 3D generation job status with retry logic for busy service"""
    import asyncio

    # Try multiple times with shorter timeouts
    for attempt in range(3):
        try:
            timeout_duration = 5.0 + (attempt * 2.0)  # 5s, 7s, 9s
            async with httpx.AsyncClient(timeout=timeout_duration) as client:
                response = await client.get(f"{HUNYUAN3D_SERVICE}/job/{job_id}")
                response.raise_for_status()
                data = response.json()

            return {
                "success": True,
                "job_id": job_id,
                **data
            }
        except httpx.TimeoutException:
            if attempt < 2:  # Try again
                await asyncio.sleep(0.5)  # Short delay between retries
                continue
            else:  # Final attempt failed - service is very busy
                logger.warning(f"Service busy after {attempt+1} attempts, assuming job {job_id} is still processing")
                return {
                    "success": True,
                    "job_id": job_id,
                    "status": "processing",
                    "progress": 50,
                    "stage": "Processing (heavy computation in progress)",
                    "result": None,
                    "error": None
                }
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise HTTPException(status_code=404, detail="Job not found")
            logger.error(f"HTTP error getting job status for {job_id}: {e.response.status_code}")
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to get job status for {job_id}: {type(e).__name__} - {e}")
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")

@app.get("/api/hunyuan3d/download/{job_id}/{file_type}")
async def download_3d_file(job_id: str, file_type: str):
    """Download 3D file"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(f"{HUNYUAN3D_SERVICE}/download/{job_id}/{file_type}")
            response.raise_for_status()

            return StreamingResponse(
                response.iter_bytes(),
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=model_{job_id}.{file_type}"
                }
            )
    except Exception as e:
        logger.error(f"Download failed for {job_id}/{file_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ Hunyuan Image API Endpoints (Route to :4006) ============

@app.post("/api/hunyuan3d/generate-image")
async def generate_image(request: ImageGenerationRequest):
    """Generate image using Hunyuan Image 2.1"""
    try:
        # Convert request to dict (compatible with both Pydantic v1 and v2)
        try:
            request_data = request.model_dump()  # Pydantic v2
        except AttributeError:
            request_data = request.dict()  # Pydantic v1 fallback

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{HUNYUAN_IMAGE_SERVICE}/generate",
                json=request_data
            )
            response.raise_for_status()
            data = response.json()

        return {
            "success": True,
            "job_id": data["job_id"],
            "status": data["status"],
            "message": "Image generation started"
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during image generation: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Image generation failed: {type(e).__name__} - {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hunyuan3d/image-job-status/{job_id}")
async def get_image_job_status(job_id: str):
    """Get image generation job status"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.get(f"{HUNYUAN_IMAGE_SERVICE}/job/{job_id}")
            response.raise_for_status()
            data = response.json()

        return {
            "success": True,
            "job_id": job_id,
            **data
        }
    except Exception as e:
        logger.error(f"Failed to get image job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hunyuan3d/download-generated-image/{job_id}")
async def download_generated_image(job_id: str):
    """Download generated image"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(f"{HUNYUAN_IMAGE_SERVICE}/download/{job_id}")
            response.raise_for_status()

            return StreamingResponse(
                response.iter_bytes(),
                media_type="image/png",
                headers={
                    "Content-Disposition": f"attachment; filename=generated_{job_id}.png"
                }
            )
    except Exception as e:
        logger.error(f"Download image failed for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/hunyuan3d/generate-3d-from-generated-image/{image_job_id}")
async def generate_3d_from_generated_image(image_job_id: str):
    """Generate 3D model from a generated image"""
    try:
        # Use longer timeout since model loading can take ~30 seconds
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Get the image
            image_response = await client.get(f"{HUNYUAN_IMAGE_SERVICE}/download/{image_job_id}")

            if image_response.status_code != 200:
                raise HTTPException(status_code=404, detail="Generated image not found")

            # Upload to 3D service
            files = {"file": (f"generated_{image_job_id}.png", image_response.content, "image/png")}
            upload_response = await client.post(f"{HUNYUAN3D_SERVICE}/upload", files=files)
            upload_data = upload_response.json()

            # Start 3D generation
            generate_response = await client.post(f"{HUNYUAN3D_SERVICE}/generate/{upload_data['image_id']}")
            generate_data = generate_response.json()

        return {
            "success": True,
            "job_id": generate_data["job_id"],
            "status": generate_data["status"],
            "message": "3D generation from generated image started",
            "source_job_id": image_job_id
        }
    except Exception as e:
        logger.error(f"3D generation from generated image failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the sequences router if needed
# from api import sequences
# app.include_router(sequences.router, prefix="/api/sequences", tags=["sequences"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4005)