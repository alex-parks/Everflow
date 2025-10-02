from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from api import sequences
from api import hunyuan3d_api
# from api.video_processor import router as video_router

app = FastAPI(title="VFX Enhancement Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sequences.router, prefix="/api/sequences", tags=["sequences"])
app.include_router(hunyuan3d_api.router, prefix="/api/hunyuan3d", tags=["3d-generation"])
# app.include_router(video_router, prefix="/api/video", tags=["video-processing"])

@app.get("/")
def read_root():
    return {"message": "VFX Tracking Backend API", "version": "0.1.0"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}