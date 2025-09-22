from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from api import sequences
# Temporarily use simple version until OpenImageIO works
from api.frames_simple import router as frames_router
from api.crowd_processor import router as crowd_router
# from api.video_processor import router as video_router

app = FastAPI(title="VFX Tracking API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sequences.router, prefix="/api/sequences", tags=["sequences"])
app.include_router(frames_router, prefix="/api/frames", tags=["frames"])
app.include_router(crowd_router, prefix="/api/crowd", tags=["crowd-enhancement"])
# app.include_router(video_router, prefix="/api/crowd", tags=["video-processing"])

@app.get("/")
def read_root():
    return {"message": "VFX Tracking Backend API", "version": "0.1.0"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}