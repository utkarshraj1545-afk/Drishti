
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.change_detection import router as change_detection_router

app = FastAPI(
    title="SatQuery AI Backend",
    description="AI-powered satellite change detection API",
    version="1.0.0",
)

allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://prakriti-blue.vercel.app",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

vercel_url = os.getenv("FRONTEND_URL")
if vercel_url:
    vercel_url = vercel_url.rstrip("/")
    if vercel_url not in allowed_origins:
        allowed_origins.append(vercel_url)


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(change_detection_router)


@app.get("/")
def root():
    return {
        "message": "SatQuery AI backend is running",
        "status": "ok",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }

