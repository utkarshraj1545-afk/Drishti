from fastapi import FastAPI
from app.routes.change_detection import router as change_detection_router

app = FastAPI(
    title="SatQuery AI",
    description="AI-powered satellite change detection API",
    version="1.0.0",
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
        "status": "healthy"
    }