import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import health, transcribe
from app.services.whisperx_service import get_whisperx_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for model loading and cleanup.
    """
    settings = get_settings()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Optionally preload models on startup for faster first request
    # Uncomment if you want models loaded at startup (uses more memory but faster first request)
    # service = get_whisperx_service()
    # service.load_whisper_model()
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down, cleaning up resources...")
    service = get_whisperx_service()
    service.unload_models(keep_whisper=False)
    logger.info("Cleanup complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="""
## WhisperX Transcription API

High-performance audio transcription API with speaker diarization powered by WhisperX.

### Features
- 🎙️ **Audio Transcription**: Transcribe audio files with word-level timestamps
- 🗣️ **Speaker Diarization**: Identify and label different speakers
- 📊 **Progress Tracking**: Real-time progress updates for long-running jobs
- 🔒 **API Key Authentication**: Secure endpoints with API key
- ⚡ **GPU Accelerated**: Optimized for CUDA-enabled GPUs

### Usage
1. Upload an audio file to `/api/v1/transcribe/upload`
2. Get the `task_id` from the response
3. Poll `/api/v1/transcribe/status/{task_id}` to check progress
4. Retrieve results from `/api/v1/transcribe/result/{task_id}` when completed

### Authentication
All endpoints require an API key. Include it in the `X-API-Key` header.
        """,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health.router)
    app.include_router(transcribe.router)
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1,  # Single worker for GPU memory management
    )
