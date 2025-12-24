import torch
from fastapi import APIRouter

from app.config import get_settings
from app.models import HealthResponse
from app.services.task_queue import get_task_queue

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health status and system information.",
)
async def health_check():
    """
    Health check endpoint.
    
    Returns system status including GPU availability and task queue stats.
    """
    settings = get_settings()
    task_queue = get_task_queue()
    stats = task_queue.get_stats()
    
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        gpu_available=torch.cuda.is_available(),
        active_tasks=stats["processing"],
        pending_tasks=stats["pending"],
    )


@router.get(
    "/",
    summary="API Root",
    description="Welcome endpoint with API information.",
)
async def root():
    """Root endpoint with API information."""
    settings = get_settings()
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
