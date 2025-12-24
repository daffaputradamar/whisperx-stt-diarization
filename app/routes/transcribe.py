import os
import shutil
import aiofiles
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import uuid
from datetime import datetime

from app.auth import verify_api_key
from app.config import get_settings
from app.models import (
    TaskStatus,
    TaskProgress,
    TaskCreateResponse,
    TranscriptionOptions,
    TranscriptionResult,
)
from app.services.task_queue import get_task_queue

router = APIRouter(prefix="/api/v1/transcribe", tags=["Transcription"])

# Supported audio formats
SUPPORTED_FORMATS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", 
    ".mp4", ".mpeg", ".mpga", ".oga", ".opus"
}


def get_upload_dir() -> str:
    """Get and create upload directory."""
    settings = get_settings()
    upload_dir = settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir


@router.post(
    "/upload",
    response_model=TaskCreateResponse,
    summary="Upload audio file for transcription",
    description="Upload an audio file to start a transcription job with optional speaker diarization.",
)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(None, description="Language code (auto-detected if not provided)"),
    min_speakers: Optional[int] = Form(None, ge=1, description="Minimum number of speakers"),
    max_speakers: Optional[int] = Form(None, ge=1, description="Maximum number of speakers"),
    enable_diarization: bool = Form(True, description="Enable speaker diarization"),
    api_key: str = Depends(verify_api_key),
):
    """
    Upload an audio file for transcription.
    
    The file will be processed asynchronously. Use the returned task_id 
    to check progress and retrieve results.
    
    Supported formats: mp3, wav, m4a, flac, ogg, webm, mp4, mpeg, mpga, oga, opus
    """
    settings = get_settings()
    
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(SUPPORTED_FORMATS)}",
        )
    
    # Check file size (read in chunks to avoid memory issues)
    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    file_size = 0
    
    # Save file to disk
    upload_dir = get_upload_dir()
    file_id = str(uuid.uuid4())
    file_path = os.path.join(upload_dir, f"{file_id}{ext}")
    
    try:
        async with aiofiles.open(file_path, 'wb') as out_file:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                file_size += len(chunk)
                if file_size > max_size:
                    await out_file.close()
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB",
                    )
                await out_file.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Create transcription options
    options = TranscriptionOptions(
        language=language,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        enable_diarization=enable_diarization,
    )
    
    # Create task
    task_queue = get_task_queue()
    task_id = task_queue.create_task(file_path, options)
    
    # Start processing in background
    background_tasks.add_task(task_queue.process_task, task_id)
    
    return TaskCreateResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="File uploaded successfully. Transcription job queued.",
    )


@router.get(
    "/status/{task_id}",
    response_model=TaskProgress,
    summary="Check transcription progress",
    description="Get the current status and progress of a transcription job.",
)
async def get_task_status(
    task_id: str,
    api_key: str = Depends(verify_api_key),
):
    """
    Check the status of a transcription task.
    
    Returns progress percentage, current status, and result when completed.
    """
    task_queue = get_task_queue()
    task = task_queue.get_task(task_id)
    
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found. It may have expired or never existed.",
        )
    
    return task


@router.get(
    "/result/{task_id}",
    response_model=TranscriptionResult,
    summary="Get transcription result",
    description="Get the full transcription result for a completed job.",
)
async def get_task_result(
    task_id: str,
    api_key: str = Depends(verify_api_key),
):
    """
    Get the transcription result for a completed task.
    
    Returns 404 if task not found, 202 if still processing, or the result if completed.
    """
    task_queue = get_task_queue()
    task = task_queue.get_task(task_id)
    
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found.",
        )
    
    if task.status == TaskStatus.FAILED:
        raise HTTPException(
            status_code=500,
            detail=f"Task failed: {task.error}",
        )
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=202,
            detail=f"Task is still {task.status.value}. Progress: {task.progress:.1f}%",
        )
    
    if task.result is None:
        raise HTTPException(
            status_code=500,
            detail="Task completed but result is missing.",
        )
    
    return task.result


@router.delete(
    "/task/{task_id}",
    summary="Cancel or delete a task",
    description="Cancel a pending task or delete a completed task and its files.",
)
async def delete_task(
    task_id: str,
    api_key: str = Depends(verify_api_key),
):
    """
    Delete a task and clean up associated files.
    
    Note: Running tasks cannot be cancelled immediately but will be marked for cleanup.
    """
    task_queue = get_task_queue()
    task = task_queue.get_task(task_id)
    
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found.",
        )
    
    # Clean up audio file if exists
    audio_path = getattr(task, '_audio_path', None)
    if audio_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except Exception:
            pass
    
    # Remove from queue
    with task_queue._lock:
        if task_id in task_queue._tasks:
            del task_queue._tasks[task_id]
        if task_id in task_queue._task_results:
            del task_queue._task_results[task_id]
    
    return {"message": f"Task '{task_id}' deleted successfully"}
