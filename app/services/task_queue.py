import asyncio
import uuid
from datetime import datetime
from typing import Dict, Optional, Callable, Any
from collections import OrderedDict
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

from app.models import (
    TaskStatus,
    TaskProgress,
    TranscriptionOptions,
    TranscriptionResult,
)
from app.services.whisperx_service import get_whisperx_service
from app.config import get_settings

logger = logging.getLogger(__name__)


class TaskQueue:
    """
    Thread-safe task queue for managing transcription jobs.
    
    Features:
    - Concurrent task limiting
    - Progress tracking
    - Task persistence in memory (can be extended to Redis/DB)
    - Automatic cleanup of old tasks
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._tasks: OrderedDict[str, TaskProgress] = OrderedDict()
        self._task_results: Dict[str, TranscriptionResult] = {}
        self._lock = threading.Lock()
        self._semaphore = asyncio.Semaphore(self.settings.MAX_CONCURRENT_TASKS)
        self._executor = ThreadPoolExecutor(max_workers=self.settings.MAX_CONCURRENT_TASKS)
        self._running_tasks: set = set()
        
    def create_task(self, audio_path: str, options: TranscriptionOptions) -> str:
        """
        Create a new transcription task.
        
        Args:
            audio_path: Path to the uploaded audio file
            options: Transcription options
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        with self._lock:
            self._tasks[task_id] = TaskProgress(
                task_id=task_id,
                status=TaskStatus.PENDING,
                progress=0.0,
                message="Task queued",
                created_at=datetime.utcnow(),
            )
            # Store options with task for later processing
            self._tasks[task_id]._audio_path = audio_path  # type: ignore
            self._tasks[task_id]._options = options  # type: ignore
            
        logger.info(f"Created task {task_id}")
        return task_id
    
    def get_task(self, task_id: str) -> Optional[TaskProgress]:
        """Get task progress by ID."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task_id in self._task_results:
                task.result = self._task_results[task_id]
            return task
    
    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Update task progress."""
        with self._lock:
            if task_id not in self._tasks:
                return
                
            task = self._tasks[task_id]
            
            if status is not None:
                task.status = status
                if status == TaskStatus.PROCESSING and task.started_at is None:
                    task.started_at = datetime.utcnow()
                elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    task.completed_at = datetime.utcnow()
                    
            if progress is not None:
                task.progress = progress
                
            if message is not None:
                task.message = message
                
            if error is not None:
                task.error = error
    
    def set_result(self, task_id: str, result: TranscriptionResult):
        """Store task result."""
        with self._lock:
            self._task_results[task_id] = result
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        with self._lock:
            pending = sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)
            processing = sum(1 for t in self._tasks.values() if t.status in (
                TaskStatus.PROCESSING, TaskStatus.TRANSCRIBING, 
                TaskStatus.ALIGNING, TaskStatus.DIARIZING
            ))
            completed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED)
            
        return {
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "total": len(self._tasks),
        }
    
    async def process_task(self, task_id: str):
        """
        Process a transcription task asynchronously.
        
        Uses a semaphore to limit concurrent processing.
        """
        async with self._semaphore:
            self._running_tasks.add(task_id)
            
            try:
                with self._lock:
                    task = self._tasks.get(task_id)
                    if not task:
                        return
                    audio_path = getattr(task, '_audio_path', None)
                    options = getattr(task, '_options', TranscriptionOptions())
                
                if not audio_path:
                    self.update_task(
                        task_id,
                        status=TaskStatus.FAILED,
                        error="Audio path not found",
                    )
                    return
                
                self.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    progress=0.0,
                    message="Starting transcription",
                )
                
                def progress_callback(status: str, progress: float):
                    status_map = {
                        "loading_audio": TaskStatus.PROCESSING,
                        "transcribing": TaskStatus.TRANSCRIBING,
                        "aligning": TaskStatus.ALIGNING,
                        "diarizing": TaskStatus.DIARIZING,
                        "processing": TaskStatus.PROCESSING,
                        "completed": TaskStatus.PROCESSING,
                    }
                    self.update_task(
                        task_id,
                        status=status_map.get(status, TaskStatus.PROCESSING),
                        progress=progress,
                        message=f"Status: {status}",
                    )
                
                # Run transcription in thread pool to not block event loop
                service = get_whisperx_service()
                loop = asyncio.get_event_loop()
                
                result = await service.transcribe(
                    audio_path,
                    options,
                    progress_callback,
                )
                
                self.set_result(task_id, result)
                self.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    progress=100.0,
                    message="Transcription completed successfully",
                )
                
                logger.info(f"Task {task_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                self.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    error=str(e),
                    message="Transcription failed",
                )
            finally:
                self._running_tasks.discard(task_id)
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Remove tasks older than specified hours."""
        cutoff = datetime.utcnow()
        
        with self._lock:
            old_tasks = []
            for task_id, task in self._tasks.items():
                age = (cutoff - task.created_at).total_seconds() / 3600
                if age > max_age_hours and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    old_tasks.append(task_id)
            
            for task_id in old_tasks:
                del self._tasks[task_id]
                if task_id in self._task_results:
                    del self._task_results[task_id]
                    
            logger.info(f"Cleaned up {len(old_tasks)} old tasks")


# Global task queue instance
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Get or create the task queue singleton."""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue
