from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    TRANSCRIBING = "transcribing"
    ALIGNING = "aligning"
    DIARIZING = "diarizing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptionOptions(BaseModel):
    """Options for transcription job."""
    language: Optional[str] = Field(None, description="Language code (e.g., 'en', 'es'). Auto-detected if not provided.")
    min_speakers: Optional[int] = Field(None, ge=1, description="Minimum number of speakers")
    max_speakers: Optional[int] = Field(None, ge=1, description="Maximum number of speakers")
    enable_diarization: bool = Field(True, description="Enable speaker diarization")
    return_char_alignments: bool = Field(False, description="Return character-level alignments")


class WordSegment(BaseModel):
    """Word-level segment with timing."""
    word: str
    start: float
    end: float
    score: Optional[float] = None
    speaker: Optional[str] = None


class TranscriptSegment(BaseModel):
    """Transcript segment with speaker information."""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: Optional[List[WordSegment]] = None


class TranscriptionResult(BaseModel):
    """Complete transcription result."""
    language: str
    segments: List[TranscriptSegment]
    word_segments: Optional[List[WordSegment]] = None


class TaskProgress(BaseModel):
    """Task progress information."""
    task_id: str
    status: TaskStatus
    progress: float = Field(0.0, ge=0.0, le=100.0, description="Progress percentage")
    message: str = ""
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[TranscriptionResult] = None


class TaskCreateResponse(BaseModel):
    """Response when creating a new transcription task."""
    task_id: str
    status: TaskStatus
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    gpu_available: bool
    active_tasks: int
    pending_tasks: int
