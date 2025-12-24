import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Settings
    APP_NAME: str = "WhisperX Transcription API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Keys (comma-separated list of valid API keys)
    API_KEYS: str = "your-api-key-here"
    
    # HuggingFace Token for speaker diarization
    HF_TOKEN: str = ""
    
    # WhisperX Settings
    WHISPER_MODEL: str = "large-v2"
    DEVICE: str = "cuda"
    COMPUTE_TYPE: str = "float16"  # Use "int8" for lower GPU memory
    BATCH_SIZE: int = 16  # Reduce if low on GPU memory
    
    # Storage Settings
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE_MB: int = 500  # Maximum upload file size in MB
    
    # Worker Settings
    MAX_CONCURRENT_TASKS: int = 2  # Limit concurrent transcriptions
    TASK_TIMEOUT_SECONDS: int = 3600  # 1 hour timeout
    
    # Cleanup Settings
    CLEANUP_AFTER_HOURS: int = 24  # Delete old files after this many hours
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
