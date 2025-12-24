import gc
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from typing import Optional, Dict, Any, Callable
import logging
from contextlib import contextmanager

from app.config import get_settings
from app.models import (
    TranscriptionResult,
    TranscriptSegment,
    WordSegment,
    TranscriptionOptions,
)

logger = logging.getLogger(__name__)

# Patch torch.load for PyTorch 2.6+ compatibility
# Models from trusted sources (huggingface, official repos) need weights_only=False
_original_torch_load = torch.load

def _patched_torch_load(f, *args, **kwargs):
    """Patched torch.load that defaults to weights_only=False for model compatibility."""
    # If weights_only not specified, default to False for model loading
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)

torch.load = _patched_torch_load


class WhisperXService:
    """
    High-performance WhisperX service with model caching and memory management.
    
    This service handles transcription, alignment, and speaker diarization
    with optimized GPU memory usage.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._whisper_model = None
        self._align_models: Dict[str, tuple] = {}  # Cache align models by language
        self._diarize_model = None
        self._model_lock = False
        
    def _clear_gpu_memory(self):
        """Clear GPU memory cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    @contextmanager
    def _model_context(self):
        """Context manager for model operations with memory cleanup."""
        try:
            yield
        finally:
            self._clear_gpu_memory()
    
    def load_whisper_model(self):
        """Load and cache the Whisper model."""
        if self._whisper_model is None:
            logger.info(f"Loading Whisper model: {self.settings.WHISPER_MODEL}")
            self._whisper_model = whisperx.load_model(
                self.settings.WHISPER_MODEL,
                self.settings.DEVICE,
                compute_type=self.settings.COMPUTE_TYPE,
            )
            logger.info("Whisper model loaded successfully")
        return self._whisper_model
    
    def load_align_model(self, language_code: str):
        """Load and cache alignment model for specific language."""
        if language_code not in self._align_models:
            logger.info(f"Loading alignment model for language: {language_code}")
            model_a, metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.settings.DEVICE,
            )
            self._align_models[language_code] = (model_a, metadata)
            logger.info(f"Alignment model for {language_code} loaded successfully")
        return self._align_models[language_code]
    
    def load_diarization_model(self):
        """Load and cache the diarization model."""
        if self._diarize_model is None:
            if not self.settings.HF_TOKEN:
                raise ValueError(
                    "HuggingFace token (HF_TOKEN) is required for speaker diarization. "
                    "Set it in your .env file or environment variables."
                )
            logger.info("Loading diarization model")
            self._diarize_model = DiarizationPipeline(
                use_auth_token=self.settings.HF_TOKEN,
                device=self.settings.DEVICE,
            )
            logger.info("Diarization model loaded successfully")
        return self._diarize_model
    
    def unload_models(self, keep_whisper: bool = True):
        """
        Unload models to free GPU memory.
        
        Args:
            keep_whisper: Keep Whisper model loaded (most frequently used)
        """
        if not keep_whisper and self._whisper_model is not None:
            del self._whisper_model
            self._whisper_model = None
            
        for lang, (model, _) in self._align_models.items():
            del model
        self._align_models.clear()
        
        if self._diarize_model is not None:
            del self._diarize_model
            self._diarize_model = None
            
        self._clear_gpu_memory()
        logger.info("Models unloaded and GPU memory cleared")
    
    async def transcribe(
        self,
        audio_path: str,
        options: TranscriptionOptions,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> TranscriptionResult:
        """
        Perform full transcription pipeline with alignment and diarization.
        
        Args:
            audio_path: Path to the audio file
            options: Transcription options
            progress_callback: Callback function for progress updates (status, percentage)
            
        Returns:
            TranscriptionResult with segments and speaker information
        """
        def update_progress(status: str, progress: float):
            if progress_callback:
                progress_callback(status, progress)
        
        try:
            # Step 1: Load audio
            update_progress("loading_audio", 5.0)
            logger.info(f"Loading audio file: {audio_path}")
            audio = whisperx.load_audio(audio_path)
            
            # Step 2: Transcribe
            update_progress("transcribing", 10.0)
            logger.info("Starting transcription")
            model = self.load_whisper_model()
            
            transcribe_options = {"batch_size": self.settings.BATCH_SIZE}
            if options.language:
                transcribe_options["language"] = options.language
                
            result = model.transcribe(audio, **transcribe_options)
            detected_language = result["language"]
            logger.info(f"Transcription complete. Detected language: {detected_language}")
            update_progress("transcribing", 40.0)
            
            # Step 3: Align
            update_progress("aligning", 45.0)
            logger.info("Starting alignment")
            try:
                model_a, metadata = self.load_align_model(detected_language)
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    self.settings.DEVICE,
                    return_char_alignments=options.return_char_alignments,
                )
                logger.info("Alignment complete")
            except Exception as e:
                logger.warning(f"Alignment failed for language {detected_language}: {e}")
                # Continue without alignment
            update_progress("aligning", 60.0)
            
            # Step 4: Diarization (if enabled)
            if options.enable_diarization:
                update_progress("diarizing", 65.0)
                logger.info("Starting speaker diarization")
                try:
                    diarize_model = self.load_diarization_model()
                    
                    diarize_kwargs = {}
                    if options.min_speakers:
                        diarize_kwargs["min_speakers"] = options.min_speakers
                    if options.max_speakers:
                        diarize_kwargs["max_speakers"] = options.max_speakers
                    
                    diarize_segments = diarize_model(audio, **diarize_kwargs)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    logger.info("Speaker diarization complete")
                except Exception as e:
                    logger.warning(f"Diarization failed: {e}")
                    # Continue without speaker labels
            
            update_progress("processing", 90.0)
            
            # Step 5: Format result
            segments = []
            for seg in result.get("segments", []):
                words = None
                if "words" in seg:
                    words = [
                        WordSegment(
                            word=w.get("word", ""),
                            start=w.get("start", 0.0),
                            end=w.get("end", 0.0),
                            score=w.get("score"),
                            speaker=w.get("speaker"),
                        )
                        for w in seg["words"]
                    ]
                
                segments.append(
                    TranscriptSegment(
                        start=seg.get("start", 0.0),
                        end=seg.get("end", 0.0),
                        text=seg.get("text", ""),
                        speaker=seg.get("speaker"),
                        words=words,
                    )
                )
            
            update_progress("completed", 100.0)
            
            return TranscriptionResult(
                language=detected_language,
                segments=segments,
            )
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
        finally:
            self._clear_gpu_memory()


# Global service instance (singleton pattern)
_whisperx_service: Optional[WhisperXService] = None


def get_whisperx_service() -> WhisperXService:
    """Get or create the WhisperX service singleton."""
    global _whisperx_service
    if _whisperx_service is None:
        _whisperx_service = WhisperXService()
    return _whisperx_service
