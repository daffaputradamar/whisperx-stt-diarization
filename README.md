# WhisperX Transcription API

High-performance audio transcription API with speaker diarization, powered by WhisperX and FastAPI.

## Features

- 🎙️ **Audio Transcription**: Transcribe audio files with word-level timestamps using WhisperX
- 🗣️ **Speaker Diarization**: Identify and label different speakers in the audio
- 📊 **Progress Tracking**: Real-time progress updates for long-running transcription jobs
- 🔒 **API Key Authentication**: Secure all endpoints with API key authentication
- ⚡ **GPU Accelerated**: Optimized for CUDA-enabled GPUs with smart memory management
- 🔄 **Async Processing**: Background task processing with concurrent job management

## Requirements

- Python 3.9+
- NVIDIA GPU with CUDA support (recommended)
- HuggingFace account with accepted model terms

## Installation

### 1. Clone and setup environment

```bash
cd whisperx
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install dependencies

```bash
# Install PyTorch with CUDA support first
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 3. Configure environment

```bash
# Copy example config
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac

# Edit .env with your settings
```

### 4. HuggingFace Setup (Required for Speaker Diarization)

1. Create account at [huggingface.co](https://huggingface.co)
2. Get your token from [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Accept terms for these models:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Add your token to `.env` as `HF_TOKEN`

## Running the API

### Development

```bash
python -m app.main
```

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Note**: Use `workers=1` to properly manage GPU memory. For multiple GPUs, consider running multiple instances.

## API Usage

### Authentication

All endpoints require an API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/health
```

### Endpoints

#### Health Check
```bash
GET /health
```

#### Upload Audio for Transcription
```bash
POST /api/v1/transcribe/upload

# Form parameters:
# - file: Audio file (required)
# - language: Language code (optional, auto-detected)
# - min_speakers: Minimum speakers (optional)
# - max_speakers: Maximum speakers (optional)
# - enable_diarization: Enable speaker labels (default: true)
```

#### Check Progress
```bash
GET /api/v1/transcribe/status/{task_id}
```

#### Get Result
```bash
GET /api/v1/transcribe/result/{task_id}
```

#### Delete Task
```bash
DELETE /api/v1/transcribe/task/{task_id}
```

### Example Usage with curl

```bash
# Upload audio file
curl -X POST "http://localhost:8000/api/v1/transcribe/upload" \
  -H "X-API-Key: your-api-key" \
  -F "file=@audio.mp3" \
  -F "enable_diarization=true"

# Response: {"task_id": "abc-123", "status": "pending", "message": "..."}

# Check progress
curl "http://localhost:8000/api/v1/transcribe/status/abc-123" \
  -H "X-API-Key: your-api-key"

# Get result (when completed)
curl "http://localhost:8000/api/v1/transcribe/result/abc-123" \
  -H "X-API-Key: your-api-key"
```

### Example Usage with Python

```python
import requests

API_URL = "http://localhost:8000"
API_KEY = "your-api-key"
headers = {"X-API-Key": API_KEY}

# Upload file
with open("audio.mp3", "rb") as f:
    response = requests.post(
        f"{API_URL}/api/v1/transcribe/upload",
        headers=headers,
        files={"file": f},
        data={"enable_diarization": True}
    )

task_id = response.json()["task_id"]
print(f"Task ID: {task_id}")

# Poll for completion
import time

while True:
    status = requests.get(
        f"{API_URL}/api/v1/transcribe/status/{task_id}",
        headers=headers
    ).json()
    
    print(f"Status: {status['status']} - Progress: {status['progress']:.1f}%")
    
    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        print(f"Error: {status['error']}")
        break
    
    time.sleep(2)

# Get result
result = requests.get(
    f"{API_URL}/api/v1/transcribe/result/{task_id}",
    headers=headers
).json()

# Print transcript with speakers
for segment in result["segments"]:
    speaker = segment.get("speaker", "Unknown")
    print(f"[{speaker}] {segment['text']}")
```

## Response Format

### Task Status Response
```json
{
  "task_id": "abc-123-def",
  "status": "completed",
  "progress": 100.0,
  "message": "Transcription completed successfully",
  "created_at": "2025-01-01T00:00:00Z",
  "started_at": "2025-01-01T00:00:01Z",
  "completed_at": "2025-01-01T00:01:00Z",
  "error": null,
  "result": {...}
}
```

### Transcription Result
```json
{
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, how are you?",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Hello,", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
        {"word": "how", "start": 0.6, "end": 0.8, "speaker": "SPEAKER_00"},
        {"word": "are", "start": 0.9, "end": 1.1, "speaker": "SPEAKER_00"},
        {"word": "you?", "start": 1.2, "end": 2.5, "speaker": "SPEAKER_00"}
      ]
    },
    {
      "start": 2.8,
      "end": 4.5,
      "text": "I'm doing great, thanks!",
      "speaker": "SPEAKER_01",
      "words": [...]
    }
  ]
}
```

## Supported Audio Formats

- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- FLAC (.flac)
- OGG (.ogg)
- WebM (.webm)
- MP4 (.mp4)
- MPEG (.mpeg)
- MPGA (.mpga)
- OGA (.oga)
- Opus (.opus)

## Performance Tips

1. **GPU Memory**: Reduce `BATCH_SIZE` if you encounter OOM errors
2. **Compute Type**: Use `int8` instead of `float16` for lower memory usage
3. **Concurrent Tasks**: Adjust `MAX_CONCURRENT_TASKS` based on your GPU VRAM
4. **Model Size**: Use smaller models (`medium`, `small`) for faster processing

## API Documentation

Once running, access:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Docker Deployment

### Prerequisites

- Docker & Docker Compose
- NVIDIA Container Toolkit (for GPU support)

### GPU Deployment (Recommended)

```bash
# Create .env file with your settings
echo "API_KEYS=your-secure-api-key" > .env
echo "HF_TOKEN=your-huggingface-token" >> .env

# Build and run
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### CPU Deployment (No GPU)

```bash
# Create .env file
echo "API_KEYS=your-secure-api-key" > .env
echo "HF_TOKEN=your-huggingface-token" >> .env

# Build and run with CPU config
docker-compose -f docker-compose.cpu.yml up -d --build
```

### Docker Commands

```bash
# Rebuild after code changes
docker-compose up -d --build

# View real-time logs
docker-compose logs -f whisperx-api

# Stop and remove containers
docker-compose down

# Remove volumes (deletes cached models and uploads)
docker-compose down -v

# Check container status
docker-compose ps
```

### Environment Variables

Configure via `.env` file or pass directly:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEYS` | - | Comma-separated API keys |
| `HF_TOKEN` | - | HuggingFace token |
| `WHISPER_MODEL` | `large-v2` | Model size |
| `COMPUTE_TYPE` | `float16` | Precision type |
| `BATCH_SIZE` | `16` | Transcription batch size |
| `MAX_CONCURRENT_TASKS` | `2` | Parallel jobs |

## License

MIT License
