"""
Example client for WhisperX Transcription API

This script demonstrates how to use the API to transcribe audio files
with speaker diarization.
"""

import requests
import time
import json
import sys
from pathlib import Path


class WhisperXClient:
    """Client for WhisperX Transcription API."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key}
    
    def health_check(self) -> dict:
        """Check API health status."""
        response = requests.get(f"{self.base_url}/health", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def upload_audio(
        self,
        file_path: str,
        language: str = None,
        min_speakers: int = None,
        max_speakers: int = None,
        enable_diarization: bool = True,
    ) -> str:
        """
        Upload audio file for transcription.
        
        Returns:
            Task ID for tracking progress
        """
        data = {"enable_diarization": str(enable_diarization).lower()}
        
        if language:
            data["language"] = language
        if min_speakers:
            data["min_speakers"] = str(min_speakers)
        if max_speakers:
            data["max_speakers"] = str(max_speakers)
        
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f)}
            response = requests.post(
                f"{self.base_url}/api/v1/transcribe/upload",
                headers=self.headers,
                files=files,
                data=data,
            )
        
        response.raise_for_status()
        return response.json()["task_id"]
    
    def get_status(self, task_id: str) -> dict:
        """Get task status and progress."""
        response = requests.get(
            f"{self.base_url}/api/v1/transcribe/status/{task_id}",
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()
    
    def get_result(self, task_id: str) -> dict:
        """Get transcription result."""
        response = requests.get(
            f"{self.base_url}/api/v1/transcribe/result/{task_id}",
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(
        self,
        task_id: str,
        poll_interval: float = 2.0,
        timeout: float = 3600.0,
    ) -> dict:
        """
        Wait for task to complete and return result.
        
        Args:
            task_id: Task ID to monitor
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait
            
        Returns:
            Transcription result
        """
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
            
            status = self.get_status(task_id)
            
            print(f"\r[{status['status'].upper()}] Progress: {status['progress']:.1f}% - {status['message']}", end="", flush=True)
            
            if status["status"] == "completed":
                print()  # New line after completion
                return self.get_result(task_id)
            elif status["status"] == "failed":
                print()
                raise Exception(f"Task failed: {status.get('error', 'Unknown error')}")
            
            time.sleep(poll_interval)
    
    def transcribe(
        self,
        file_path: str,
        language: str = None,
        min_speakers: int = None,
        max_speakers: int = None,
        enable_diarization: bool = True,
    ) -> dict:
        """
        Convenience method to upload, wait, and return result.
        
        Args:
            file_path: Path to audio file
            language: Language code (auto-detected if None)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            enable_diarization: Enable speaker identification
            
        Returns:
            Transcription result with segments and speaker labels
        """
        print(f"Uploading: {file_path}")
        task_id = self.upload_audio(
            file_path,
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            enable_diarization=enable_diarization,
        )
        print(f"Task ID: {task_id}")
        print("Processing...")
        
        return self.wait_for_completion(task_id)


def format_transcript(result: dict, include_timestamps: bool = True) -> str:
    """Format transcription result as readable text."""
    lines = []
    
    for segment in result["segments"]:
        speaker = segment.get("speaker", "Unknown")
        text = segment["text"].strip()
        
        if include_timestamps:
            start = segment["start"]
            end = segment["end"]
            lines.append(f"[{start:.2f}s - {end:.2f}s] [{speaker}] {text}")
        else:
            lines.append(f"[{speaker}] {text}")
    
    return "\n".join(lines)


def main():
    """Example usage of WhisperX API client."""
    
    # Configuration
    API_URL = "http://localhost:8000"
    API_KEY = "your-api-key-here"  # Replace with your API key
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python client_example.py <audio_file> [language]")
        print("\nExample: python client_example.py audio.mp3 en")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Validate file exists
    if not Path(audio_file).exists():
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)
    
    # Create client
    client = WhisperXClient(API_URL, API_KEY)
    
    # Check health
    print("Checking API health...")
    try:
        health = client.health_check()
        print(f"API Status: {health['status']}")
        print(f"GPU Available: {health['gpu_available']}")
        print(f"Active Tasks: {health['active_tasks']}")
        print()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")
        sys.exit(1)
    
    # Transcribe
    try:
        result = client.transcribe(
            audio_file,
            language=language,
            enable_diarization=True,
        )
        
        print(f"\nLanguage: {result['language']}")
        print(f"Segments: {len(result['segments'])}")
        print("\n" + "=" * 50)
        print("TRANSCRIPT")
        print("=" * 50 + "\n")
        print(format_transcript(result))
        
        # Save result to JSON
        output_file = Path(audio_file).stem + "_transcript.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nFull result saved to: {output_file}")
        
    except Exception as e:
        print(f"Transcription failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
