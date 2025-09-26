#!/usr/bin/env python3
"""
Lightning Fast MLX Whisper API Server
Optimized for Apple Silicon M4 with 27x real-time performance
"""

import os
import tempfile
import time
import sys
import subprocess
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Lazy import for faster startup
_whisper_model = None

app = FastAPI(
    title="Lightning MLX Whisper API",
    description="Ultra-fast speech-to-text API using Apple's MLX framework",
    version="1.0.0"
)

# Enable CORS for web usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DEFAULT_MODEL = "distil-medium.en"
DEFAULT_BATCH_SIZE = 12
SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".mov"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Model mapping for Lightning Whisper MLX
MODEL_MAPPING = {
    "tiny": "tiny",
    "tiny.en": "tiny.en",
    "base": "base",
    "base.en": "base.en",
    "small": "small",
    "small.en": "small.en",
    "medium": "medium",
    "medium.en": "medium.en",
    "large": "large",
    "large-v1": "large-v1",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "distil-small.en": "distil-small.en",
    "distil-medium.en": "distil-medium.en",
    "distil-large-v2": "distil-large-v2",
    "distil-large-v3": "distil-large-v3",
    "turbo": "distil-large-v3"
}

# Cache for loaded models
_model_cache = {}

def format_size(bytes_size):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}TB"

def monitor_download_progress(cache_dir: str, model_name: str, downloading_flag):
    """Monitor cache directory for download progress"""
    import glob
    import time

    model_pattern = f"*{model_name}*"
    initial_files = set()

    # Get initial state
    for pattern_path in glob.glob(os.path.join(cache_dir, "**", model_pattern), recursive=True):
        if os.path.isfile(pattern_path):
            try:
                initial_files.add((pattern_path, os.path.getsize(pattern_path)))
            except (OSError, IOError):
                pass

    last_total_size = sum(size for _, size in initial_files)

    while downloading_flag[0]:
        time.sleep(0.5)
        current_files = set()
        current_total_size = 0

        # Scan for files
        for pattern_path in glob.glob(os.path.join(cache_dir, "**", model_pattern), recursive=True):
            if os.path.isfile(pattern_path):
                try:
                    size = os.path.getsize(pattern_path)
                    current_files.add((pattern_path, size))
                    current_total_size += size
                except (OSError, IOError):
                    pass

        # Check for new or growing files
        new_files = current_files - initial_files
        if new_files or current_total_size > last_total_size:
            if current_total_size > last_total_size:
                downloaded = current_total_size - last_total_size
                print(f"\r📥 Downloaded: {format_size(current_total_size)} (+{format_size(downloaded)})", end='', flush=True)
                last_total_size = current_total_size

            if new_files:
                for file_path, size in new_files:
                    filename = os.path.basename(file_path)
                    print(f"\r📁 New file: {filename} ({format_size(size)})", end='', flush=True)
                initial_files.update(new_files)

def get_whisper_model(model: str = DEFAULT_MODEL, batch_size: int = DEFAULT_BATCH_SIZE, progress_callback=None):
    """Lazy load the whisper model with caching and optional progress callback"""
    global _model_cache

    # Map model name
    mapped_model = MODEL_MAPPING.get(model, model)
    cache_key = f"{mapped_model}_{batch_size}"

    if cache_key not in _model_cache:
        try:
            from lightning_whisper_mlx import LightningWhisperMLX
            import threading
            import time
            import glob

            # Set cache directory to ensure persistence
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            os.makedirs(cache_dir, exist_ok=True)

            # Check if model files already exist in cache
            model_pattern = f"*{mapped_model}*"
            existing_files = []
            for pattern_path in glob.glob(os.path.join(cache_dir, "**", model_pattern), recursive=True):
                if os.path.isfile(pattern_path):
                    existing_files.append(pattern_path)

            if existing_files:
                total_size = sum(os.path.getsize(f) for f in existing_files)
                if progress_callback:
                    progress_callback(f"📦 Model '{mapped_model}' already cached ({format_size(total_size)})\n")
                    progress_callback(f"🔄 Loading cached model...\n")
                else:
                    print(f"📦 Model '{mapped_model}' already cached ({format_size(total_size)})")
                    print(f"🔄 Loading cached model...")
            else:
                if progress_callback:
                    progress_callback(f"🔄 Loading model '{mapped_model}' (first download)...\n")
                else:
                    print(f"🔄 Loading model '{mapped_model}' (first download)...")

            # Use a flag for thread communication
            downloading_flag = [True]
            last_update = [time.time()]

            def progress_monitor():
                while downloading_flag[0]:
                    current_files = set()
                    current_total_size = 0

                    # Scan for files
                    import glob
                    model_pattern = f"*{mapped_model}*"
                    for pattern_path in glob.glob(os.path.join(cache_dir, "**", model_pattern), recursive=True):
                        if os.path.isfile(pattern_path):
                            try:
                                size = os.path.getsize(pattern_path)
                                current_files.add((pattern_path, size))
                                current_total_size += size
                            except (OSError, IOError):
                                pass

                    # Send progress update every 2 seconds
                    now = time.time()
                    if now - last_update[0] > 2.0 and current_total_size > 0:
                        if progress_callback:
                            progress_callback(f"📥 Downloaded: {format_size(current_total_size)}\n")
                        last_update[0] = now

                    time.sleep(0.5)

            # Start progress monitor thread if callback provided
            if progress_callback:
                progress_thread = threading.Thread(target=progress_monitor, daemon=True)
                progress_thread.start()

            try:
                _model_cache[cache_key] = LightningWhisperMLX(
                    model=mapped_model,
                    batch_size=batch_size,
                    quant=None
                )
            finally:
                downloading_flag[0] = False
                if progress_callback:
                    progress_thread.join(timeout=1.0)

            success_msg = f"✅ Model '{mapped_model}' loaded successfully!\n"
            if progress_callback:
                progress_callback(success_msg)
            else:
                print(success_msg.strip())

        except ImportError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Lightning Whisper MLX not available: {e}. Install with 'uv add lightning-whisper-mlx'"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load Whisper model '{mapped_model}': {e}"
            )

    else:
        # Model already loaded in runtime cache
        if progress_callback:
            progress_callback(f"⚡ Model '{mapped_model}' already loaded in memory\n")

    return _model_cache[cache_key]

def format_time(seconds: float) -> str:
    """Format seconds to SRT/VTT time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

def format_time_vtt(seconds: float) -> str:
    """Format seconds to VTT time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def generate_srt(segments: list) -> str:
    """Generate SRT subtitle format"""
    srt_content = []
    for i, segment in enumerate(segments, 1):
        # Handle both dict format and list format [start_ms, end_ms, text]
        if isinstance(segment, dict):
            start_time = format_time(segment.get('start', 0))
            end_time = format_time(segment.get('end', segment.get('start', 0) + 3))
            text = segment.get('text', '').strip()
        elif isinstance(segment, list) and len(segment) >= 3:
            # Format: [start_ms, end_ms, text]
            start_time = format_time(segment[0] / 1000.0)  # Convert ms to seconds
            end_time = format_time(segment[1] / 1000.0)
            text = segment[2].strip()
        else:
            continue  # Skip invalid segments

        srt_content.append(f"{i}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(text)
        srt_content.append("")

    return "\n".join(srt_content)

def generate_vtt(segments: list) -> str:
    """Generate VTT subtitle format"""
    vtt_content = ["WEBVTT", ""]

    for segment in segments:
        # Handle both dict format and list format [start_ms, end_ms, text]
        if isinstance(segment, dict):
            start_time = format_time_vtt(segment.get('start', 0))
            end_time = format_time_vtt(segment.get('end', segment.get('start', 0) + 3))
            text = segment.get('text', '').strip()
        elif isinstance(segment, list) and len(segment) >= 3:
            # Format: [start_ms, end_ms, text]
            start_time = format_time_vtt(segment[0] / 1000.0)  # Convert ms to seconds
            end_time = format_time_vtt(segment[1] / 1000.0)
            text = segment[2].strip()
        else:
            continue  # Skip invalid segments

        vtt_content.append(f"{start_time} --> {end_time}")
        vtt_content.append(text)
        vtt_content.append("")

    return "\n".join(vtt_content)

def generate_markdown(result: dict, model: str, processing_time: float, filename: str = None) -> str:
    """Generate Markdown with YAML frontmatter"""
    md_content = []

    # YAML frontmatter - solo i dati utili
    md_content.append("---")
    if filename:
        md_content.append(f"source_file: {filename}")
    md_content.append(f"model: {model}")
    md_content.append(f"language: {result.get('language', 'auto')}")
    md_content.append(f"processing_time: {processing_time:.2f}")
    md_content.append(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md_content.append("---")
    md_content.append("")

    # Clean text content - just add the raw text
    text = result.get('text', '').strip()
    if text:
        md_content.append(text)

    return "\n".join(md_content)

async def transcribe_with_progress(file, model, language, temperature, response_format):
    """Generator function for streaming transcription with progress updates"""
    import asyncio
    import queue
    import threading

    # Create a queue for progress messages
    progress_queue = queue.Queue()

    def progress_callback(message):
        progress_queue.put(message)

    # Validate file in stream
    if not file.filename:
        yield "❌ Error: No file provided\n"
        return

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        yield f"❌ Error: Unsupported format: {file_ext}. Supported: {', '.join(SUPPORTED_FORMATS)}\n"
        return

    # Read file content
    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        yield f"❌ Error: File too large: {len(file_content)} bytes. Max: {MAX_FILE_SIZE} bytes\n"
        return

    yield f"📁 File: {file.filename} ({format_size(len(file_content))})\n"
    yield f"🤖 Model: {model}\n\n"

    # Run model loading and transcription in a thread
    result_container = {}
    error_container = {}

    def run_transcription():
        try:
            start_time = time.time()

            # Load model with progress callback
            whisper = get_whisper_model(model, progress_callback=progress_callback)

            # Create temp file for transcription
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_file.flush()

                progress_callback("🎯 Starting transcription...\n")

                # Prepare transcription options
                transcribe_options = {}
                if language:
                    transcribe_options['language'] = language
                if temperature != 0.0:
                    transcribe_options['temperature'] = temperature

                # Transcribe
                result = whisper.transcribe(tmp_file.name, **transcribe_options)

                # Clean up
                os.unlink(tmp_file.name)

                processing_time = time.time() - start_time
                result_container['result'] = result
                result_container['processing_time'] = processing_time

                progress_callback("✅ Transcription completed!\n")

        except Exception as e:
            error_container['error'] = str(e)
            progress_callback(f"❌ Error: {e}\n")

    # Start transcription thread
    transcription_thread = threading.Thread(target=run_transcription)
    transcription_thread.start()

    # Stream progress updates
    while transcription_thread.is_alive():
        try:
            # Get progress messages with timeout
            message = progress_queue.get(timeout=0.5)
            yield message
        except queue.Empty:
            # Send a heartbeat to keep connection alive
            yield "⏳ Processing...\n"
            continue

    # Get any remaining messages
    while not progress_queue.empty():
        try:
            message = progress_queue.get_nowait()
            yield message
        except queue.Empty:
            break

    # Wait for thread to complete
    transcription_thread.join()

    # Check for errors
    if 'error' in error_container:
        return

    # Format and yield final result
    if 'result' in result_container:
        result = result_container['result']
        processing_time = result_container['processing_time']

        yield f"⏱️  Processing time: {processing_time:.2f}s\n"

        if 'duration' in result and result['duration'] > 0:
            real_time_factor = result['duration'] / processing_time
            yield f"⚡ Real-time factor: {real_time_factor:.1f}x\n"

        yield "\n" + "="*50 + "\n"
        yield "📝 TRANSCRIPTION RESULT:\n"
        yield "="*50 + "\n\n"

        # Format response based on requested format
        if response_format == "text":
            yield result.get('text', '')
        elif response_format == "srt":
            segments = result.get('segments', [])
            if segments:
                yield generate_srt(segments)
            else:
                text = result.get('text', '')
                yield f"1\n00:00:00,000 --> 00:00:10,000\n{text}\n"
        elif response_format == "vtt":
            segments = result.get('segments', [])
            if segments:
                yield generate_vtt(segments)
            else:
                text = result.get('text', '')
                yield f"WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n{text}\n"
        elif response_format == "md" or response_format == "markdown":
            yield generate_markdown(result, model, processing_time, filename=file.filename)
        else:
            # JSON format
            import json
            json_result = {
                "text": result.get('text', ''),
                "language": result.get('language', language or 'auto'),
                "model": model,
                "processing_time": round(processing_time, 2),
                "file_duration": result.get('duration', 0),
                "real_time_factor": round(result.get('duration', 0) / processing_time, 1) if processing_time > 0 else 0,
                "segments": result.get('segments', []),
                "words": result.get('words', []) if 'words' in result else None
            }
            yield json.dumps(json_result, indent=2)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Lightning MLX Whisper API is running",
        "version": "1.0.0",
        "hardware": "Apple Silicon with MLX acceleration"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        import mlx.core as mx
        return {
            "status": "healthy",
            "mlx_available": True,
            "mlx_device": str(mx.default_device()),
            "models_loaded": _whisper_model is not None
        }
    except ImportError:
        return {
            "status": "degraded",
            "mlx_available": False,
            "error": "MLX not available"
        }

@app.get("/models")
async def list_models():
    """List available Whisper models"""
    available_models = list(MODEL_MAPPING.keys())
    loaded_models = list(_model_cache.keys())

    return {
        "available_models": available_models,
        "loaded_models": loaded_models,
        "default_model": DEFAULT_MODEL,
        "recommended": "distil-medium.en (fast and accurate)",
        "cache_location": os.path.expanduser("~/.cache/huggingface/hub")
    }

@app.post("/models/preload")
async def preload_model(
    model: str = Form(...),
    batch_size: Optional[int] = Form(DEFAULT_BATCH_SIZE)
):
    """Preload a model to cache for faster first use"""
    if model not in MODEL_MAPPING:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' not supported. Available models: {list(MODEL_MAPPING.keys())}"
        )

    try:
        start_time = time.time()
        whisper_model = get_whisper_model(model, batch_size)
        load_time = time.time() - start_time

        return {
            "status": "success",
            "model": model,
            "batch_size": batch_size,
            "load_time": round(load_time, 2),
            "cached": True,
            "cache_key": f"{MODEL_MAPPING[model]}_{batch_size}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preload model: {e}")

@app.get("/cache/status")
async def cache_status():
    """Get cache status and statistics"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    cache_size = 0
    cache_files = 0

    if os.path.exists(cache_dir):
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    cache_size += os.path.getsize(file_path)
                    cache_files += 1
                except (OSError, IOError):
                    pass

    return {
        "cache_directory": cache_dir,
        "cache_size_bytes": cache_size,
        "cache_size_mb": round(cache_size / (1024 * 1024), 2),
        "cached_files": cache_files,
        "loaded_models": list(_model_cache.keys()),
        "available_space": "Unknown"  # Could add disk space check
    }

@app.post("/cache/clear")
async def clear_cache(
    model: Optional[str] = Form(None),
    confirm: bool = Form(False)
):
    """Clear model cache (runtime only, not disk cache)"""
    if not confirm:
        return {
            "message": "Cache clear requires confirmation. Set confirm=true to proceed.",
            "warning": "This will clear runtime model cache. Disk cache will remain."
        }

    global _model_cache

    if model:
        # Clear specific model
        keys_to_remove = [k for k in _model_cache.keys() if k.startswith(model)]
        for key in keys_to_remove:
            del _model_cache[key]

        return {
            "status": "success",
            "message": f"Cleared cache for model: {model}",
            "removed_keys": keys_to_remove
        }
    else:
        # Clear all models
        cleared_count = len(_model_cache)
        _model_cache.clear()

        return {
            "status": "success",
            "message": f"Cleared all model cache ({cleared_count} models)",
            "cleared_models": cleared_count
        }

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: Optional[str] = Form(DEFAULT_MODEL),
    language: Optional[str] = Form(None),
    temperature: Optional[float] = Form(0.0),
    response_format: Optional[str] = Form("json"),
    stream_progress: Optional[bool] = Form(False)
):
    """
    Transcribe audio file using Lightning Whisper MLX with hardware acceleration

    - **file**: Audio file to transcribe (mp3, wav, m4a, flac, ogg, mp4, mov)
    - **model**: Whisper model to use (tiny, small, medium, large, distil-medium.en, etc.)
    - **language**: Source language (auto-detected if not specified)
    - **temperature**: Sampling temperature (0.0 = deterministic)
    - **response_format**: Output format:
        - **json**: Enhanced JSON with segments and metadata (default)
        - **text**: Plain text transcription
        - **srt**: SRT subtitle format with timestamps
        - **vtt**: WebVTT subtitle format
        - **md/markdown**: Markdown with YAML frontmatter metadata + clean text
    - **stream_progress**: Stream progress updates for model loading (useful for curl)

    **Performance**: Up to 27x real-time on Apple Silicon with MLX acceleration
    **Caching**: Models are cached after first download at ~/.cache/huggingface/hub
    """

    # If streaming progress is requested, use streaming response
    if stream_progress:
        return StreamingResponse(
            transcribe_with_progress(file, model, language, temperature, response_format),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )

    # Regular non-streaming response
    start_time = time.time()

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {file_ext}. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Check file size
    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(file_content)} bytes. Max: {MAX_FILE_SIZE} bytes"
        )

    # Get whisper model
    whisper = get_whisper_model(model)

    # Transcribe
    try:
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_file.flush()

            # Prepare transcription options
            transcribe_options = {}
            if language:
                transcribe_options['language'] = language
            if temperature != 0.0:
                transcribe_options['temperature'] = temperature

            # Transcribe
            result = whisper.transcribe(tmp_file.name, **transcribe_options)

            # Clean up
            os.unlink(tmp_file.name)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    processing_time = time.time() - start_time

    # Format response based on requested format
    if response_format == "text":
        return result.get('text', '')
    elif response_format == "srt":
        segments = result.get('segments', [])
        if segments:
            return generate_srt(segments)
        else:
            # Fallback for models without segment info
            text = result.get('text', '')
            return f"1\n00:00:00,000 --> 00:00:10,000\n{text}\n"
    elif response_format == "vtt":
        segments = result.get('segments', [])
        if segments:
            return generate_vtt(segments)
        else:
            # Fallback for models without segment info
            text = result.get('text', '')
            return f"WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n{text}\n"
    elif response_format == "md" or response_format == "markdown":
        return generate_markdown(result, model, processing_time, filename=file.filename)
    else:
        # Enhanced JSON format (default)
        return {
            "text": result.get('text', ''),
            "language": result.get('language', language or 'auto'),
            "model": model,
            "processing_time": round(processing_time, 2),
            "file_duration": result.get('duration', 0),
            "real_time_factor": round(result.get('duration', 0) / processing_time, 1) if processing_time > 0 else 0,
            "segments": result.get('segments', []),
            "words": result.get('words', []) if 'words' in result else None
        }

@app.post("/transcribe/url")
async def transcribe_url(
    url: str = Form(...),
    model: Optional[str] = Form(DEFAULT_MODEL),
    language: Optional[str] = Form(None)
):
    """
    Transcribe audio from URL (YouTube, etc.)
    """
    try:
        import yt_dlp
        import requests

        # Download audio
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Try yt-dlp first (for YouTube, etc.)
            try:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': f'{tmp_dir}/%(title)s.%(ext)s',
                    'noplaylist': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    audio_file = ydl.prepare_filename(info)
            except:
                # Fallback: direct download
                response = requests.get(url, stream=True)
                audio_file = f"{tmp_dir}/audio.mp3"
                with open(audio_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Transcribe
            whisper = get_whisper_model(model)
            result = whisper.transcribe(audio_file)

            return {
                "text": result.get('text', ''),
                "language": result.get('language', language or 'auto'),
                "model": model,
                "source_url": url
            }

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="yt-dlp not available. Install with 'uv add yt-dlp'"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL transcription failed: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "whisper_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable in production
        log_level="info"
    )