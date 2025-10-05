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
import logging
import contextlib
import io
import threading
import queue
import re
from typing import Optional, Dict, Any, Union
from urllib.parse import urlparse, parse_qs
import json
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy import for faster startup
_whisper_model = None
_mlx_whisper_model = None

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
    "turbo": "distil-large-v3",
    # Italian specialized models
    "turbo-it": "bofenghuang/whisper-large-v3-distil-it-v0.2",
    "italian-turbo": "bofenghuang/whisper-large-v3-distil-it-v0.2",
    "distil-it": "bofenghuang/whisper-large-v3-distil-it-v0.2"
}

# Model mapping for MLX-Whisper (uses MLX Community models on HuggingFace)
MLX_WHISPER_MODEL_MAPPING = {
    "tiny": "mlx-community/whisper-tiny",
    "tiny.en": "mlx-community/whisper-tiny.en",
    "base": "mlx-community/whisper-base",
    "base.en": "mlx-community/whisper-base.en",
    "small": "mlx-community/whisper-small",
    "small.en": "mlx-community/whisper-small.en",
    "medium": "mlx-community/whisper-medium",
    "medium.en": "mlx-community/whisper-medium.en",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v1": "mlx-community/whisper-large-v3-mlx",
    "large-v2": "mlx-community/whisper-large-v3-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-turbo",
    # Custom models (pass through as-is)
    "bofenghuang/whisper-large-v3-distil-it-v0.2": "bofenghuang/whisper-large-v3-distil-it-v0.2",
    "turbo-it": "bofenghuang/whisper-large-v3-distil-it-v0.2",
    "italian-turbo": "bofenghuang/whisper-large-v3-distil-it-v0.2",
    "distil-it": "bofenghuang/whisper-large-v3-distil-it-v0.2"
}

# Cache for loaded models
_model_cache = {}

class ProgressCapture:
    """Captures download progress from stdout/stderr and sends to queue"""
    def __init__(self, progress_queue):
        self.progress_queue = progress_queue
        self.buffer = ""

    def write(self, data):
        self.buffer += data

        # Cerca pattern di download: "file.bin: 60%", "Fetching X files:"
        patterns = [
            r'Fetching \d+ files:.*\d+%',           # Fetching 26 files: 60%
            r'[\w\-\.]+\.bin:.*\d+%',               # ctranslate2/model.bin: 60%
            r'[\w\-\.]+\.safetensors:.*\d+%',       # model.safetensors: 80%
            r'config\.json:.*\d+%',                 # config.json: 100%
            r'tokenizer\.json:.*\d+%',              # tokenizer.json: 100%
            r'[\w\-\.]+\.txt:.*\d+%'                # vocab.txt: 100%
        ]

        # Divide in linee per processare ogni linea
        lines = data.split('\n')
        for line in lines:
            for pattern in patterns:
                if re.search(pattern, line):
                    # Pulisci la linea da caratteri di controllo
                    clean_line = line.strip().replace('\r', '').replace('\x1b[?25l', '').replace('\x1b[?25h', '')
                    clean_line = re.sub(r'\x1b\[[0-9;]*[mK]', '', clean_line)  # Rimuovi codici ANSI
                    if clean_line and len(clean_line) > 10:  # Filtra linee troppo corte
                        self.progress_queue.put(f"📥 {clean_line}")
                    break

    def flush(self):
        pass

def transcribe_with_progress_streaming(audio_file, model_path, transcribe_options):
    """Transcribe with real-time download progress streaming - generator that yields progress then returns result"""
    progress_queue = queue.Queue()
    result_container = {'result': None, 'error': None}

    def run_transcription():
        try:
            # Redirect stdout/stderr per catturare download progress
            progress_capture = ProgressCapture(progress_queue)

            with contextlib.redirect_stdout(progress_capture), \
                 contextlib.redirect_stderr(progress_capture):

                import mlx_whisper
                result = mlx_whisper.transcribe(audio_file, **transcribe_options)
                result_container['result'] = result

        except Exception as e:
            result_container['error'] = e
        finally:
            progress_queue.put("TRANSCRIPTION_DONE")

    # Avvia trascrizione in thread separato
    thread = threading.Thread(target=run_transcription)
    thread.start()

    # Yielda messaggi di progresso in tempo reale
    while thread.is_alive():
        try:
            message = progress_queue.get(timeout=0.1)
            if message == "TRANSCRIPTION_DONE":
                break
            yield f"{message}\n"
        except queue.Empty:
            continue

    # Ensure thread is completely finished
    thread.join()

    # Get any remaining messages
    while not progress_queue.empty():
        try:
            message = progress_queue.get_nowait()
            if message != "TRANSCRIPTION_DONE":
                yield f"{message}\n"
        except queue.Empty:
            break

    # Controlla risultato
    if result_container['error']:
        raise result_container['error']

    # Yield the result as the last item
    yield result_container['result']

def format_size(bytes_size):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}TB"


def get_whisper_model(model: str = DEFAULT_MODEL, batch_size: int = DEFAULT_BATCH_SIZE, verbose=False):
    """Lazy load the whisper model with caching"""
    global _model_cache

    # Map model name
    mapped_model = MODEL_MAPPING.get(model, model)
    cache_key = f"{mapped_model}_{batch_size}"

    status_msg = ""
    if cache_key not in _model_cache:
        try:
            from lightning_whisper_mlx import LightningWhisperMLX
            import glob

            # Set cache directory to ensure persistence
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            os.makedirs(cache_dir, exist_ok=True)

            # Check if model is already cached
            model_pattern = f"*{mapped_model}*"
            existing_files = []
            for pattern_path in glob.glob(os.path.join(cache_dir, "**", model_pattern), recursive=True):
                if os.path.isfile(pattern_path):
                    existing_files.append(pattern_path)

            if existing_files:
                status_msg = f"📦 Model '{mapped_model}' already cached, loading..."
                logger.info(status_msg)
            else:
                status_msg = f"🔄 Downloading model '{mapped_model}' (first time)..."
                logger.info(status_msg)

            _model_cache[cache_key] = LightningWhisperMLX(
                model=mapped_model,
                batch_size=batch_size,
                quant=None
            )

            success_msg = f"✅ Model '{mapped_model}' loaded successfully!"
            logger.info(success_msg)
            if verbose:
                status_msg += f"\n{success_msg}"

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
        status_msg = f"⚡ Model '{mapped_model}' already loaded in memory"

    return _model_cache[cache_key], status_msg if verbose else ""



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


# =============================================================================
# Media Type Detection and Download Functions
# =============================================================================

def detect_media_type(media: str) -> str:
    """
    Detect media type from input string
    Returns: 'local_file', 'youtube_video', 'youtube_playlist', 'remote_audio'
    """
    # Check if it's a local file path
    if os.path.exists(media) or not media.startswith(('http://', 'https://')):
        return 'local_file'

    # Parse URL
    parsed = urlparse(media)

    # Check for YouTube URLs
    if parsed.netloc in ['youtube.com', 'www.youtube.com', 'youtu.be', 'www.youtu.be']:
        # Check if it's a playlist
        if 'list=' in media:
            return 'youtube_playlist'
        else:
            return 'youtube_video'

    # Check for common audio file extensions in URL
    path = parsed.path.lower()
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4', '.mov', '.webm']
    if any(path.endswith(ext) for ext in audio_extensions):
        return 'remote_audio'

    # Default to remote audio for other HTTP URLs
    return 'remote_audio'


def safe_filename(name: str) -> str:
    """Convert string to safe filename by removing/replacing invalid characters"""
    # Remove or replace invalid characters
    name = re.sub(r'[<>:"/\\|?*]', '-', name)
    name = re.sub(r'[^\w\s\-_.,()[\]]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    # Limit length
    if len(name) > 100:
        name = name[:100]
    return name


def download_youtube_audio(url: str, output_dir: str = ".") -> Dict[str, Any]:
    """
    Download audio from YouTube URL or playlist
    Returns dict with downloaded file info
    """
    try:
        import yt_dlp

        # Detect if it's a playlist
        media_type = detect_media_type(url)

        if media_type == 'youtube_playlist':
            # Get playlist info first
            with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True}) as ydl:
                playlist_info = ydl.extract_info(url, download=False)
                playlist_title = safe_filename(playlist_info.get('title', 'playlist'))
                playlist_dir = os.path.join(output_dir, playlist_title)
                os.makedirs(playlist_dir, exist_ok=True)

            # Download all videos in playlist
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(playlist_dir, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
                'extractaudio': True,
                'audioformat': 'mp3',
                'embed_subs': False,
                'writesubtitles': False,
                # Modern browser headers to avoid detection
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Accept-Encoding': 'gzip,deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                },
                # Increased retry counts for playlists
                'extractor_retries': 5,
                'fragment_retries': 5,
                'retry_sleep_functions': {
                    'fragment': lambda n: min(15, 3 * n),
                    'extractor': lambda n: min(30, 2 * n),
                },
                'ignoreerrors': False,
                'abort_on_error': True
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url)

            return {
                'type': 'playlist',
                'title': playlist_title,
                'directory': playlist_dir,
                'count': len(info.get('entries', [])),
                'files': [os.path.join(playlist_dir, f"{safe_filename(entry.get('title', 'video'))}.mp3")
                         for entry in info.get('entries', []) if entry]
            }

        else:
            # Single video download - improved configuration for reliability
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': False,  # Enable output for debugging
                'no_warnings': False,
                'extractaudio': True,
                'audioformat': 'mp3',
                'embed_subs': False,
                'writesubtitles': False,
                'cookiefile': None,
                # Modern browser headers to avoid detection
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Accept-Encoding': 'gzip,deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                },
                # Increased retry counts and sleep functions
                'extractor_retries': 5,
                'fragment_retries': 5,
                'retry_sleep_functions': {
                    'fragment': lambda n: min(15, 3 * n),
                    'extractor': lambda n: min(30, 2 * n),
                },
                # Additional reliability options
                'ignoreerrors': False,
                'abort_on_error': True,
                # Anti-detection measures
                'sleep_interval': 1,
                'max_sleep_interval': 5,
                'sleep_interval_requests': 1,
                'skip_unavailable_fragments': True
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url)
                    title = safe_filename(info.get('title', 'video'))
                    filename = os.path.join(output_dir, f"{title}.mp3")

                return {
                    'type': 'single',
                    'title': title,
                    'filename': filename,
                    'duration': info.get('duration', 0)
                }
            except Exception as first_error:
                # Fallback: try with minimal config similar to command line
                logger.warning(f"First attempt failed: {first_error}. Trying fallback configuration...")

                fallback_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                    }],
                    'extractaudio': True,
                    'quiet': False,
                    # Add additional anti-detection measures for fallback
                    'sleep_interval': 1,
                    'max_sleep_interval': 5,
                    'sleep_interval_requests': 1,
                    'extractor_retries': 10,
                    'fragment_retries': 10,
                    'skip_unavailable_fragments': True,
                }

                try:
                    with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                        info = ydl.extract_info(url)
                        title = safe_filename(info.get('title', 'video'))
                        filename = os.path.join(output_dir, f"{title}.mp3")

                    return {
                        'type': 'single',
                        'title': title,
                        'filename': filename,
                        'duration': info.get('duration', 0)
                    }
                except Exception as second_error:
                    # Final fallback: exact command line equivalent
                    logger.warning(f"Second attempt failed: {second_error}. Trying exact command line equivalent...")

                    cmdline_opts = {
                        'format': 'bestaudio/best',
                        'outtmpl': os.path.join(output_dir, 'test_audio.%(ext)s'),
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                        }],
                        'extractaudio': True,
                    }

                    with yt_dlp.YoutubeDL(cmdline_opts) as ydl:
                        info = ydl.extract_info(url)
                        title = safe_filename(info.get('title', 'video'))
                        # Use the actual downloaded filename
                        filename = os.path.join(output_dir, "test_audio.mp3")

                    return {
                        'type': 'single',
                        'title': title,
                        'filename': filename,
                        'duration': info.get('duration', 0)
                    }

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="yt-dlp not available. Install with: uv add yt-dlp"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"YouTube download failed: {str(e)}"
        )


def download_remote_audio(url: str, output_dir: str = ".") -> str:
    """Download remote audio file and return local path"""
    try:
        import requests

        # Get filename from URL
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)

        if not filename or '.' not in filename:
            filename = "remote_audio.mp3"

        local_path = os.path.join(output_dir, filename)

        # Download file
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return local_path

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="requests library not available. Install with: uv add requests"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Remote audio download failed: {str(e)}"
        )


# =============================================================================
# API Endpoints
# =============================================================================

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
    available_mlx_models = list(MLX_WHISPER_MODEL_MAPPING.keys())
    loaded_models = list(_model_cache.keys())

    return {
        "lightning_whisper_mlx": {
            "available_models": available_models,
            "default_model": DEFAULT_MODEL,
            "recommended": "distil-medium.en (fast and accurate)",
            "endpoint": "/transcribe"
        },
        "mlx_whisper": {
            "available_models": available_mlx_models,
            "default_model": "medium",
            "recommended": "tiny (fastest), medium (balanced), turbo (best quality)",
            "endpoint": "/transcribe/mlx"
        },
        "loaded_models": loaded_models,
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
    verbose: Optional[bool] = Form(False),
    stream: Optional[bool] = Form(False)
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
    - **stream**: Enable live progress streaming (use with `curl -N --no-buffer`)
    **Performance**: Up to 27x real-time on Apple Silicon with MLX acceleration
    **Caching**: Models are cached after first download at ~/.cache/huggingface/hub
    """
    start_time = time.time()

    # Read file content first (needed for both streaming and normal mode)
    file_content = await file.read()

    # Se streaming è abilitato, usa un generatore per i messaggi live
    if stream:
        def transcribe_generator():
            # Validate file
            if not file.filename:
                yield "❌ Error: No file provided\n"
                return

            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in SUPPORTED_FORMATS:
                yield f"❌ Error: Unsupported format: {file_ext}\n"
                return

            yield f"📁 File: {file.filename} ({format_size(len(file_content))})\n"
            yield f"🤖 Model: {model}\n"
            if language:
                yield f"🗣️  Language: {language}\n"
            yield "\n"

            # Check file size
            file_content_local = file_content
            if len(file_content_local) > MAX_FILE_SIZE:
                yield f"❌ Error: File too large: {format_size(len(file_content_local))}\n"
                return

            # Get whisper model
            yield "🔄 Checking model cache...\n"
            try:
                whisper, status_msg = get_whisper_model(model, verbose=True)
                if "already cached" in status_msg:
                    yield "📦 Model already cached, loading...\n"
                elif "Downloading" in status_msg:
                    yield "🔄 Downloading model (first time)...\n"
                yield "✅ Model loaded successfully!\n"
            except Exception as e:
                yield f"❌ Error loading model: {e}\n"
                return

            yield "\n🎯 Starting transcription...\n"

            # Transcribe
            try:
                with tempfile.NamedTemporaryFile(suffix=file_ext, delete=True) as tmp_file:
                    tmp_file.write(file_content_local)
                    tmp_file.flush()

                    # Prepare transcription options
                    transcribe_options = {}
                    if language:
                        transcribe_options['language'] = language
                    if temperature != 0.0:
                        transcribe_options['temperature'] = temperature

                    yield "🔄 Processing audio...\n"

                    # Transcribe
                    result = whisper.transcribe(tmp_file.name, **transcribe_options)

                    # Clean up
                    os.unlink(tmp_file.name)

                processing_time = time.time() - start_time
                yield "✅ Transcription completed!\n"
                yield f"⏱️  Processing time: {processing_time:.2f}s\n"

                if 'duration' in result and result['duration'] > 0:
                    rtf = result['duration'] / processing_time
                    yield f"⚡ Real-time factor: {rtf:.1f}x\n"

                yield f"🗣️  Detected language: {result.get('language', 'unknown')}\n"
                yield "\n" + "=" * 48 + "\n"
                yield "📝 TRANSCRIPTION RESULT:\n"
                yield "=" * 48 + "\n\n"

                # Format response based on requested format
                if response_format == "text":
                    yield result.get('text', '')
                elif response_format == "md" or response_format == "markdown":
                    yield generate_markdown(result, model, processing_time, filename=file.filename)
                else:
                    # JSON format come testo leggibile
                    yield f"Text: {result.get('text', '')}\n"
                    yield f"Language: {result.get('language', 'auto')}\n"
                    yield f"Model: {model}\n"

            except Exception as e:
                yield f"❌ Transcription failed: {e}\n"

        return StreamingResponse(
            transcribe_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

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
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(file_content)} bytes. Max: {MAX_FILE_SIZE} bytes"
        )

    # Get whisper model
    status_msg = ""
    if verbose:
        whisper, status_msg = get_whisper_model(model, verbose=True)
    else:
        whisper, _ = get_whisper_model(model, verbose=False)

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
        return PlainTextResponse(result.get('text', ''))
    elif response_format == "srt":
        segments = result.get('segments', [])
        if segments:
            return PlainTextResponse(generate_srt(segments))
        else:
            # Fallback for models without segment info
            text = result.get('text', '')
            return PlainTextResponse(f"1\n00:00:00,000 --> 00:00:10,000\n{text}\n")
    elif response_format == "vtt":
        segments = result.get('segments', [])
        if segments:
            return PlainTextResponse(generate_vtt(segments))
        else:
            # Fallback for models without segment info
            text = result.get('text', '')
            return PlainTextResponse(f"WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n{text}\n")
    elif response_format == "md" or response_format == "markdown":
        return PlainTextResponse(generate_markdown(result, model, processing_time, filename=file.filename))
    else:
        # Enhanced JSON format (default)
        response_data = {
            "text": result.get('text', ''),
            "language": result.get('language', language or 'auto'),
            "model": model,
            "processing_time": round(processing_time, 2),
            "file_duration": result.get('duration', 0),
            "real_time_factor": round(result.get('duration', 0) / processing_time, 1) if processing_time > 0 else 0,
            "segments": result.get('segments', []),
            "words": result.get('words', []) if 'words' in result else None
        }

        # Add status message when verbose is True
        if verbose and status_msg:
            response_data["status"] = status_msg

        return response_data

@app.post("/transcribe/mlx")
async def transcribe_audio_mlx(
    media: Optional[str] = Form(None),
    file: Optional[UploadFile] = None,
    model: Optional[str] = Form("medium"),
    language: Optional[str] = Form(None),
    temperature: Optional[float] = Form(0.0),
    response_format: Optional[str] = Form("json"),
    verbose: Optional[bool] = Form(False),
    stream: Optional[bool] = Form(False)
):
    """
    Transcribe audio using MLX-Whisper with automatic media detection

    - **media**: Universal media input - supports:
        * Local file paths (e.g., "audio.mp3")
        * YouTube videos (e.g., "https://youtu.be/KygDdSZbGZk")
        * YouTube playlists (e.g., "https://youtube.com/playlist?list=...")
        * Remote audio URLs (e.g., "https://server.com/audio.mp3")
    - **file**: Deprecated - use media parameter instead
    - **model**: Whisper model to use (tiny, small, medium, large, turbo) or HuggingFace model path
    - **language**: Source language (auto-detected if not specified)
    - **temperature**: Sampling temperature (0.0 = deterministic)
    - **response_format**: Output format (json, text, srt, vtt, md/markdown)
    - **stream**: Enable live progress streaming (use with `curl -N --no-buffer`)

    **Examples:**
    ```bash
    # Local file
    curl -X POST "http://localhost:8000/transcribe/mlx" -F "media=audio.mp3" -F "model=turbo"

    # YouTube video
    curl -X POST "http://localhost:8000/transcribe/mlx" -F "media=https://youtu.be/KygDdSZbGZk" -F "model=medium"

    # Remote audio
    curl -X POST "http://localhost:8000/transcribe/mlx" -F "media=https://server.com/audio.mp3" -F "model=small"
    ```
    """
    start_time = time.time()

    # Handle backward compatibility: if file is provided but media is not, use file
    if media is None and file is not None:
        media_input = file
        media_type = 'local_file'
        media_name = file.filename
    elif isinstance(media, str):
        # String input: could be URL or local file path
        media_type = detect_media_type(media)
        media_input = media
        media_name = media
    elif hasattr(media, 'filename'):
        # UploadFile passed as media
        media_input = media
        media_type = 'local_file'
        media_name = media.filename
    else:
        raise HTTPException(status_code=400, detail="No media input provided. Use 'media' parameter for files, URLs, or YouTube links.")

    # Read file content first if it's an UploadFile (needed for both streaming and normal mode)
    file_content = None
    file_ext = None
    if media_type == 'local_file' and hasattr(media_input, 'read'):
        file_content = await media_input.read()
        file_ext = Path(media_input.filename).suffix.lower() if media_input.filename else '.mp3'

    if stream:
        def transcribe_generator_mlx():
            temp_files_to_cleanup = []
            temp_dirs_to_cleanup = []

            try:
                yield f"🎯 Media Type: {media_type}\n"
                yield f"📁 Input: {media_name}\n"
                yield f"🤖 Model: {model} (MLX-Whisper)\n"
                if language:
                    yield f"🗣️  Language: {language}\n"
                yield "\n"

                # Step 1: Process different media types and get local file path(s)
                audio_files = []

                if media_type == 'local_file':
                    # Handle uploaded file
                    if hasattr(media_input, 'read'):
                        # It's an UploadFile - file_content was already read above

                        # Validate file format
                        if file_ext not in SUPPORTED_FORMATS:
                            yield f"❌ Error: Unsupported format: {file_ext}\n"
                            return

                        # Check file size
                        if len(file_content) > MAX_FILE_SIZE:
                            yield f"❌ Error: File too large: {format_size(len(file_content))}\n"
                            return

                        # Create temporary file
                        tmp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
                        temp_files_to_cleanup.append(tmp_file.name)
                        tmp_file.write(file_content)
                        tmp_file.flush()
                        tmp_file.close()

                        audio_files.append(tmp_file.name)
                        yield f"📁 File uploaded: {media_input.filename} ({format_size(len(file_content))})\n"

                    else:
                        # It's a local file path
                        if not os.path.exists(media_input):
                            yield f"❌ Error: File not found: {media_input}\n"
                            return

                        # Check file size
                        file_size = os.path.getsize(media_input)
                        if file_size > MAX_FILE_SIZE:
                            yield f"❌ Error: File too large: {format_size(file_size)}\n"
                            return

                        audio_files.append(media_input)
                        yield f"📁 Local file: {media_input} ({format_size(file_size)})\n"

                elif media_type == 'youtube_video':
                    yield "🔄 Downloading YouTube video...\n"
                    try:
                        # Create temp directory for downloads
                        temp_dir = tempfile.mkdtemp()
                        temp_dirs_to_cleanup.append(temp_dir)

                        download_result = download_youtube_audio(media_input, temp_dir)
                        if download_result['type'] == 'single':
                            audio_files.append(download_result['filename'])
                            temp_files_to_cleanup.append(download_result['filename'])
                            yield f"✅ Downloaded: {download_result['title']}\n"
                        else:
                            yield f"❌ Error: Expected single video but got {download_result['type']}\n"
                            return
                    except Exception as e:
                        yield f"❌ YouTube download failed: {str(e)}\n"
                        return

                elif media_type == 'youtube_playlist':
                    yield "🔄 Downloading YouTube playlist...\n"
                    try:
                        # Create temp directory for downloads
                        temp_dir = tempfile.mkdtemp()
                        temp_dirs_to_cleanup.append(temp_dir)

                        download_result = download_youtube_audio(media_input, temp_dir)
                        if download_result['type'] == 'playlist':
                            audio_files.extend(download_result['files'])
                            temp_files_to_cleanup.extend(download_result['files'])
                            yield f"✅ Downloaded {download_result['count']} videos from playlist: {download_result['title']}\n"
                        else:
                            yield f"❌ Error: Expected playlist but got {download_result['type']}\n"
                            return
                    except Exception as e:
                        yield f"❌ Playlist download failed: {str(e)}\n"
                        return

                elif media_type == 'remote_audio':
                    yield "🔄 Downloading remote audio...\n"
                    try:
                        # Create temp directory for downloads
                        temp_dir = tempfile.mkdtemp()
                        temp_dirs_to_cleanup.append(temp_dir)

                        downloaded_file = download_remote_audio(media_input, temp_dir)
                        audio_files.append(downloaded_file)
                        temp_files_to_cleanup.append(downloaded_file)

                        file_size = os.path.getsize(downloaded_file)
                        yield f"✅ Downloaded: {os.path.basename(downloaded_file)} ({format_size(file_size)})\n"

                        # Check file size after download
                        if file_size > MAX_FILE_SIZE:
                            yield f"❌ Error: Downloaded file too large: {format_size(file_size)}\n"
                            return

                    except Exception as e:
                        yield f"❌ Remote audio download failed: {str(e)}\n"
                        return

                else:
                    yield f"❌ Error: Unsupported media type: {media_type}\n"
                    return

                # Step 2: Transcribe all audio files
                yield f"\n🎯 Starting transcription of {len(audio_files)} file(s)...\n"

                # Import MLX-Whisper
                import mlx_whisper

                # Map model name to MLX-Whisper format
                mlx_model = MLX_WHISPER_MODEL_MAPPING.get(model, model)
                yield f"🔧 Using MLX model: {mlx_model}\n\n"

                all_results = []

                for i, audio_file in enumerate(audio_files, 1):
                    if len(audio_files) > 1:
                        filename = os.path.basename(audio_file)
                        yield f"📝 Processing file {i}/{len(audio_files)}: {filename}\n"

                    try:
                        # Prepare transcription options
                        transcribe_options = {
                            'path_or_hf_repo': mlx_model
                        }
                        if language:
                            transcribe_options['language'] = language
                        if temperature != 0.0:
                            transcribe_options['temperature'] = temperature

                        # Add word_timestamps for SRT/VTT support
                        if response_format in ["srt", "vtt"]:
                            transcribe_options['word_timestamps'] = True

                        yield "🔄 Processing audio...\n"

                        # Transcribe using MLX-Whisper
                        result = mlx_whisper.transcribe(audio_file, **transcribe_options)

                        # Add file info to result
                        result['source_file'] = os.path.basename(audio_file)
                        all_results.append(result)

                        if len(audio_files) > 1:
                            yield f"✅ Completed file {i}/{len(audio_files)}\n"

                    except Exception as e:
                        yield f"❌ Transcription failed for {os.path.basename(audio_file)}: {str(e)}\n"
                        continue

                if not all_results:
                    yield "❌ No files were successfully transcribed\n"
                    return

                processing_time = time.time() - start_time
                yield "✅ All transcriptions completed!\n"
                yield f"⏱️  Total processing time: {processing_time:.2f}s\n"

                # Calculate average real-time factor
                total_duration = sum(r.get('duration', 0) for r in all_results)
                if total_duration > 0:
                    rtf = total_duration / processing_time
                    yield f"⚡ Real-time factor: {rtf:.1f}x\n"

                # Show detected languages
                languages = set(r.get('language', 'unknown') for r in all_results)
                yield f"🗣️  Detected language(s): {', '.join(languages)}\n"

                yield "\n" + "=" * 48 + "\n"
                yield "📝 TRANSCRIPTION RESULT(S):\n"
                yield "=" * 48 + "\n\n"

                # Format response based on requested format
                if len(all_results) == 1:
                    # Single result
                    result = all_results[0]
                    if response_format == "text":
                        yield result.get('text', '')
                    elif response_format == "md" or response_format == "markdown":
                        yield generate_markdown(result, model, processing_time, filename=result.get('source_file', media_name))
                    else:
                        # JSON format as readable text
                        yield f"Text: {result.get('text', '')}\n"
                        yield f"Language: {result.get('language', 'auto')}\n"
                        yield f"Model: {model}\n"
                        if result.get('source_file'):
                            yield f"Source: {result['source_file']}\n"
                else:
                    # Multiple results
                    for i, result in enumerate(all_results, 1):
                        yield f"\n--- File {i}: {result.get('source_file', f'file_{i}')} ---\n"
                        if response_format == "text":
                            yield result.get('text', '')
                        elif response_format == "md" or response_format == "markdown":
                            yield generate_markdown(result, model, processing_time, filename=result.get('source_file', f'file_{i}'))
                        else:
                            # JSON format as readable text
                            yield f"Text: {result.get('text', '')}\n"
                            yield f"Language: {result.get('language', 'auto')}\n"
                        yield "\n"

            except Exception as e:
                yield f"❌ Transcription failed: {str(e)}\n"

            finally:
                # Clean up temporary files and directories
                for temp_file in temp_files_to_cleanup:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                    except Exception:
                        pass  # Ignore cleanup errors

                for temp_dir in temp_dirs_to_cleanup:
                    try:
                        if os.path.exists(temp_dir):
                            import shutil
                            shutil.rmtree(temp_dir)
                    except Exception:
                        pass  # Ignore cleanup errors

        return StreamingResponse(
            transcribe_generator_mlx(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # Non-streaming mode - process media and transcribe
    try:
        # Process different media types
        local_audio_path = None
        downloaded_files = []
        cleanup_files = []

        if media_type == 'local_file':
            if hasattr(media_input, 'read'):
                # UploadFile - file_content was already read above
                if len(file_content) > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large: {format_size(len(file_content))}"
                    )

                if file_ext not in SUPPORTED_FORMATS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported format: {file_ext}. Supported: {', '.join(SUPPORTED_FORMATS)}"
                    )

                # Create temp file
                tmp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
                tmp_file.write(file_content)
                tmp_file.flush()
                tmp_file.close()
                local_audio_path = tmp_file.name
                cleanup_files.append(local_audio_path)

            else:
                # Local file path
                if not os.path.exists(media_input):
                    raise HTTPException(status_code=404, detail=f"File not found: {media_input}")
                local_audio_path = media_input

        elif media_type in ['youtube_video', 'youtube_playlist']:
            download_info = download_youtube_audio(media_input, ".")

            if download_info['type'] == 'playlist':
                local_audio_path = download_info['files'][0]
                downloaded_files.extend(download_info['files'])
            else:
                local_audio_path = download_info['filename']
                downloaded_files.append(local_audio_path)

        elif media_type == 'remote_audio':
            local_audio_path = download_remote_audio(media_input, ".")
            downloaded_files.append(local_audio_path)

        if not local_audio_path or not os.path.exists(local_audio_path):
            raise HTTPException(status_code=500, detail="Could not prepare audio file for transcription")

        # Transcribe using MLX-Whisper
        import mlx_whisper

        mlx_model = MLX_WHISPER_MODEL_MAPPING.get(model, model)

        # Prepare transcription options
        transcribe_options = {
            'path_or_hf_repo': mlx_model
        }
        if language:
            transcribe_options['language'] = language
        if temperature != 0.0:
            transcribe_options['temperature'] = temperature

        # Add word_timestamps for SRT/VTT support
        if response_format in ["srt", "vtt"]:
            transcribe_options['word_timestamps'] = True

        # Transcribe
        result = mlx_whisper.transcribe(local_audio_path, **transcribe_options)

        processing_time = time.time() - start_time

        # Cleanup temporary files
        for cleanup_file in cleanup_files:
            if os.path.exists(cleanup_file):
                os.unlink(cleanup_file)

        # Format response based on requested format
        if response_format == "text":
            return PlainTextResponse(result.get('text', ''))
        elif response_format == "srt":
            segments = result.get('segments', [])
            if segments:
                return PlainTextResponse(generate_srt(segments))
            else:
                text = result.get('text', '')
                return PlainTextResponse(f"1\n00:00:00,000 --> 00:00:10,000\n{text}\n")
        elif response_format == "vtt":
            segments = result.get('segments', [])
            if segments:
                return PlainTextResponse(generate_vtt(segments))
            else:
                text = result.get('text', '')
                return PlainTextResponse(f"WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n{text}\n")
        elif response_format == "md" or response_format == "markdown":
            return PlainTextResponse(generate_markdown(result, model, processing_time, filename=media_name))
        else:
            # Enhanced JSON format (default)
            response_data = {
                "text": result.get('text', ''),
                "language": result.get('language', language or 'auto'),
                "model": model,
                "processing_time": round(processing_time, 2),
                "file_duration": result.get('duration', 0),
                "real_time_factor": round(result.get('duration', 0) / processing_time, 1) if processing_time > 0 else 0,
                "segments": result.get('segments', []),
                "words": result.get('words', []) if 'words' in result else None,
                "framework": "MLX-Whisper",
                "media_type": media_type,
                "media_source": media_name
            }

            if downloaded_files:
                response_data["downloaded_files"] = downloaded_files

            return response_data

    except Exception as e:
        # Cleanup on error
        for cleanup_file in cleanup_files:
            if os.path.exists(cleanup_file):
                os.unlink(cleanup_file)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

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