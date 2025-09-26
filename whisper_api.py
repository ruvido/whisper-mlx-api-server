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
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
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
    file: UploadFile = File(...),
    model: Optional[str] = Form("medium"),
    language: Optional[str] = Form(None),
    temperature: Optional[float] = Form(0.0),
    response_format: Optional[str] = Form("json"),
    verbose: Optional[bool] = Form(False),
    stream: Optional[bool] = Form(False)
):
    """
    Transcribe audio file using MLX-Whisper standard with support for custom models

    - **file**: Audio file to transcribe (mp3, wav, m4a, flac, ogg, mp4, mov)
    - **model**: Whisper model to use or HuggingFace model path (e.g., 'bofenghuang/whisper-large-v3-distil-it-v0.2')
    - **language**: Source language (auto-detected if not specified)
    - **temperature**: Sampling temperature (0.0 = deterministic)
    - **response_format**: Output format (json, text, srt, vtt, md/markdown)
    - **stream**: Enable live progress streaming (use with `curl -N --no-buffer`)
    **Custom Models**: Supports HuggingFace models like 'bofenghuang/whisper-large-v3-distil-it-v0.2'
    **Performance**: Optimized for Apple Silicon with MLX acceleration
    """
    start_time = time.time()

    # Read file content first (needed for both streaming and normal mode)
    file_content = await file.read()

    # Se streaming è abilitato, usa un generatore per i messaggi live
    if stream:
        def transcribe_generator_mlx():
            # Validate file
            if not file.filename:
                yield "❌ Error: No file provided\n"
                return

            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in SUPPORTED_FORMATS:
                yield f"❌ Error: Unsupported format: {file_ext}\n"
                return

            yield f"📁 File: {file.filename} ({format_size(len(file_content))})\n"
            yield f"🤖 Model: {model} (MLX-Whisper)\n"
            if language:
                yield f"🗣️  Language: {language}\n"
            yield "\n"

            # Check file size
            if len(file_content) > MAX_FILE_SIZE:
                yield f"❌ Error: File too large: {format_size(len(file_content))}\n"
                return

            # MLX-Whisper loads models automatically during transcription
            yield "🔄 MLX-Whisper ready...\n"

            yield "\n🎯 Starting transcription...\n"

            # Transcribe
            try:
                # Create temp file without auto-deletion
                tmp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
                try:
                    tmp_file.write(file_content)
                    tmp_file.flush()
                    tmp_file.close()  # Close the file handle

                    # Prepare transcription options
                    transcribe_options = {}
                    if language:
                        transcribe_options['language'] = language
                    if temperature != 0.0:
                        transcribe_options['temperature'] = temperature

                    yield "🔄 Processing audio...\n"

                    # Transcribe using MLX-Whisper - EXACTLY as in official docs
                    import mlx_whisper

                    # Map model name to MLX-Whisper format
                    mlx_model = MLX_WHISPER_MODEL_MAPPING.get(model, model)
                    yield f"🔧 Using MLX model: {mlx_model}\n"

                    # OFFICIAL DOCS EXAMPLE - NO extra parameters, just the essentials
                    if language:
                        result = mlx_whisper.transcribe(
                            tmp_file.name,
                            path_or_hf_repo=mlx_model,
                            language=language
                        )
                    else:
                        result = mlx_whisper.transcribe(
                            tmp_file.name,
                            path_or_hf_repo=mlx_model
                        )

                finally:
                    # Clean up AFTER transcription is done
                    if os.path.exists(tmp_file.name):
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
            transcribe_generator_mlx(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # Normal mode (non-streaming) for MLX-Whisper
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

    # MLX-Whisper loads models automatically

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

            # Transcribe using MLX-Whisper
            import mlx_whisper

            # Map model name to MLX-Whisper format
            mlx_model = MLX_WHISPER_MODEL_MAPPING.get(model, model)
            transcribe_options['path_or_hf_repo'] = mlx_model

            # Use progress streaming but don't show progress (non-streaming mode)
            result = None
            for item in transcribe_with_progress_streaming(tmp_file.name, mlx_model, transcribe_options):
                if not isinstance(item, str):  # Result object (not progress string)
                    result = item

            # DON'T delete the temp file here - let the context manager handle it

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
        response_data = {
            "text": result.get('text', ''),
            "language": result.get('language', language or 'auto'),
            "model": model,
            "processing_time": round(processing_time, 2),
            "file_duration": result.get('duration', 0),
            "real_time_factor": round(result.get('duration', 0) / processing_time, 1) if processing_time > 0 else 0,
            "segments": result.get('segments', []),
            "words": result.get('words', []) if 'words' in result else None,
            "framework": "MLX-Whisper"
        }

        # Add status message when verbose is True
        if verbose and status_msg:
            response_data["status"] = status_msg

        return response_data

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