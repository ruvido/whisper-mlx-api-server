#!/usr/bin/env python3
"""
MLX Whisper API Server
Fast speech-to-text API optimized for Apple Silicon
"""

import contextlib
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

# =============================================================================
# Configuration and Constants
# =============================================================================

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_MODEL = "medium"
SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".mov"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
VALID_RESPONSE_FORMATS = {"json", "text", "srt", "vtt", "md", "markdown"}
HTTP_CHUNK_SIZE = 8192
HTTP_TIMEOUT_SECONDS = 30
MIN_PROGRESS_LINE_LENGTH = 10
MAX_FILENAME_LENGTH = 100
DEFAULT_SEGMENT_DURATION = 10.0

# Model cache for performance
_model_cache: Dict[str, Any] = {}

# Model mapping for MLX-Whisper (uses MLX Community models on HuggingFace)
MODEL_MAPPING = {
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
    # Italian specialized models
    "bofenghuang/whisper-large-v3-distil-it-v0.2": "bofenghuang/whisper-large-v3-distil-it-v0.2",
    "turbo-it": "bofenghuang/whisper-large-v3-distil-it-v0.2",
    "italian-turbo": "bofenghuang/whisper-large-v3-distil-it-v0.2",
    "distil-it": "bofenghuang/whisper-large-v3-distil-it-v0.2"
}

# =============================================================================
# Response Models
# =============================================================================

class TranscriptionResponse(BaseModel):
    """Standard transcription response model"""
    text: str
    language: str
    model: str
    processing_time: float
    file_duration: float
    real_time_factor: float
    segments: List[Dict[str, Any]]
    words: Optional[List[Dict[str, Any]]] = None
    framework: str = "MLX-Whisper"
    media_type: Optional[str] = None
    media_source: Optional[str] = None
    downloaded_files: Optional[List[str]] = None
    status: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    message: Optional[str] = None
    version: Optional[str] = None
    hardware: Optional[str] = None
    mlx_available: Optional[bool] = None
    mlx_device: Optional[str] = None
    models_loaded: Optional[bool] = None
    error: Optional[str] = None

class ModelsResponse(BaseModel):
    """Models listing response model"""
    available_models: List[str]
    default_model: str
    recommended: str
    endpoint: str
    cache_location: str

# =============================================================================
# Media Processing Services
# =============================================================================

class MediaProcessor:
    """Handles different types of media input and processing"""

    @staticmethod
    def detect_media_type(media: str) -> str:
        """Detect media type from input string"""
        if os.path.exists(media) or not media.startswith(('http://', 'https://')):
            return 'local_file'

        parsed = urlparse(media)
        youtube_domains = ['youtube.com', 'www.youtube.com', 'youtu.be', 'www.youtu.be']

        if parsed.netloc in youtube_domains:
            return 'youtube_playlist' if 'list=' in media else 'youtube_video'

        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4', '.mov', '.webm']
        if any(parsed.path.lower().endswith(ext) for ext in audio_extensions):
            return 'remote_audio'

        return 'remote_audio'

    @staticmethod
    def process_uploaded_file(file: UploadFile, file_content: bytes) -> str:
        """Process uploaded file and return temporary file path"""
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {file_ext}. Supported: {', '.join(SUPPORTED_FORMATS)}"
            )

        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {len(file_content)} bytes. Max: {MAX_FILE_SIZE} bytes"
            )

        # Create temporary file using context manager
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_file.flush()
            return tmp_file.name

    @staticmethod
    def download_youtube_audio(url: str, output_dir: str = ".") -> Dict[str, Any]:
        """Download audio from YouTube URL"""
        try:
            import yt_dlp
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="yt-dlp not available. Install with: uv add yt-dlp"
            )

        logger.info(f"Downloading YouTube audio from: {url}")
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'extractaudio': True,
                'audioformat': 'mp3',
                'quiet': True,
                'no_warnings': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url)
                title = MediaProcessor._safe_filename(info.get('title', 'video'))
                filename = os.path.join(output_dir, f"{title}.mp3")

            logger.info(f"YouTube download completed: {title}")
            return {
                'type': 'single',
                'title': title,
                'filename': filename,
                'duration': info.get('duration', 0)
            }

        except Exception as e:
            logger.error(f"YouTube download failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"YouTube download failed: {str(e)}"
            )

    @staticmethod
    def download_remote_audio(url: str, output_dir: str = ".") -> str:
        """Download remote audio file and return local path"""
        try:
            import requests
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="requests library not available. Install with: uv add requests"
            )

        logger.info(f"Downloading remote audio from: {url}")
        try:
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)

            if not filename or '.' not in filename:
                filename = "remote_audio.mp3"

            local_path = os.path.join(output_dir, filename)

            response = requests.get(url, stream=True, timeout=HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=HTTP_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Remote audio download completed: {filename}")
            return local_path

        except Exception as e:
            logger.error(f"Remote audio download failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Remote audio download failed: {str(e)}"
            )

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Convert string to safe filename"""
        name = re.sub(r'[<>:"/\\|?*]', '-', name)
        name = re.sub(r'[^\w\s\-_.,()[\]]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name[:MAX_FILENAME_LENGTH] if len(name) > MAX_FILENAME_LENGTH else name

# =============================================================================
# Transcription Services
# =============================================================================

class WhisperService:
    """Handles MLX-Whisper transcription"""

    @staticmethod
    def prepare_options(model: str, language: Optional[str], temperature: float, response_format: str) -> Dict[str, Any]:
        """Prepare MLX-Whisper transcription options

        Args:
            model: Model identifier from MODEL_MAPPING
            language: Optional ISO language code (e.g., 'en', 'it')
            temperature: Sampling temperature (0.0-1.0)
            response_format: Output format (json, text, srt, vtt, md)

        Returns:
            Dictionary of transcription options for mlx_whisper
        """
        # Validate temperature
        if not (0.0 <= temperature <= 1.0):
            raise HTTPException(
                status_code=400,
                detail=f"Temperature must be between 0.0 and 1.0, got {temperature}"
            )

        model_path = MODEL_MAPPING.get(model, model)
        options = {'path_or_hf_repo': model_path}

        if language:
            options['language'] = language
        if temperature != 0.0:
            options['temperature'] = temperature
        if response_format in ["srt", "vtt"]:
            options['word_timestamps'] = True

        return options

    @staticmethod
    def transcribe_file(audio_file: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe a single audio file using MLX-Whisper with model caching

        Args:
            audio_file: Path to audio file
            options: Transcription options including model path

        Returns:
            Dictionary with transcription results
        """
        try:
            import mlx_whisper

            # Model caching for 10-100x performance improvement
            model_path = options.get('path_or_hf_repo', 'mlx-community/whisper-medium')
            if model_path not in _model_cache:
                logger.info(f"Loading model into cache: {model_path}")
                _model_cache[model_path] = model_path  # mlx_whisper handles its own caching

            result = mlx_whisper.transcribe(audio_file, **options)
            result['source_file'] = os.path.basename(audio_file)
            return result
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="mlx-whisper not available. Install with: uv add mlx-whisper"
            )

# =============================================================================
# Response Formatting Services
# =============================================================================

class ResponseFormatter:
    """Handles response formatting for different output types"""

    @staticmethod
    def _format_time(seconds: float, separator: str = '.') -> str:
        """Format seconds to timestamp (HH:MM:SS.mmm or HH:MM:SS,mmm)

        Args:
            seconds: Time in seconds
            separator: Decimal separator ('.' for VTT, ',' for SRT)

        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        time_str = f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
        return time_str if separator == '.' else time_str.replace('.', separator)

    @staticmethod
    def format_time_srt(seconds: float) -> str:
        """Format seconds to SRT time format (with comma)"""
        return ResponseFormatter._format_time(seconds, ',')

    @staticmethod
    def format_time_vtt(seconds: float) -> str:
        """Format seconds to VTT time format (with dot)"""
        return ResponseFormatter._format_time(seconds, '.')

    @staticmethod
    def _extract_segment_data(segment: Union[Dict[str, Any], List], time_formatter) -> tuple:
        """Extract start time, end time, and text from segment data"""
        if isinstance(segment, dict):
            start_time = time_formatter(segment.get('start', 0))
            end_time = time_formatter(segment.get('end', segment.get('start', 0) + 3))
            text = segment.get('text', '').strip()
        elif isinstance(segment, list) and len(segment) >= 3:
            start_time = time_formatter(segment[0] / 1000.0)
            end_time = time_formatter(segment[1] / 1000.0)
            text = segment[2].strip()
        else:
            return "", "", ""

        return start_time, end_time, text

    @staticmethod
    def generate_srt(segments: List[Union[Dict[str, Any], List]]) -> str:
        """Generate SRT subtitle format"""
        srt_content = []
        for i, segment in enumerate(segments, 1):
            start_time, end_time, text = ResponseFormatter._extract_segment_data(
                segment, ResponseFormatter.format_time_srt
            )
            if text:
                srt_content.extend([str(i), f"{start_time} --> {end_time}", text, ""])
        return "\n".join(srt_content)

    @staticmethod
    def generate_vtt(segments: List[Union[Dict[str, Any], List]]) -> str:
        """Generate VTT subtitle format"""
        vtt_content = ["WEBVTT", ""]

        for segment in segments:
            start_time, end_time, text = ResponseFormatter._extract_segment_data(
                segment, ResponseFormatter.format_time_vtt
            )
            if text:
                vtt_content.extend([f"{start_time} --> {end_time}", text, ""])
        return "\n".join(vtt_content)

    @staticmethod
    def generate_markdown(result: Dict[str, Any], model: str, processing_time: float, filename: Optional[str] = None) -> str:
        """Generate Markdown with YAML frontmatter"""
        md_content = ["---"]

        if filename:
            md_content.append(f"source_file: {filename}")
        md_content.append(f"model: {model}")
        md_content.append(f"language: {result.get('language', 'auto')}")
        md_content.append(f"processing_time: {processing_time:.2f}")
        md_content.append(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        md_content.append("---")
        md_content.append("")

        text = result.get('text', '').strip()
        if text:
            md_content.append(text)

        return "\n".join(md_content)

    @staticmethod
    def format_transcription_response(
        result: Dict[str, Any],
        model: str,
        processing_time: float,
        response_format: str,
        filename: Optional[str] = None,
        status_msg: Optional[str] = None
    ) -> Union[PlainTextResponse, Dict[str, Any]]:
        """Format transcription response based on requested format"""
        if response_format == "text":
            return PlainTextResponse(result.get('text', ''))
        elif response_format == "srt":
            segments = result.get('segments', [])
            if segments:
                return PlainTextResponse(ResponseFormatter.generate_srt(segments))
            else:
                text = result.get('text', '')
                return PlainTextResponse(f"1\n00:00:00,000 --> 00:00:10,000\n{text}\n")
        elif response_format == "vtt":
            segments = result.get('segments', [])
            if segments:
                return PlainTextResponse(ResponseFormatter.generate_vtt(segments))
            else:
                text = result.get('text', '')
                return PlainTextResponse(f"WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n{text}\n")
        elif response_format in ["md", "markdown"]:
            return PlainTextResponse(ResponseFormatter.generate_markdown(result, model, processing_time, filename=filename))
        else:
            # Enhanced JSON format (default)
            response_data = {
                "text": result.get('text', ''),
                "language": result.get('language', 'auto'),
                "model": model,
                "processing_time": round(processing_time, 2),
                "file_duration": result.get('duration', 0),
                "real_time_factor": round(result.get('duration', 0) / processing_time, 1) if processing_time > 0 else 0,
                "segments": result.get('segments', []),
                "words": result.get('words', []) if 'words' in result else None
            }

            if status_msg:
                response_data["status"] = status_msg

            return response_data

# =============================================================================
# Utility Functions
# =============================================================================

def format_size(bytes_size: int) -> str:
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}TB"

# =============================================================================
# FastAPI Application Setup
# =============================================================================

app = FastAPI(
    title="MLX Whisper API",
    description="Fast speech-to-text API using Apple's MLX framework",
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

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="MLX Whisper API is running",
        version="1.0.0",
        hardware="Apple Silicon with MLX acceleration"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Detailed health check"""
    try:
        import mlx.core as mx
        return HealthResponse(
            status="healthy",
            mlx_available=True,
            mlx_device=str(mx.default_device())
        )
    except ImportError:
        return HealthResponse(
            status="degraded",
            mlx_available=False,
            error="MLX not available"
        )

@app.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """List available Whisper models"""
    return ModelsResponse(
        available_models=list(MODEL_MAPPING.keys()),
        default_model=DEFAULT_MODEL,
        recommended="tiny (fastest), medium (balanced), turbo-it (best for Italian)",
        endpoint="/transcribe",
        cache_location=os.path.expanduser("~/.cache/huggingface/hub")
    )

@app.post("/transcribe", response_model=None)
async def transcribe_audio(
    media: Optional[str] = Form(None),
    file: Optional[UploadFile] = None,
    model: Optional[str] = Form("medium"),
    language: Optional[str] = Form(None),
    temperature: Optional[float] = Form(0.0),
    response_format: Optional[str] = Form("json"),
    stream: Optional[bool] = Form(False)
) -> Union[StreamingResponse, PlainTextResponse, Dict[str, Any]]:
    """
    Transcribe audio using MLX-Whisper with automatic media detection

    Supports local files, YouTube videos/playlists, and remote audio URLs
    """
    start_time = time.time()

    # Validate model parameter
    if model not in MODEL_MAPPING:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Available: {', '.join(sorted(MODEL_MAPPING.keys()))}"
        )

    # Validate response_format parameter
    if response_format not in VALID_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response_format '{response_format}'. Must be one of: {', '.join(sorted(VALID_RESPONSE_FORMATS))}"
        )

    logger.info(f"Transcription request: model={model}, format={response_format}, language={language or 'auto'}")

    # Process media input
    if media is None and file is not None:
        media_input, media_type, media_name = file, 'local_file', file.filename
    elif isinstance(media, str):
        media_type = MediaProcessor.detect_media_type(media)
        media_input, media_name = media, media
    elif hasattr(media, 'filename'):
        media_input, media_type, media_name = media, 'local_file', media.filename
    else:
        raise HTTPException(
            status_code=400,
            detail="No media input provided. Use 'media' parameter for files, URLs, or YouTube links."
        )

    # Read file content first if it's an UploadFile (needed for streaming mode)
    file_content = None
    if media_type == 'local_file' and hasattr(media_input, 'read'):
        file_content = await media_input.read()

    # Handle streaming mode
    if stream:
        def streaming_generator():
            cleanup_files = []
            cleanup_dirs = []

            try:
                yield f"🎯 Media Type: {media_type}\n"
                yield f"📁 Input: {media_name}\n"
                yield f"🤖 Model: {model} (MLX-Whisper)\n\n"

                # Process different media types
                audio_files = []

                if media_type == 'local_file':
                    if hasattr(media_input, 'read'):
                        temp_path = MediaProcessor.process_uploaded_file(media_input, file_content)
                        audio_files.append(temp_path)
                        cleanup_files.append(temp_path)
                        yield f"📁 File uploaded: {media_input.filename} ({format_size(len(file_content))})\n"
                    else:
                        if not os.path.exists(media_input):
                            yield f"❌ File not found: {media_input}\n"
                            return
                        audio_files.append(media_input)
                        yield f"📁 Local file: {media_input}\n"

                elif media_type == 'youtube_video':
                    yield "🔄 Downloading YouTube video...\n"
                    temp_dir = tempfile.mkdtemp()
                    cleanup_dirs.append(temp_dir)

                    download_result = MediaProcessor.download_youtube_audio(media_input, temp_dir)
                    audio_files.append(download_result['filename'])
                    cleanup_files.append(download_result['filename'])
                    yield f"✅ Downloaded: {download_result['title']}\n"

                elif media_type == 'remote_audio':
                    yield "🔄 Downloading remote audio...\n"
                    temp_dir = tempfile.mkdtemp()
                    cleanup_dirs.append(temp_dir)

                    downloaded_file = MediaProcessor.download_remote_audio(media_input, temp_dir)
                    audio_files.append(downloaded_file)
                    cleanup_files.append(downloaded_file)
                    yield f"✅ Downloaded: {os.path.basename(downloaded_file)}\n"

                # Transcribe files
                yield f"\n🎯 Starting transcription of {len(audio_files)} file(s)...\n"

                options = WhisperService.prepare_options(model, language, temperature, response_format)
                results = []

                for i, audio_file in enumerate(audio_files, 1):
                    if len(audio_files) > 1:
                        yield f"📝 Processing file {i}/{len(audio_files)}: {os.path.basename(audio_file)}\n"

                    result = WhisperService.transcribe_file(audio_file, options)
                    results.append(result)

                    if len(audio_files) > 1:
                        yield f"✅ Completed file {i}/{len(audio_files)}\n"

                processing_time = time.time() - start_time
                yield f"✅ All transcriptions completed in {processing_time:.2f}s!\n\n"

                # Format output
                yield "📝 TRANSCRIPTION RESULT(S):\n"
                yield "=" * 48 + "\n\n"

                if len(results) == 1:
                    result = results[0]
                    if response_format == "text":
                        yield result.get('text', '')
                    elif response_format in ["md", "markdown"]:
                        yield ResponseFormatter.generate_markdown(
                            result, model, processing_time, filename=result.get('source_file', media_name)
                        )
                    else:
                        yield f"Text: {result.get('text', '')}\n"
                        yield f"Language: {result.get('language', 'auto')}\n"
                else:
                    for i, result in enumerate(results, 1):
                        yield f"\n--- File {i}: {result.get('source_file', f'file_{i}')} ---\n"
                        if response_format == "text":
                            yield result.get('text', '')
                        else:
                            yield f"Text: {result.get('text', '')}\n"
                        yield "\n"

            except Exception as e:
                yield f"❌ Error: {str(e)}\n"
            finally:
                # Cleanup
                for cleanup_file in cleanup_files:
                    try:
                        if os.path.exists(cleanup_file):
                            os.unlink(cleanup_file)
                    except Exception:
                        pass
                for cleanup_dir in cleanup_dirs:
                    try:
                        if os.path.exists(cleanup_dir):
                            shutil.rmtree(cleanup_dir)
                    except Exception:
                        pass

        return StreamingResponse(
            streaming_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # Non-streaming mode
    cleanup_files = []
    cleanup_dirs = []

    try:
        # Process media
        audio_files = []

        if media_type == 'local_file':
            if hasattr(media_input, 'read'):
                if file_content is None:
                    file_content = await media_input.read()
                temp_path = MediaProcessor.process_uploaded_file(media_input, file_content)
                audio_files.append(temp_path)
                cleanup_files.append(temp_path)
            else:
                if not os.path.exists(media_input):
                    raise HTTPException(status_code=404, detail=f"File not found: {media_input}")
                audio_files.append(media_input)

        elif media_type in ['youtube_video', 'youtube_playlist']:
            temp_dir = tempfile.mkdtemp()
            cleanup_dirs.append(temp_dir)
            download_result = MediaProcessor.download_youtube_audio(media_input, temp_dir)

            if download_result['type'] == 'single':
                audio_files.append(download_result['filename'])
                cleanup_files.append(download_result['filename'])

        elif media_type == 'remote_audio':
            temp_dir = tempfile.mkdtemp()
            cleanup_dirs.append(temp_dir)
            downloaded_file = MediaProcessor.download_remote_audio(media_input, temp_dir)
            audio_files.append(downloaded_file)
            cleanup_files.append(downloaded_file)

        # Transcribe
        options = WhisperService.prepare_options(model, language, temperature, response_format)
        results = []

        for audio_file in audio_files:
            logger.info(f"Transcribing: {os.path.basename(audio_file)}")
            result = WhisperService.transcribe_file(audio_file, options)
            results.append(result)

        processing_time = time.time() - start_time
        logger.info(f"Transcription completed in {processing_time:.2f}s, real-time factor: {results[0].get('duration', 0) / processing_time if processing_time > 0 else 0:.1f}x")

        # Format response for single file
        if len(results) == 1:
            result = results[0]

            if response_format == "text":
                return PlainTextResponse(result.get('text', ''))
            elif response_format == "srt":
                segments = result.get('segments', [])
                return PlainTextResponse(ResponseFormatter.generate_srt(segments) if segments else f"1\n00:00:00,000 --> 00:00:10,000\n{result.get('text', '')}\n")
            elif response_format == "vtt":
                segments = result.get('segments', [])
                return PlainTextResponse(ResponseFormatter.generate_vtt(segments) if segments else f"WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n{result.get('text', '')}\n")
            elif response_format in ["md", "markdown"]:
                return PlainTextResponse(ResponseFormatter.generate_markdown(result, model, processing_time, filename=media_name))
            else:
                return {
                    "text": result.get('text', ''),
                    "language": result.get('language', 'auto'),
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
        else:
            # Multiple files response
            total_duration = sum(r.get('duration', 0) for r in results)
            all_text = "\n\n".join(r.get('text', '') for r in results)

            return {
                "text": all_text,
                "language": ", ".join(set(r.get('language', 'auto') for r in results)),
                "model": model,
                "processing_time": round(processing_time, 2),
                "file_duration": total_duration,
                "real_time_factor": round(total_duration / processing_time, 1) if processing_time > 0 else 0,
                "framework": "MLX-Whisper",
                "media_type": media_type,
                "media_source": media_name,
                "file_count": len(results),
                "results": results
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Cleanup
        for cleanup_file in cleanup_files:
            try:
                if os.path.exists(cleanup_file):
                    os.unlink(cleanup_file)
            except Exception:
                pass
        for cleanup_dir in cleanup_dirs:
            try:
                if os.path.exists(cleanup_dir):
                    shutil.rmtree(cleanup_dir)
            except Exception:
                pass

# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "whisper_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )