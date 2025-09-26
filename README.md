# 🚀 Lightning MLX Whisper API

Ultra-fast speech-to-text API server optimized for Apple Silicon with **dual Whisper implementations**.

## ⚡ Performance

- **27x real-time** transcription speed on Mac M4
- **10x faster** than Whisper CPP
- **Dual endpoints** for maximum flexibility
- Full hardware acceleration with Apple's Neural Engine

## 🛠️ Installation

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+ (managed by UV)
- FFmpeg (installed automatically)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd whisper-mlx-api
   ```

2. **Install dependencies** (UV handles everything):
   ```bash
   uv sync
   ```

3. **Start the server**:
   ```bash
   ./start_server.sh
   ```

## 🔌 API Usage

### Lightning Whisper MLX (Recommended)
```bash
curl -N --no-buffer -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.mp3" \
  -F "model=turbo" \
  -F "language=it" \
  -F "stream=true"
```

### MLX-Whisper (Alternative)
```bash
curl -N --no-buffer -X POST "http://localhost:8000/transcribe/mlx" \
  -F "file=@audio.mp3" \
  -F "model=turbo" \
  -F "language=it" \
  -F "stream=true"
```

### Transcribe from URL (YouTube, etc.)
```bash
curl -X POST "http://localhost:8000/transcribe/url" \
  -F "url=https://youtube.com/watch?v=..." \
  -F "model=medium"
```

## 📊 API Endpoints

### Core Endpoints
- `GET /` - Health check
- `GET /health` - Detailed health status with MLX status
- `GET /models` - List available models for both frameworks
- `POST /transcribe` - **Lightning Whisper MLX** (fastest)
- `POST /transcribe/mlx` - **MLX-Whisper** (alternative)
- `POST /transcribe/url` - Transcribe from URL (YouTube, etc.)

### Model Management
- `POST /models/preload` - Preload model to cache for faster first use
- `GET /cache/status` - Get cache status and statistics
- `POST /cache/clear` - Clear runtime model cache

### Output Formats
- **json**: Enhanced JSON with segments and metadata (default)
- **text**: Plain text transcription
- **srt**: SRT subtitle format with proper timestamps
- **vtt**: WebVTT subtitle format
- **md/markdown**: Markdown with metadata and timestamped segments

### Model Caching
- **Location**: `~/.cache/huggingface/hub/`
- **Behavior**: Models download once and persist across restarts
- **Runtime Cache**: Models stay loaded in memory for faster repeated use

## ⚙️ Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

## 🎯 Supported Formats

Audio: MP3, WAV, M4A, FLAC, OGG, MP4, MOV

## 📈 Response Format

```json
{
  "text": "Transcribed text here",
  "language": "en",
  "model": "distil-medium.en",
  "processing_time": 1.2,
  "file_duration": 30.0,
  "real_time_factor": 25.0
}
```

## 🔧 Production Deployment

### Using systemd (Linux-like)
```bash
sudo cp whisper-mlx-api.service /etc/systemd/system/
sudo systemctl enable whisper-mlx-api
sudo systemctl start whisper-mlx-api
```

### Using launchd (macOS)
```bash
cp com.whisper-mlx-api.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.whisper-mlx-api.plist
```

## 🚨 Troubleshooting

### MLX Not Available
- Ensure you're on Apple Silicon Mac
- Check: `python -c "import mlx.core; print('MLX OK')"`

### Performance Issues
- Use `distil-medium.en` for best speed/accuracy balance
- Increase batch_size for longer audio files
- Ensure sufficient RAM (8GB+ recommended)

### Memory Issues
- Reduce batch_size in configuration
- Use smaller models (tiny, small)
- Monitor with `top` or Activity Monitor

## 📝 Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Development server (auto-reload)
uv run uvicorn whisper_api:app --reload --host 0.0.0.0 --port 8000
```