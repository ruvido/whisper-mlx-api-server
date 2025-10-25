# 🎙️ Whisper MLX API

Trascrizione audio ultrarapida per Apple Silicon. Ottimizzato per italiano.

## Quick Start

```bash
uv sync && ./whisper-api-server.sh
```

Server: `http://localhost:8000` | Docs: `http://localhost:8000/docs`

## Esempi

**File → Markdown**
```bash
curl -X POST http://localhost:8000/transcribe \
  -F file=@audio.mp3 -F model=turbo -F language=it -F response_format=md
```

**YouTube → Text**
```bash
curl -X POST http://localhost:8000/transcribe \
  -F media=https://youtu.be/VIDEO_ID -F model=turbo -F language=it -F response_format=text
```

**Streaming (con progress)**
```bash
curl -N -X POST http://localhost:8000/transcribe \
  -F file=@audio.mp3 -F model=turbo -F language=it -F stream=true
```

**Sottotitoli SRT**
```bash
curl -X POST http://localhost:8000/transcribe \
  -F file=@video.mp4 -F model=turbo -F language=it -F response_format=srt > output.srt
```

## Formati Output

`text` | `md` | `srt` | `vtt` | `json` (default)

Usa: `-F "response_format=FORMATO"`

## Modelli

**Consigliato:** `turbo` (veloce + accurato per italiano)
Altri: `tiny`, `base`, `small`, `large`

## Deployment (Always-On)

```bash
cp com.whisper-mlx-api.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.whisper-mlx-api.plist
```

Auto-start al boot + auto-restart se crasha. Log: `~/Library/Logs/WhisperMLX/whisper-api.log`

## Info Utili

**Formati audio:** MP3, WAV, M4A, FLAC, OGG, MP4, MOV (max 100MB)

**Performance:** Prima richiesta ~5-10s (carica modello), successive 10-100x più veloci (cache)

**Cache modelli:** `~/.cache/huggingface/hub/` - Pulizia: `rm -rf ~/.cache/huggingface/hub/models--*whisper*`

**API:** `GET /health`, `GET /models`, `POST /transcribe`

---

**🇮🇹 Ottimizzato per italiano**
