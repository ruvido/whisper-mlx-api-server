# 🎙️ Whisper MLX API - Trascrizione Audio Ultrarapida per Apple Silicon

API server di trascrizione audio ottimizzato per Apple Silicon, con supporto specializzato per l'**italiano**.

## 🚀 TL;DR - Quick Start

```bash
# 1. Avvia il server
./start_server.sh

# 2. Trascrivi un file audio in italiano (in un altro terminale)
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@tuofile.mp3" \
  -F "model=turbo" \
  -F "language=it"
```

Il server parte su `http://localhost:8000`. Il modello viene scaricato automaticamente al primo utilizzo.

## ⚡ Caratteristiche Principali

- **Ottimizzato per Apple Silicon** - Sfrutta completamente MLX e Neural Engine
- **Modelli specializzati per l'italiano** - turbo-it, italian-turbo, distil-it
- **Supporto universale media** - File locali, YouTube, URL remoti
- **Streaming real-time** - Vedi i progressi durante la trascrizione
- **Formati multipli** - JSON, text, SRT, VTT, Markdown

## 🌐 Modelli Disponibili

| Modello | Tipo | Lingua | Raccomandato per |
|---------|------|--------|------------------|
| `tiny` | Veloce | Multilingua | Test rapidi |
| `base` | Bilanciato | Multilingua | Uso generale |
| `small` | Buono | Multilingua | Qualità media |
| `medium` | **Default** | Multilingua | 🇮🇹 Italiano generico |
| `large` | Massimo | Multilingua | Massima accuratezza |
| `turbo` | Veloce | Multilingua | Performance |
| **`turbo-it`** | **🏆 Ottimale** | **Italiano** | **✅ Consigliato italiano** |
| `italian-turbo` | Ottimale | Italiano | Alternativa turbo-it |
| `distil-it` | Accurato | Italiano | Italiano accurato |

**Modello consigliato per italiano:** `turbo-it` (bofenghuang/whisper-large-v3-distil-it-v0.2)

## 🛠️ Installazione

### Prerequisiti
- macOS con Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- FFmpeg (installato automaticamente)

### Quick Start

```bash
# 1. Clona il repository
git clone <your-repo-url>
cd whisper-mlx-api

# 2. Installa dipendenze (UV gestisce tutto)
uv sync

# 3. Avvia il server
./start_server.sh
```

Il server sarà disponibile su `http://localhost:8000`

## 📡 Utilizzo API

### Esempi Pratici per l'Italiano

#### File locale con modello italiano
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.mp3" \
  -F "model=turbo-it" \
  -F "language=it"
```

#### Con streaming (vedi progresso in tempo reale)
```bash
curl -N --no-buffer -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.mp3" \
  -F "model=turbo-it" \
  -F "language=it" \
  -F "stream=true"
```

#### Video YouTube
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "media=https://youtu.be/VIDEO_ID" \
  -F "model=turbo-it" \
  -F "language=it"
```

#### URL audio remoto
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "media=https://example.com/audio.mp3" \
  -F "model=turbo-it" \
  -F "language=it"
```

#### Output come sottotitoli SRT
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.mp3" \
  -F "model=turbo-it" \
  -F "language=it" \
  -F "response_format=srt"
```

#### Output come Markdown
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.mp3" \
  -F "model=turbo-it" \
  -F "language=it" \
  -F "response_format=markdown"
```

## 📊 Endpoint API

### Core Endpoints

- **`GET /`** - Health check
- **`GET /health`** - Stato dettagliato con info MLX
- **`GET /models`** - Lista modelli disponibili
- **`POST /transcribe`** - Trascrizione audio (supporta file e media)

### Parametri `/transcribe`

| Parametro | Tipo | Default | Descrizione |
|-----------|------|---------|-------------|
| `file` | UploadFile | - | File audio da trascrivere |
| `media` | str | - | URL YouTube o audio remoto |
| `model` | str | `"medium"` | Modello da usare |
| `language` | str | auto | Lingua (es: `"it"`) |
| `temperature` | float | `0.0` | Temperature per sampling (0.0-1.0) |
| `response_format` | str | `"json"` | Formato output |
| `stream` | bool | `false` | Streaming real-time |

**Note:**
- Usa `file` per file locali (multipart form upload)
- Usa `media` per YouTube URL o remote audio URL

### Formati Output

| Formato | Descrizione | Uso |
|---------|-------------|-----|
| `json` | JSON completo con segments, timestamps, metadata | Default, più completo |
| `text` | Solo testo trascritto | Semplice, leggibile |
| `srt` | Sottotitoli SRT con timestamps | Video editing |
| `vtt` | WebVTT subtitles | Web player |
| `markdown` | Markdown con frontmatter YAML | Documentazione |

### Formati Audio Supportati

MP3, WAV, M4A, FLAC, OGG, MP4, MOV

Dimensione massima: 100MB

## 📈 Esempio Response (JSON)

```json
{
  "text": "Ciao, questo è un test di trascrizione in italiano.",
  "language": "it",
  "model": "turbo-it",
  "processing_time": 1.2,
  "file_duration": 30.0,
  "real_time_factor": 25.0,
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "Ciao, questo è un test"
    }
  ],
  "framework": "MLX-Whisper",
  "media_type": "local_file"
}
```

## 🔧 Deployment Produzione

### macOS (launchd)

```bash
# Copia il file plist
cp com.whisper-mlx-api.plist ~/Library/LaunchAgents/

# Carica il servizio
launchctl load ~/Library/LaunchAgents/com.whisper-mlx-api.plist

# Avvia automaticamente al login
launchctl enable gui/$UID/com.whisper-mlx-api
```

### Configurazione

Puoi configurare il server tramite variabili d'ambiente:

```bash
# Esempio:
HOST=0.0.0.0 PORT=8000 LOG_LEVEL=info ./start_server.sh
```

Variabili disponibili:
- `HOST` (default: 0.0.0.0)
- `PORT` (default: 8000)
- `LOG_LEVEL` (default: info)

## 🚨 Troubleshooting

### MLX Non Disponibile

Assicurati di essere su Mac Apple Silicon:
```bash
python -c "import mlx.core as mx; print(f'MLX OK: {mx.default_device()}')"
```

Se MLX non è disponibile:
```bash
uv add mlx mlx-whisper
```

### Performance Issues

- **Usa modelli più piccoli** per file lunghi: `tiny`, `base`, `small`
- **Aumenta RAM disponibile** (consigliato 8GB+)
- **Chiudi altre app** per liberare Neural Engine
- **Modello consigliato per italiano**: `turbo-it` (veloce e accurato)

### Errori comuni

**"Model not found"**
- Il modello viene scaricato al primo uso
- Controlla la connessione internet
- Cache: `~/.cache/huggingface/hub`

**"File too large"**
- Limite: 100MB
- Comprimi o dividi il file audio

**"Unsupported format"**
- Formati supportati: MP3, WAV, M4A, FLAC, OGG, MP4, MOV
- Converti con ffmpeg se necessario

## 📝 Sviluppo

```bash
# Installa dipendenze dev
uv sync --extra dev

# Esegui test
uv run pytest

# Server con auto-reload
uv run uvicorn whisper_api:app --reload --host 0.0.0.0 --port 8000

# Controlla health
curl http://localhost:8000/health
```

## 🎯 Cache dei Modelli

I modelli vengono scaricati automaticamente da HuggingFace al primo utilizzo e salvati in:

```
~/.cache/huggingface/hub/
```

Una volta scaricati, vengono riutilizzati per le successive trascrizioni.

**⚡ Performance**: I modelli vengono caricati in memoria al primo utilizzo e mantenuti in cache. Le richieste successive con lo stesso modello sono **10-100x più veloci** perché il modello è già caricato.

**Pulizia cache disco:**
```bash
rm -rf ~/.cache/huggingface/hub/models--*whisper*
```

**Pulizia cache memoria** (riavvia il server):
```bash
# Trova e termina il processo
lsof -ti:8000 | xargs kill
# Riavvia
./start_server.sh
```

## 🌟 Best Practices

1. **Per italiano**: Usa `model=turbo` e `language=it` (veloce e accurato)
2. **Streaming**: Usa `stream=true` per file lunghi per vedere il progresso in tempo reale
3. **Formati**: Preferisci MP3 o M4A per le migliori performance
4. **Performance**: Mantieni il server attivo - il modello rimane in cache per richieste successive 10-100x più veloci
5. **Batch**: Per molti file, riutilizza la stessa sessione server

## 🤝 Contributi

Contributi benvenuti! Apri una issue o una pull request.

## 📄 Licenza

MIT License

---

**🇮🇹 Ottimizzato per la lingua italiana | Powered by MLX-Whisper**
