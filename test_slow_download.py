#!/usr/bin/env python3
"""
Test streaming progress with artificial slow download simulation
"""

import asyncio
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

async def simulate_model_download():
    """Simulate a slow model download with progress updates"""
    yield "📁 File: angelo-custode.ogg (3.9MB)\n"
    yield "🤖 Model: distil-large-v3\n\n"

    # Simulate model loading with progress
    yield "🔄 Loading model 'distil-large-v3'...\n"

    total_size = 1200 * 1024 * 1024  # 1.2GB model
    downloaded = 0
    chunk_size = 50 * 1024 * 1024  # 50MB chunks

    while downloaded < total_size:
        await asyncio.sleep(1)  # Simulate 1 second per chunk
        downloaded += chunk_size
        if downloaded > total_size:
            downloaded = total_size

        progress = (downloaded / total_size) * 100

        def format_size(bytes_size):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if bytes_size < 1024:
                    return f"{bytes_size:.1f}{unit}"
                bytes_size /= 1024
            return f"{bytes_size:.1f}TB"

        yield f"📥 Downloaded: {format_size(downloaded)} ({progress:.1f}%)\n"

    yield "✅ Model 'distil-large-v3' loaded successfully!\n\n"
    yield "🎯 Starting transcription...\n"

    # Simulate transcription
    await asyncio.sleep(2)

    yield "✅ Transcription completed!\n"
    yield "⏱️ Processing time: 25.47s\n"
    yield "⚡ Real-time factor: 8.2x\n\n"
    yield "="*50 + "\n"
    yield "📝 TRANSCRIPTION RESULT:\n"
    yield "="*50 + "\n\n"
    yield "Questo è il risultato della trascrizione simulata...\n"

@app.post("/test-progress")
async def test_progress():
    """Test endpoint for streaming progress"""
    return StreamingResponse(
        simulate_model_download(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test_slow_download:app", host="0.0.0.0", port=8001, reload=False)