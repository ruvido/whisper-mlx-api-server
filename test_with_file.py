#!/usr/bin/env python3
"""
Test the progress indicator using the actual audio file
"""

import os
import sys
import tempfile
import time
import threading
import glob

# Import the functions from our main script
sys.path.insert(0, '.')

def format_size(bytes_size):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}TB"

def monitor_download_progress(cache_dir: str, model_name: str, downloading_flag):
    """Monitor cache directory for download progress"""
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

def test_transcription_with_progress(audio_file_path: str, model: str = "tiny"):
    """Test transcription with progress indicator"""
    print(f"🎵 Testing transcription of: {audio_file_path}")
    print(f"🤖 Using model: {model}")

    # Check if file exists
    if not os.path.exists(audio_file_path):
        print(f"❌ File not found: {audio_file_path}")
        return

    file_size = os.path.getsize(audio_file_path)
    print(f"📏 File size: {format_size(file_size)}")

    # Model mapping
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
        "distil-small.en": "distil-small.en",
        "distil-medium.en": "distil-medium.en",
    }

    mapped_model = MODEL_MAPPING.get(model, model)
    cache_key = f"{mapped_model}_4"

    # Set cache directory
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"📁 Cache directory: {cache_dir}")
    print(f"🔄 Loading model '{mapped_model}'...")

    # Use a flag for thread communication
    downloading_flag = [True]

    # Start progress monitor thread
    progress_thread = threading.Thread(
        target=monitor_download_progress,
        args=(cache_dir, mapped_model, downloading_flag),
        daemon=True
    )
    progress_thread.start()

    start_time = time.time()

    try:
        # Try to load and use the model
        try:
            from lightning_whisper_mlx import LightningWhisperMLX

            # Load model with progress monitoring
            whisper = LightningWhisperMLX(
                model=mapped_model,
                batch_size=4,
                quant=None
            )

            # Stop progress monitoring
            downloading_flag[0] = False
            progress_thread.join(timeout=1.0)

            load_time = time.time() - start_time
            print(f"\n✅ Model '{mapped_model}' loaded in {load_time:.2f}s")

            # Transcribe the audio file
            print(f"🎯 Transcribing audio...")
            transcribe_start = time.time()

            result = whisper.transcribe(audio_file_path)

            transcribe_time = time.time() - transcribe_start

            print(f"✅ Transcription completed in {transcribe_time:.2f}s")
            print(f"📝 Transcribed text: {result.get('text', '')[:200]}...")

            if 'duration' in result:
                real_time_factor = result['duration'] / transcribe_time if transcribe_time > 0 else 0
                print(f"⚡ Real-time factor: {real_time_factor:.1f}x")

        except ImportError:
            print(f"\n⚠️  lightning_whisper_mlx not available")
            print("📦 Install with: pip install lightning-whisper-mlx")
            # Still show progress monitoring working
            print("🔄 Simulating download for 5 seconds...")
            time.sleep(5)
            downloading_flag[0] = False
            progress_thread.join(timeout=1.0)
            print(f"\n✅ Test completed!")

        except Exception as e:
            downloading_flag[0] = False
            progress_thread.join(timeout=1.0)
            print(f"\n❌ Error: {e}")

    finally:
        downloading_flag[0] = False
        if progress_thread.is_alive():
            progress_thread.join(timeout=1.0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test transcription with progress indicator")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--model", default="tiny", help="Model to use (default: tiny)")
    args = parser.parse_args()

    test_transcription_with_progress(args.audio_file, args.model)