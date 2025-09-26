#!/usr/bin/env python3
"""
Simple test script to test the model download progress indicator
"""

import os
import sys
import tempfile
import time
import threading
import glob

# Add the current directory to path so we can import our functions
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

def test_model_loading(model_name="tiny"):
    """Test model loading with progress indicator"""
    print(f"🧪 Testing progress indicator for model: {model_name}")

    # Set cache directory
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"📁 Cache directory: {cache_dir}")
    print(f"🔄 Loading model '{model_name}'...")

    # Use a flag for thread communication
    downloading_flag = [True]

    # Start progress monitor thread
    progress_thread = threading.Thread(
        target=monitor_download_progress,
        args=(cache_dir, model_name, downloading_flag),
        daemon=True
    )
    progress_thread.start()

    try:
        # Try to import and load the model
        try:
            from lightning_whisper_mlx import LightningWhisperMLX
            model = LightningWhisperMLX(model=model_name, batch_size=4, quant=None)
            print(f"\n✅ Model '{model_name}' loaded successfully!")
            return model
        except ImportError:
            print(f"\n⚠️  lightning_whisper_mlx not available. Install with: pip install lightning-whisper-mlx")
            # Simulate download time for testing
            print("🔄 Simulating download for 10 seconds...")
            time.sleep(10)
            print(f"\n✅ Simulation complete!")
            return None
        except Exception as e:
            print(f"\n❌ Failed to load model: {e}")
            return None
    finally:
        downloading_flag[0] = False
        progress_thread.join(timeout=1.0)
        print()  # New line after progress

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test model download progress indicator")
    parser.add_argument("--model", default="tiny", help="Model to test (default: tiny)")
    args = parser.parse_args()

    test_model_loading(args.model)