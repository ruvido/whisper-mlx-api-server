#!/usr/bin/env python3
"""
Simulate model download to test progress indicator
"""

import os
import sys
import time
import threading
import tempfile
import random

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
    import glob

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

def simulate_model_download(model_name="test-model", cache_dir=None):
    """Simulate downloading model files to test progress indicator"""
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="whisper_test_")

    print(f"🧪 Simulating download for model: {model_name}")
    print(f"📁 Test cache directory: {cache_dir}")

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
        # Simulate downloading several model files
        files_to_create = [
            (f"pytorch_model-{model_name}.bin", 150 * 1024 * 1024),  # 150MB model file
            (f"tokenizer-{model_name}.json", 2 * 1024 * 1024),       # 2MB tokenizer
            (f"config-{model_name}.json", 4 * 1024),                # 4KB config
            (f"vocab-{model_name}.txt", 500 * 1024)                 # 500KB vocab
        ]

        for filename, target_size in files_to_create:
            filepath = os.path.join(cache_dir, filename)
            print(f"\n🔄 Creating {filename}...")

            # Simulate progressive file creation
            chunk_size = target_size // 20  # 20 chunks
            with open(filepath, 'wb') as f:
                written = 0
                while written < target_size:
                    chunk = min(chunk_size, target_size - written)
                    f.write(b'0' * chunk)
                    f.flush()  # Ensure data is written
                    written += chunk
                    time.sleep(0.2)  # Simulate network delay

            time.sleep(0.5)  # Brief pause between files

    finally:
        downloading_flag[0] = False
        progress_thread.join(timeout=1.0)
        print()  # New line after progress

    print(f"✅ Simulation complete!")
    print(f"📂 Test files created in: {cache_dir}")

    # List created files
    print("\n📋 Created files:")
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath)
            print(f"  📄 {file}: {format_size(size)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulate model download to test progress indicator")
    parser.add_argument("--model", default="test-model", help="Model name to simulate (default: test-model)")
    parser.add_argument("--cache-dir", help="Cache directory (default: temporary)")
    args = parser.parse_args()

    simulate_model_download(args.model, args.cache_dir)