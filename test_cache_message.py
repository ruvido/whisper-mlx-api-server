#!/usr/bin/env python3
"""
Test the cache message logic directly
"""

import os
import sys
import glob

# Add current directory to path
sys.path.insert(0, '.')

def format_size(bytes_size):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}TB"

def test_cache_detection(model_name="tiny"):
    """Test cache detection logic"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    print(f"🧪 Testing cache detection for model: {model_name}")
    print(f"📁 Cache directory: {cache_dir}")

    if not os.path.exists(cache_dir):
        print("📭 Cache directory doesn't exist")
        return

    # Check for model files
    model_pattern = f"*{model_name}*"
    existing_files = []

    print(f"🔍 Searching for pattern: {model_pattern}")

    for pattern_path in glob.glob(os.path.join(cache_dir, "**", model_pattern), recursive=True):
        if os.path.isfile(pattern_path):
            existing_files.append(pattern_path)
            print(f"  📄 Found: {pattern_path} ({format_size(os.path.getsize(pattern_path))})")

    if existing_files:
        total_size = sum(os.path.getsize(f) for f in existing_files)
        print(f"📦 Model '{model_name}' already cached ({format_size(total_size)})")
        print(f"🔄 Loading cached model...")
    else:
        print(f"🔄 Loading model '{model_name}' (first download)...")

    print()

if __name__ == "__main__":
    # Test with different model names
    test_cache_detection("tiny")
    test_cache_detection("distil-large-v3")
    test_cache_detection("base")
    test_cache_detection("medium")