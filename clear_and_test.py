#!/usr/bin/env python3
"""
Clear cache and test download progress indicator
"""

import os
import shutil
import tempfile

def clear_model_cache(model_name="tiny"):
    """Clear specific model from cache"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    print(f"🗑️  Clearing cache for model: {model_name}")
    print(f"📁 Cache directory: {cache_dir}")

    if not os.path.exists(cache_dir):
        print("📭 Cache directory doesn't exist yet")
        return

    # Find and remove model-related files
    removed_files = []
    removed_size = 0

    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            if model_name in file.lower():
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path)
                    os.remove(file_path)
                    removed_files.append(file)
                    removed_size += size
                    print(f"  🗑️  Removed: {file} ({size/1024/1024:.1f}MB)")
                except (OSError, IOError) as e:
                    print(f"  ❌ Failed to remove {file}: {e}")

        # Also remove empty directories
        for dir_name in dirs:
            if model_name in dir_name.lower():
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):  # Only remove if empty
                        os.rmdir(dir_path)
                        print(f"  🗑️  Removed empty directory: {dir_name}")
                except (OSError, IOError) as e:
                    print(f"  ❌ Failed to remove directory {dir_name}: {e}")

    if removed_files:
        print(f"✅ Removed {len(removed_files)} files ({removed_size/1024/1024:.1f}MB total)")
    else:
        print(f"📭 No files found for model '{model_name}'")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clear model cache and test download")
    parser.add_argument("--model", default="tiny", help="Model to clear (default: tiny)")
    parser.add_argument("--audio-file", help="Audio file to test with after clearing")
    args = parser.parse_args()

    # Clear the cache
    clear_model_cache(args.model)

    if args.audio_file:
        print(f"\n🧪 Now testing with cleared cache...")
        import subprocess
        import sys

        # Run the test with the audio file
        cmd = [sys.executable, "test_with_file.py", args.audio_file, "--model", args.model]
        subprocess.run(cmd)
    else:
        print(f"\n💡 To test download progress, run:")
        print(f"   python3 test_with_file.py /path/to/audio/file --model {args.model}")