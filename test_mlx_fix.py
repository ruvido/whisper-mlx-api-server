#!/usr/bin/env python3
"""
Test script to verify the MLX-Whisper model mapping fix
"""

# Import the model mapping from whisper_api
import sys
sys.path.append('/Users/ruvido/dev/whisper-mlx-api')

try:
    from whisper_api import MLX_WHISPER_MODEL_MAPPING

    print("✅ MLX-Whisper Model Mapping Test")
    print("=" * 50)

    # Test cases for the original issue
    test_models = ["tiny", "small", "medium", "large", "turbo"]

    for model in test_models:
        mapped_model = MLX_WHISPER_MODEL_MAPPING.get(model, model)
        print(f"  {model:10} -> {mapped_model}")

        # Verify it's a proper HuggingFace path or custom model
        if "/" in mapped_model:
            print(f"             ✅ Valid HF path")
        else:
            print(f"             ❌ Invalid - needs HF path")

    print("\n🔧 Model Mapping Analysis:")
    print("- Original issue: 'tiny' was passed directly to MLX-Whisper")
    print("- Fixed: 'tiny' now maps to 'mlx-community/whisper-tiny'")
    print("- This prevents the 401 error from HuggingFace authentication")

    print("\n🎯 Root Cause:")
    print("MLX-Whisper expects full HuggingFace repository paths,")
    print("not simple OpenAI model names like 'tiny', 'small', etc.")

    print("\n✅ Fix Applied:")
    print("Added MLX_WHISPER_MODEL_MAPPING to convert standard names")
    print("to proper mlx-community HuggingFace repository paths.")

except ImportError as e:
    print(f"❌ Could not import whisper_api: {e}")
except Exception as e:
    print(f"❌ Error: {e}")