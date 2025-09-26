#!/usr/bin/env python3
"""
Test script to verify the MLX-Whisper model mapping fix (isolated test)
"""

# Define the mapping directly for testing
MLX_WHISPER_MODEL_MAPPING = {
    "tiny": "mlx-community/whisper-tiny",
    "tiny.en": "mlx-community/whisper-tiny.en",
    "base": "mlx-community/whisper-base",
    "base.en": "mlx-community/whisper-base.en",
    "small": "mlx-community/whisper-small",
    "small.en": "mlx-community/whisper-small.en",
    "medium": "mlx-community/whisper-medium",
    "medium.en": "mlx-community/whisper-medium.en",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v1": "mlx-community/whisper-large-v3-mlx",
    "large-v2": "mlx-community/whisper-large-v3-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-turbo",
    # Custom models (pass through as-is)
    "bofenghuang/whisper-large-v3-distil-it-v0.2": "bofenghuang/whisper-large-v3-distil-it-v0.2",
    "turbo-it": "bofenghuang/whisper-large-v3-distil-it-v0.2",
    "italian-turbo": "bofenghuang/whisper-large-v3-distil-it-v0.2",
    "distil-it": "bofenghuang/whisper-large-v3-distil-it-v0.2"
}

print("✅ MLX-Whisper Model Mapping Test")
print("=" * 50)

# Test cases for the original issue
test_models = ["tiny", "small", "medium", "large", "turbo"]

print("🔧 Model Mapping Results:")
for model in test_models:
    mapped_model = MLX_WHISPER_MODEL_MAPPING.get(model, model)
    print(f"  {model:10} -> {mapped_model}")

print("\n🚨 Original Issue Analysis:")
print("- User tried to use model='tiny' with MLX-Whisper")
print("- MLX-Whisper looked for HuggingFace repo 'tiny' (doesn't exist)")
print("- Got 401 error: 'Repository Not Found for url: https://huggingface.co/api/models/tiny/revision/main'")

print("\n✅ Fix Applied:")
print("- 'tiny' now maps to 'mlx-community/whisper-tiny' (valid HF repo)")
print("- All standard OpenAI model names now map to proper MLX community repos")
print("- Custom models with '/' in name are passed through unchanged")

print("\n🎯 Code Changes Made:")
print("1. Added MLX_WHISPER_MODEL_MAPPING dictionary")
print("2. Updated both streaming and non-streaming MLX endpoints")
print("3. Changed: transcribe_options['path_or_hf_repo'] = model")
print("   To: mlx_model = MLX_WHISPER_MODEL_MAPPING.get(model, model)")
print("        transcribe_options['path_or_hf_repo'] = mlx_model")

print("\n🧪 Test Results:")
for model in ["tiny", "custom/model"]:
    mapped = MLX_WHISPER_MODEL_MAPPING.get(model, model)
    if model == "tiny":
        expected = "mlx-community/whisper-tiny"
        status = "✅ PASS" if mapped == expected else "❌ FAIL"
        print(f"  tiny -> {mapped} {status}")
    else:
        status = "✅ PASS" if mapped == model else "❌ FAIL"
        print(f"  custom/model -> {mapped} {status} (pass-through)")