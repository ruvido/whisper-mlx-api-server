#!/usr/bin/env python3
"""
Functional test for Whisper MLX API
Tests transcription with real Italian audio file using turbo model
"""

import requests
import sys
import os
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
TEST_AUDIO = os.path.expanduser("~/Desktop/angelo-custode.ogg")
MODEL = "turbo"
LANGUAGE = "it"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")

    if response.status_code != 200:
        print(f"❌ Health check failed: {response.status_code}")
        return False

    data = response.json()
    print(f"✅ Health check passed")
    print(f"   Status: {data['status']}")
    print(f"   MLX Available: {data['mlx_available']}")
    print(f"   Device: {data['mlx_device']}")
    return True

def test_models():
    """Test models endpoint"""
    print("\n🔍 Testing models endpoint...")
    response = requests.get(f"{API_URL}/models")

    if response.status_code != 200:
        print(f"❌ Models endpoint failed: {response.status_code}")
        return False

    data = response.json()
    print(f"✅ Models endpoint passed")
    print(f"   Available models: {len(data['available_models'])}")
    print(f"   Default model: {data['default_model']}")
    return True

def test_transcription():
    """Test transcription with Italian audio file"""
    print(f"\n🔍 Testing transcription...")
    print(f"   File: {TEST_AUDIO}")
    print(f"   Model: {MODEL}")
    print(f"   Language: {LANGUAGE}")

    # Check if file exists
    if not Path(TEST_AUDIO).exists():
        print(f"❌ Test audio file not found: {TEST_AUDIO}")
        return False

    # Prepare request
    with open(TEST_AUDIO, 'rb') as f:
        files = {'file': ('angelo-custode.ogg', f, 'audio/ogg')}
        data = {
            'model': MODEL,
            'language': LANGUAGE,
            'response_format': 'json'
        }

        print("   Sending request...")
        response = requests.post(f"{API_URL}/transcribe", files=files, data=data)

    if response.status_code != 200:
        print(f"❌ Transcription failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

    result = response.json()

    # Validate response
    required_fields = ['text', 'language', 'model', 'processing_time', 'real_time_factor']
    missing_fields = [f for f in required_fields if f not in result]

    if missing_fields:
        print(f"❌ Missing fields in response: {missing_fields}")
        return False

    print(f"✅ Transcription successful!")
    print(f"   Text length: {len(result['text'])} characters")
    print(f"   Language detected: {result['language']}")
    print(f"   Model used: {result['model']}")
    print(f"   Processing time: {result['processing_time']:.2f}s")
    print(f"   Real-time factor: {result['real_time_factor']:.1f}x")
    print(f"   Framework: {result.get('framework', 'N/A')}")

    if result.get('segments'):
        print(f"   Segments: {len(result['segments'])}")

    # Print first 200 chars of transcription
    text_preview = result['text'][:200]
    if len(result['text']) > 200:
        text_preview += "..."
    print(f"\n   Transcription preview:")
    print(f"   \"{text_preview}\"")

    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 Whisper MLX API - Functional Test Suite")
    print("=" * 60)

    tests = [
        ("Health Check", test_health),
        ("Models Endpoint", test_models),
        ("Transcription (Italian)", test_transcription),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
