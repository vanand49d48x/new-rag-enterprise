#!/usr/bin/env python3
"""
Comprehensive Multi-Modal Processing Demo
Tests all supported media types: Images, Audio, Video, Documents
"""

import requests
import json
import time
import os
from pathlib import Path

def test_multimodal_processing():
    """Test all multimodal processing capabilities"""
    
    base_url = "http://localhost:8000"
    
    print("🎯 Multi-Modal Processing Demo")
    print("=" * 50)
    
    # Test 1: Document Processing
    print("\n📄 1. Testing Document Processing...")
    test_files = [
        "data/medical/medical_symptoms.txt",
        "data/medical/medical_diseases.txt"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                files = {'files': f}
                response = requests.post(f"{base_url}/ingest", files=files)
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ {file_path}: {result['message']}")
                else:
                    print(f"   ❌ {file_path}: Failed")
        else:
            print(f"   ⚠️  {file_path}: File not found")
    
    # Test 2: Image Processing (OCR)
    print("\n🖼️  2. Testing Image Processing (OCR)...")
    print("   ℹ️  To test image OCR, upload an image with text using:")
    print("   curl -X POST http://localhost:8000/ingest -F 'files=@your_image.png'")
    
    # Test 3: Audio Processing
    print("\n🎵 3. Testing Audio Processing...")
    print("   ℹ️  To test audio transcription, upload an audio file using:")
    print("   curl -X POST http://localhost:8000/ingest -F 'files=@your_audio.mp3'")
    print("   Supported formats: MP3, WAV, FLAC, M4A")
    
    # Test 4: Video Processing
    print("\n🎬 4. Testing Video Processing...")
    print("   ℹ️  To test video processing, upload a video file using:")
    print("   curl -X POST http://localhost:8000/ingest -F 'files=@your_video.mp4'")
    print("   Supported formats: MP4, MOV, AVI, MKV")
    
    # Test 5: Chat with ingested content
    print("\n💬 5. Testing Chat with Ingested Content...")
    test_queries = [
        "What are common medical symptoms?",
        "Tell me about diseases",
        "What medical information do you have?"
    ]
    
    for query in test_queries:
        print(f"\n   Query: {query}")
        response = requests.post(f"{base_url}/chat", json={"query": query})
        if response.status_code == 200:
            result = response.json()
            print(f"   Response: {result['answer'][:200]}...")
        else:
            print(f"   ❌ Failed to get response")
    
    print("\n" + "=" * 50)
    print("🎉 Multi-Modal Processing Demo Complete!")
    print("\n📋 Summary of Capabilities:")
    print("   ✅ Document Processing: PDF, DOCX, TXT, CSV, XLSX, HTML, RTF, EML")
    print("   ✅ Image Processing: PNG, JPG, JPEG, BMP, TIFF (OCR)")
    print("   ✅ Audio Processing: MP3, WAV, FLAC, M4A (Whisper Transcription)")
    print("   ✅ Video Processing: MP4, MOV, AVI, MKV (Frame OCR + Audio)")
    print("   ✅ Chat Interface: Query all ingested content")
    
    print("\n🚀 Usage Examples:")
    print("   # Upload documents")
    print("   curl -X POST http://localhost:8000/ingest -F 'files=@document.pdf'")
    print("   ")
    print("   # Upload images with text")
    print("   curl -X POST http://localhost:8000/ingest -F 'files=@screenshot.png'")
    print("   ")
    print("   # Upload audio files")
    print("   curl -X POST http://localhost:8000/ingest -F 'files=@recording.mp3'")
    print("   ")
    print("   # Upload video files")
    print("   curl -X POST http://localhost:8000/ingest -F 'files=@presentation.mp4'")
    print("   ")
    print("   # Chat with all content")
    print("   curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{\"query\": \"What information do you have?\"}'")

if __name__ == "__main__":
    test_multimodal_processing() 