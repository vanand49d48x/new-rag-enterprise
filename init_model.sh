#!/bin/bash

# Enterprise RAG System - Model Download Script
# Supports multiple models for different use cases

# Configuration
MODEL_CONFIG=${MODEL_CONFIG:-"qwen2.5-3b"}  # Default model

case $MODEL_CONFIG in
    "qwen2.5-3b")
        MODEL_NAME="qwen2.5-3b-instruct-q4_k_m.gguf"
        MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/$MODEL_NAME"
        ;;
    "qwen2-7b")
        MODEL_NAME="qwen2-7b-instruct-q4_k_m.gguf"
        MODEL_URL="https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/$MODEL_NAME"
        ;;
    "tinyllama")
        MODEL_NAME="tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
        MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/$MODEL_NAME"
        ;;
    "llama3.1-3b")
        MODEL_NAME="llama-3.1-3b-instruct-q4_k_m.gguf"
        MODEL_URL="https://huggingface.co/TheBloke/Llama-3.1-3B-Instruct-GGUF/resolve/main/$MODEL_NAME"
        ;;
    "phi3.5-3.8b")
        MODEL_NAME="phi-3.5-3.8b-instruct-q4_k_m.gguf"
        MODEL_URL="https://huggingface.co/TheBloke/Phi-3.5-3.8B-Instruct-GGUF/resolve/main/$MODEL_NAME"
        ;;
    *)
        echo "❌ Unknown model config: $MODEL_CONFIG"
        echo "Available models: qwen2.5-3b, qwen2-7b, tinyllama, llama3.1-3b, phi3.5-3.8b"
        exit 1
        ;;
esac

MODEL_PATH="models/$MODEL_NAME"

echo "🚀 Enterprise RAG System - Model Download"
echo "📦 Model: $MODEL_CONFIG ($MODEL_NAME)"
echo "🔗 URL: $MODEL_URL"

# Create models directory if it doesn't exist
mkdir -p models

# Check if model already exists
if [ -f "$MODEL_PATH" ]; then
    echo "✅ Model already exists: $MODEL_PATH"
    echo "📊 Size: $(du -h "$MODEL_PATH" | cut -f1)"
    exit 0
fi

echo "⏳ Downloading model..."
echo "📥 This may take several minutes depending on your internet connection..."

# Download with progress
curl -L -o "$MODEL_PATH" "$MODEL_URL"

# Check if download was successful
if [ $? -eq 0 ] && [ -f "$MODEL_PATH" ]; then
    echo "✅ Model downloaded successfully!"
    echo "📊 Size: $(du -h "$MODEL_PATH" | cut -f1)"
    echo "📍 Location: $MODEL_PATH"
else
    echo "❌ Failed to download model"
    exit 1
fi
