#!/bin/bash
# deploy.sh - Architecture-aware deployment script

set -e

echo "🚀 Enterprise RAG System - Architecture-Aware Deployment"

# Detect architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# Set platform based on architecture
if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    PLATFORM="linux/arm64"
    echo "Using ARM64 platform"
elif [ "$ARCH" = "x86_64" ]; then
    PLATFORM="linux/amd64"
    echo "Using AMD64 platform"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# Export platform for docker-compose
export DOCKER_DEFAULT_PLATFORM=$PLATFORM

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install it first."
    exit 1
fi

# Function to check if model exists
check_model() {
    if [ ! -f "models/qwen1_5-7b-chat-q4_K_M.gguf" ]; then
        echo "📥 Qwen model not found. Downloading..."
        ./init_model.sh
    else
        echo "✅ Qwen model found."
    fi
}

# Check model availability
check_model

# Start services with platform specification
echo "🔧 Starting Docker services with platform: $PLATFORM"
docker-compose up -d --build

echo "⏳ Waiting for services to start..."

# Function to wait for service health
wait_for_service() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo "✅ $service is ready!"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts - waiting 10 seconds..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service failed to start within expected time."
    return 1
}

# Wait for services to be ready
if wait_for_service "Qdrant" "http://localhost:6333/health"; then
    echo "✅ Qdrant is ready"
else
    echo "❌ Qdrant failed to start"
    docker-compose logs qdrant
    exit 1
fi

if wait_for_service "Llama-cpp" "http://localhost:8080/health"; then
    echo "✅ Llama-cpp is ready"
else
    echo "❌ Llama-cpp failed to start"
    docker-compose logs llama-cpp
    exit 1
fi

if wait_for_service "Backend API" "http://localhost:8000/"; then
    echo "✅ Backend API is ready"
else
    echo "❌ Backend API failed to start"
    docker-compose logs backend
    exit 1
fi

echo ""
echo "🎉 Enterprise RAG System is ready!"
echo ""
echo "📊 Service URLs:"
echo "   API: http://localhost:8000"
echo "   Qdrant Dashboard: http://localhost:6333"
echo "   Prometheus: http://localhost:9090"
echo "   Grafana: http://localhost:3000 (admin/admin)"
echo ""
echo "🔧 Architecture: $PLATFORM"
echo ""
echo "📚 Quick Start:"
echo "   1. Upload documents: curl -X POST http://localhost:8000/ingest -F 'files=@your_document.pdf'"
echo "   2. Ask questions: curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{\"query\": \"Your question here\"}'"
echo ""
echo "🔧 Management:"
echo "   View logs: docker-compose logs -f"
echo "   Stop services: docker-compose down"
echo "   Restart: docker-compose restart"
echo ""
