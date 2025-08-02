#!/bin/bash

# Enhanced Upload System Startup Script
# This script starts the complete enhanced upload and processing system

set -e

echo "🚀 Starting Enhanced Upload System"
echo "=================================="

# Function to detect system capabilities
detect_system() {
    echo "🔍 Detecting system capabilities..."
    
    # Detect OS and get system info
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        CPU_CORES=$(nproc)
        TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        CPU_CORES=$(sysctl -n hw.ncpu)
        TOTAL_MEM=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    else
        echo "❌ Unsupported OS: $OSTYPE"
        exit 1
    fi
    
    # Set environment variables for backend tier detection
    export HOST_RAM_GB=$TOTAL_MEM
    export HOST_CPU_COUNT=$CPU_CORES
    
    echo "✅ OS: $OS, Cores: $CPU_CORES, Memory: ${TOTAL_MEM}GB"
    echo "✅ Environment variables set: HOST_RAM_GB=$TOTAL_MEM, HOST_CPU_COUNT=$CPU_CORES"
}

# Available models
AVAILABLE_MODELS=("tinyllama" "qwen25_3b" "qwen2_7b")

# Function to handle model selection
handle_model_selection() {
    local MODEL_CHOICE="$1"
    
    # If no model specified, show menu
    if [ -z "$MODEL_CHOICE" ]; then
        echo "🤖 Available Models:"
        echo "1) tinyllama   - TinyLlama-1.1B (Fastest)"
        echo "2) qwen25_3b   - Qwen2.5-3B (Balanced)"
        echo "3) qwen2_7b    - Qwen2-7B (Highest Quality)"
        echo ""
        read -p "Enter model name (or number 1-3): " MODEL_CHOICE
        
        # Convert number to model name
        case $MODEL_CHOICE in
            1) MODEL_CHOICE="tinyllama" ;;
            2) MODEL_CHOICE="qwen25_3b" ;;
            3) MODEL_CHOICE="qwen2_7b" ;;
        esac
    fi
    
    # Validate model
    if [[ ! " ${AVAILABLE_MODELS[@]} " =~ " ${MODEL_CHOICE} " ]]; then
        echo "❌ Invalid model: $MODEL_CHOICE"
        echo "Available models: ${AVAILABLE_MODELS[*]}"
        exit 1
    fi
    
    echo "🔄 Switching to model: $MODEL_CHOICE"
    
    # Activate virtual environment and switch model
    if [ -d "venv" ]; then
        source venv/bin/activate
        ./switch_model.sh "$MODEL_CHOICE"
    else
        echo "⚠️  Virtual environment not found. Creating one..."
        python3 -m venv venv
        source venv/bin/activate
        pip install pyyaml
        ./switch_model.sh "$MODEL_CHOICE"
    fi
}

# Detect system capabilities first
detect_system

# Handle model selection
handle_model_selection "$1"

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

# Function to check if a port is in use
check_port() {
    local port=$1
    local service=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port $port is already in use. $service might already be running."
        return 1
    else
        return 0
    fi
}

# Check for port conflicts
echo "🔍 Checking for port conflicts..."
check_port 8000 "Backend API" || echo "   Backend API port 8000 is in use"
check_port 1080 "TUS Upload Service" || echo "   TUS Upload Service port 1080 is in use"
check_port 6333 "Qdrant Vector DB" || echo "   Qdrant Vector DB port 6333 is in use"
check_port 8080 "LLaMA.cpp Server" || echo "   LLaMA.cpp Server port 8080 is in use"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads
mkdir -p processed
mkdir -p logs

# Set environment variables
export UPLOAD_DIR=$(pwd)/uploads
export PROCESSED_DIR=$(pwd)/processed
export QDRANT_HOST=localhost
export QDRANT_PORT=6334

echo "🔧 Environment variables set:"
echo "   UPLOAD_DIR: $UPLOAD_DIR"
echo "   PROCESSED_DIR: $PROCESSED_DIR"
echo "   QDRANT_HOST: $QDRANT_HOST"
echo "   QDRANT_PORT: $QDRANT_PORT"

# Start the services
echo "🚀 Starting services with docker-compose..."

# Use the enhanced docker-compose configuration
if [ -f "docker-compose.yml" ]; then
    echo "📋 Using docker-compose.yml configuration"
    docker-compose up -d
elif [ -f "docker-compose.gpu.yml" ]; then
    echo "📋 Using GPU-optimized docker-compose.gpu.yml configuration"
    docker-compose -f docker-compose.gpu.yml up -d
else
    echo "❌ No docker-compose configuration found!"
    exit 1
fi

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🏥 Checking service health..."

# Function to check service health
check_service() {
    local url=$1
    local service=$2
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo "✅ $service is healthy"
            return 0
        else
            echo "⏳ Waiting for $service... (attempt $attempt/$max_attempts)"
            sleep 2
            attempt=$((attempt + 1))
        fi
    done
    
    echo "❌ $service failed to start after $max_attempts attempts"
    return 1
}

# Check each service
check_service "http://localhost:8000/" "Backend API" || echo "   Backend API health check failed"
check_service "http://localhost:1080/" "TUS Upload Service" || echo "   TUS Upload Service health check failed"
check_service "http://localhost:6334/health" "Qdrant Vector DB" || echo "   Qdrant Vector DB health check failed"
check_service "http://localhost:8080/completion" "LLaMA.cpp Server" || echo "   LLaMA.cpp Server health check failed"

# Show service status
echo "📊 Service Status:"
docker-compose ps

# Show logs for debugging
echo "📋 Recent logs:"
docker-compose logs --tail=20

echo ""
echo "🎉 Enhanced Upload System is ready!"
echo ""
echo "📱 Access Points:"
echo "   • Local RAG Dashboard: http://localhost:8000/dashboard"
echo "   • Enhanced Upload Interface: http://localhost:8000/enhanced-upload"
echo "   • API Documentation: http://localhost:8000/docs"
echo "   • Qdrant Dashboard: http://localhost:6333/dashboard"
echo "   • Prometheus Metrics: http://localhost:9090"
echo "   • Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo ""
echo "📁 Upload Directories:"
echo "   • Uploads: $UPLOAD_DIR"
echo "   • Processed: $PROCESSED_DIR"
echo ""
echo "🔧 Management Commands:"
echo "   • View logs: docker-compose logs -f"
echo "   • Stop services: docker-compose down"
echo "   • Restart services: docker-compose restart"
echo "   • Test system: python test_enhanced_upload.py"
echo "   • Test dashboard: python test_dashboard.py"
echo "   • Test deletion: python test_deletion.py"
echo ""
echo "💡 Tips:"
echo "   • Use the Local RAG Dashboard for comprehensive file management"
echo "   • Upload files via the web interface at http://localhost:8000/dashboard"
echo "   • Monitor processing status in real-time"
echo "   • Delete files with complete data removal (privacy-first)"
echo "   • Check the logs for any processing errors"
echo "   • Use the API endpoints for programmatic access"
echo ""

# Optional: Run the test script
if command -v python3 &> /dev/null; then
    echo "🧪 Running system test..."
    if python3 test_enhanced_upload.py; then
        echo "✅ System test passed!"
    else
        echo "⚠️  System test failed. Check the logs for details."
    fi
else
    echo "⚠️  Python3 not found. Skipping system test."
fi

echo ""
echo "🚀 Enhanced Upload System startup complete!" 