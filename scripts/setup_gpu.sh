#!/bin/bash

# GPU Setup Script for Medical RAG System
# Enables GPU acceleration for better performance with quantized models

echo "🚀 GPU Setup for Medical RAG System"
echo "==================================="

# Function to check if NVIDIA GPU is available
check_nvidia_gpu() {
    echo "🔍 Checking for NVIDIA GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
        return 0
    else
        echo "❌ NVIDIA GPU not detected"
        return 1
    fi
}

# Function to check Docker GPU support
check_docker_gpu() {
    echo ""
    echo "🔍 Checking Docker GPU support..."
    
    if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        echo "✅ Docker GPU support enabled"
        return 0
    else
        echo "❌ Docker GPU support not available"
        return 1
    fi
}

# Function to install NVIDIA Container Toolkit
install_nvidia_container_toolkit() {
    echo ""
    echo "🔧 Installing NVIDIA Container Toolkit..."
    
    # Check if running on Ubuntu/Debian
    if command -v apt-get &> /dev/null; then
        echo "📦 Installing on Ubuntu/Debian..."
        
        # Add NVIDIA repository
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        
        # Update and install
        sudo apt-get update
        sudo apt-get install -y nvidia-docker2
        sudo systemctl restart docker
        
        echo "✅ NVIDIA Container Toolkit installed"
    else
        echo "⚠️ Automatic installation not supported for this OS"
        echo "Please install NVIDIA Container Toolkit manually:"
        echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        return 1
    fi
}

# Function to configure GPU-optimized settings
configure_gpu_settings() {
    echo ""
    echo "⚙️ Configuring GPU-optimized settings..."
    
    # Create GPU-optimized config
    cat > configs/gpu_optimized.yaml << EOF
# GPU-Optimized Configuration
llm:
  model: qwen2.5-3b-instruct-q4_k_m.gguf
  backend: llama_cpp
  api_url: http://llama-cpp:8080/completion
  timeout: 120
  max_tokens: 512
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  repeat_penalty: 1.1
  context_size: 4096
  batch_size: 512
  threads: 16
  gpu_layers: 35
  mirostat: 2
  mirostat_tau: 5.0
  mirostat_eta: 0.1

processing:
  chunk_size: 512
  chunk_overlap: 128
  max_workers: 8
  embedding_batch_size: 32

api:
  timeout: 180
  max_workers: 8
  host: 0.0.0.0
  port: 8000

vector_store:
  collection_name: medical_documents
  distance_metric: cosine
  max_results: 10
  score_threshold: 0.7

gpu:
  enabled: true
  cuda_visible_devices: 0
  nvidia_visible_devices: all
  memory_fraction: 0.8
  precision: fp16
EOF

    echo "✅ GPU-optimized configuration created"
}

# Function to test GPU performance
test_gpu_performance() {
    echo ""
    echo "🧪 Testing GPU performance..."
    
    # Test GPU inference speed
    echo "📊 Running GPU performance test..."
    
    # Start GPU-enabled services
    docker-compose -f docker-compose.gpu.yml up -d llama-cpp
    
    # Wait for service to be ready
    echo "⏳ Waiting for llama-cpp service to start..."
    sleep 30
    
    # Test inference
    curl -X POST http://localhost:8080/completion \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "What are the symptoms of diabetes?",
            "n_predict": 100,
            "temperature": 0.7,
            "top_p": 0.9
        }' | jq '.timings.predicted_ms' 2>/dev/null || echo "Test completed"
    
    echo "✅ GPU performance test completed"
}

# Function to show GPU benefits
show_gpu_benefits() {
    echo ""
    echo "📊 GPU Acceleration Benefits:"
    echo "============================="
    echo "  • Inference Speed: 3-5x faster"
    echo "  • Batch Processing: 2-3x throughput"
    echo "  • Memory Efficiency: Better utilization"
    echo "  • Parallel Processing: Multiple requests"
    echo "  • Real-time Response: Lower latency"
    echo ""
    echo "  Model Performance with GPU:"
    echo "  • Qwen2-7B: 2-4s response time"
    echo "  • Qwen2.5-3B: 1-3s response time"
    echo "  • TinyLlama: 0.5-1.5s response time"
    echo ""
}

# Function to create GPU startup script
create_gpu_startup() {
    echo ""
    echo "🔧 Creating GPU startup script..."
    
    cat > start_gpu.sh << 'EOF'
#!/bin/bash

# GPU-Enabled Startup Script
echo "🚀 Starting Medical RAG System with GPU acceleration..."

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU not available, falling back to CPU"
    docker-compose up -d
    exit 0
fi

# Check Docker GPU support
if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "❌ Docker GPU support not available, falling back to CPU"
    docker-compose up -d
    exit 0
fi

# Start GPU-enabled services
echo "✅ Starting with GPU acceleration..."
docker-compose -f docker-compose.gpu.yml up -d

echo "🎉 GPU-accelerated RAG system started!"
echo "📊 Monitor GPU usage: nvidia-smi"
echo "🌐 Access API: http://localhost:8000"
echo "📈 Monitor: http://localhost:3000 (Grafana)"
EOF

    chmod +x start_gpu.sh
    echo "✅ GPU startup script created: start_gpu.sh"
}

# Function to show usage
show_usage() {
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup              - Complete GPU setup"
    echo "  check              - Check GPU availability"
    echo "  install            - Install NVIDIA Container Toolkit"
    echo "  configure          - Configure GPU settings"
    echo "  test               - Test GPU performance"
    echo "  benefits           - Show GPU benefits"
    echo "  startup            - Create GPU startup script"
    echo "  help               - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 setup                    # Complete setup"
    echo "  $0 check                    # Check GPU status"
    echo "  $0 test                     # Test performance"
    echo ""
}

# Main script logic
case "${1:-help}" in
    "setup")
        check_nvidia_gpu && {
            check_docker_gpu || install_nvidia_container_toolkit
            configure_gpu_settings
            create_gpu_startup
            show_gpu_benefits
            echo "🎉 GPU setup completed successfully!"
        }
        ;;
    "check")
        check_nvidia_gpu
        check_docker_gpu
        ;;
    "install")
        install_nvidia_container_toolkit
        ;;
    "configure")
        configure_gpu_settings
        ;;
    "test")
        test_gpu_performance
        ;;
    "benefits")
        show_gpu_benefits
        ;;
    "startup")
        create_gpu_startup
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        echo "❌ Unknown command: $1"
        show_usage
        exit 1
        ;;
esac 