#!/bin/bash
# Enterprise RAG System - Unified Startup Script
# Usage: ./start.sh [model_name]
# Examples: ./start.sh tinyllama
#          ./start.sh qwen25_3b
#          ./start.sh qwen2_7b

set -e

# Available models
AVAILABLE_MODELS=("tinyllama" "qwen25_3b" "qwen2_7b")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Enterprise RAG System - Unified Startup${NC}"
echo "================================================"

# Function to detect system capabilities
detect_system() {
    echo -e "${BLUE}📋 Detecting system capabilities...${NC}"
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        CPU_CORES=$(nproc)
        TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        CPU_CORES=$(sysctl -n hw.ncpu)
        TOTAL_MEM=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    else
        echo -e "${RED}❌ Unsupported OS: $OSTYPE${NC}"
        exit 1
    fi
    
    # Set environment variables for backend tier detection
    export HOST_RAM_GB=$TOTAL_MEM
    export HOST_CPU_COUNT=$CPU_CORES
    
    echo -e "${GREEN}✅ OS: $OS, Cores: $CPU_CORES, Memory: ${TOTAL_MEM}GB${NC}"
    echo -e "${GREEN}✅ Environment variables set: HOST_RAM_GB=$TOTAL_MEM, HOST_CPU_COUNT=$CPU_CORES${NC}"
}

# Function to handle model selection
handle_model_selection() {
    local MODEL_CHOICE="$1"
    
    # If no model specified, show menu
    if [ -z "$MODEL_CHOICE" ]; then
        echo -e "${BLUE}🤖 Available Models:${NC}"
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
        echo -e "${RED}❌ Invalid model: $MODEL_CHOICE${NC}"
        echo "Available models: ${AVAILABLE_MODELS[*]}"
        exit 1
    fi
    
    echo -e "${BLUE}🔄 Switching to model: $MODEL_CHOICE${NC}"
    
    # Activate virtual environment and switch model
    if [ -d "venv" ]; then
        source venv/bin/activate
        ./switch_model.sh "$MODEL_CHOICE"
    else
        echo -e "${YELLOW}⚠️  Virtual environment not found. Creating one...${NC}"
        python3 -m venv venv
        source venv/bin/activate
        pip install pyyaml
        ./switch_model.sh "$MODEL_CHOICE"
    fi
}

# Function to check dependencies
check_dependencies() {
    echo -e "${BLUE}🔍 Checking dependencies...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose first.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Dependencies check passed${NC}"
}

# Function to parse config and generate environment
generate_env_from_config() {
    echo -e "${BLUE}🔧 Generating environment from config.yaml...${NC}"
    
    # Read model name from config
    MODEL_NAME=$(python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config['model']['name'])
" 2>/dev/null || echo "qwen2.5-3b-instruct-q4_k_m.gguf")
    
    # Read performance settings
    THREADS=$(python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config['performance']['threads'])
" 2>/dev/null || echo "12")
    
    BATCH_SIZE=$(python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config['performance']['batch_size'])
" 2>/dev/null || echo "512")
    
    CTX_SIZE=$(python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config['model']['context_size'])
" 2>/dev/null || echo "4096")
    
    # Generate environment file
    cat > configs/auto_generated.env << EOF
# Auto-generated from config.yaml
MODEL_NAME=$MODEL_NAME
LLAMA_IMAGE=ghcr.io/ggerganov/llama.cpp:server
THREADS=$THREADS
BATCH_SIZE=$BATCH_SIZE
CTX_SIZE=$CTX_SIZE
GPU_LAYERS=0
N_PREDICT=256
REPEAT_PENALTY=1.1
TEMP=0.7
TOP_P=0.9
TOP_K=40

# System information for tier detection
HOST_RAM_GB=$TOTAL_MEM
HOST_CPU_COUNT=$CPU_CORES

# System optimizations
OMP_NUM_THREADS=$THREADS
MKL_NUM_THREADS=$THREADS
OPENBLAS_NUM_THREADS=$THREADS
MALLOC_ARENA_MAX=2
EOF
    
    echo -e "${GREEN}✅ Environment generated: MODEL_NAME=$MODEL_NAME, THREADS=$THREADS${NC}"
}

# Function to check model availability
check_model() {
    local model_name=$1
    echo -e "${BLUE}📥 Checking model: $model_name${NC}"
    
    if [ -f "models/$model_name" ]; then
        echo -e "${GREEN}✅ Model found: $model_name${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  Model not found: $model_name${NC}"
        echo -e "${BLUE}💡 Available models:${NC}"
        ls -la models/ 2>/dev/null | grep -E "\.gguf$" || echo "No models found in models/ directory"
        return 1
    fi
}

# Function to start services
start_services() {
    echo -e "${BLUE}🚀 Starting Enterprise RAG System...${NC}"
    
    # Stop any existing services
    docker-compose down 2>/dev/null || true
    
    # Start services with generated config
    docker-compose --env-file configs/auto_generated.env up -d
    
    echo -e "${GREEN}✅ Services started successfully${NC}"
}

# Function to wait for services and show status
wait_and_status() {
    echo -e "${BLUE}⏳ Waiting for services to be ready...${NC}"
    sleep 30
    
    echo -e "${BLUE}📊 Service Status:${NC}"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo -e "${BLUE}🔗 Service URLs:${NC}"
    echo -e "   • Backend API: http://localhost:8000"
    echo -e "   • LLM Server: http://localhost:8080"
    echo -e "   • Vector DB: http://localhost:6334"
    echo -e "   • Grafana: http://localhost:3000 (admin/admin)"
    echo -e "   • Prometheus: http://localhost:9090"
}

# Function to show configuration summary
show_config_summary() {
    echo -e "${BLUE}📋 Configuration Summary:${NC}"
    
    # Read and display key config values
    MODEL_NAME=$(python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config['model']['name'])
" 2>/dev/null || echo "Unknown")
    
    THREADS=$(python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config['performance']['threads'])
" 2>/dev/null || echo "Unknown")
    
    QUANTIZATION=$(python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config['quantization']['method'].upper())
" 2>/dev/null || echo "Unknown")
    
    LORA_ENABLED=$(python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('Enabled' if config['lora']['enabled'] else 'Disabled')
" 2>/dev/null || echo "Unknown")
    
    MOE_ENABLED=$(python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('Enabled' if config['moe']['enabled'] else 'Disabled')
" 2>/dev/null || echo "Unknown")
    
    echo -e "   • Model: $MODEL_NAME"
    echo -e "   • Threads: $THREADS"
    echo -e "   • Quantization: $QUANTIZATION"
    echo -e "   • LoRA: $LORA_ENABLED"
    echo -e "   • MoE: $MOE_ENABLED"
}

# Main execution
main() {
    detect_system
    check_dependencies
    
    # Handle model selection
    handle_model_selection "$1"
    
    generate_env_from_config
    
    # Check if model exists
    MODEL_NAME=$(python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config['model']['name'])
" 2>/dev/null || echo "qwen2.5-3b-instruct-q4_k_m.gguf")
    
    if check_model "$MODEL_NAME"; then
        start_services
        wait_and_status
        show_config_summary
        echo -e "${GREEN}🎉 Enterprise RAG System is ready!${NC}"
        echo -e "${BLUE}💡 Access your RAG system at: http://localhost:8000${NC}"
    else
        echo -e "${YELLOW}⚠️  Please download the required model and run again.${NC}"
        echo -e "${BLUE}💡 To change model, run: ./start.sh [model_name]${NC}"
    fi
}

# Run main function
main "$@"

