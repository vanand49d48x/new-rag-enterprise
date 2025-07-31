#!/bin/bash
# Centralized Configuration Generator
# Generates optimal configurations based on system detection and customer overrides

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration file
CONFIG_FILE="configs/auto_generated.yaml"
ENV_FILE="configs/auto_generated.env"

# Function to detect system capabilities
detect_system() {
    echo -e "${BLUE}🔍 Detecting system capabilities...${NC}"
    
    # Get CPU info
    CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo "4")
    CPU_CORES=${CPU_CORES:-4}
    
    # Get RAM info
    TOTAL_RAM=$(sysctl -n hw.memsize 2>/dev/null || grep MemTotal /proc/meminfo | awk '{print $2}' 2>/dev/null || echo "8589934592")
    RAM_GB=$((TOTAL_RAM / 1024 / 1024 / 1024))
    RAM_GB=${RAM_GB:-8}
    
    # Check for GPU
    GPU_AVAILABLE=false
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_AVAILABLE=true
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    fi
    
    echo -e "${GREEN}✅ System detected:${NC}"
    echo "   CPU Cores: $CPU_CORES"
    echo "   RAM: ${RAM_GB}GB"
    echo "   GPU: $([ "$GPU_AVAILABLE" = true ] && echo "Available ($GPU_COUNT GPUs)" || echo "Not available")"
    
    # Determine tier
    if [ $RAM_GB -lt 8 ] || [ $CPU_CORES -le 2 ]; then
        TIER="laptop"
    elif [ $RAM_GB -lt 16 ] || [ $CPU_CORES -le 4 ]; then
        TIER="workstation"
    elif [ $RAM_GB -lt 32 ] || [ $CPU_CORES -le 8 ]; then
        TIER="server"
    else
        TIER="enterprise"
    fi
    
    echo -e "${GREEN}✅ Recommended tier: $TIER${NC}"
}

# Function to get model recommendations
get_model_recommendations() {
    local tier=$1
    local gpu_available=$2
    
    case $tier in
        "laptop")
            if [ "$gpu_available" = true ]; then
                MODEL_NAME="qwen2.5-3b-instruct-q4_k_m.gguf"
                CTX_SIZE=2048
                THREADS=4
                BATCH_SIZE=128
                GPU_LAYERS=20
            else
                MODEL_NAME="tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
                CTX_SIZE=1024
                THREADS=2
                BATCH_SIZE=64
                GPU_LAYERS=0
            fi
            ;;
        "workstation")
            if [ "$gpu_available" = true ]; then
                MODEL_NAME="qwen2.5-3b-instruct-q4_k_m.gguf"
                CTX_SIZE=3072
                THREADS=6
                BATCH_SIZE=256
                GPU_LAYERS=25
            else
                MODEL_NAME="qwen2.5-3b-instruct-q4_k_m.gguf"
                CTX_SIZE=2048
                THREADS=4
                BATCH_SIZE=128
                GPU_LAYERS=0
            fi
            ;;
        "server")
            if [ "$gpu_available" = true ]; then
                MODEL_NAME="qwen2-7b-instruct-q4_k_m.gguf"
                CTX_SIZE=4096
                THREADS=8
                BATCH_SIZE=512
                GPU_LAYERS=35
            else
                MODEL_NAME="qwen2.5-3b-instruct-q4_k_m.gguf"
                CTX_SIZE=3072
                THREADS=6
                BATCH_SIZE=256
                GPU_LAYERS=0
            fi
            ;;
        "enterprise")
            if [ "$gpu_available" = true ]; then
                MODEL_NAME="qwen2-7b-instruct-q4_k_m.gguf"
                CTX_SIZE=6144
                THREADS=12
                BATCH_SIZE=512
                GPU_LAYERS=35
            else
                MODEL_NAME="qwen2.5-3b-instruct-q4_k_m.gguf"
                CTX_SIZE=4096
                THREADS=8
                BATCH_SIZE=256
                GPU_LAYERS=0
            fi
            ;;
    esac
    
    echo -e "${GREEN}✅ Model recommendations for $tier tier:${NC}"
    echo "   Model: $MODEL_NAME"
    echo "   Context Size: $CTX_SIZE"
    echo "   Threads: $THREADS"
    echo "   Batch Size: $BATCH_SIZE"
    echo "   GPU Layers: $GPU_LAYERS"
}

# Function to apply customer overrides
apply_overrides() {
    echo -e "${BLUE}🔧 Applying customer overrides...${NC}"
    
    # Check for override file
    if [ -f "configs/customer_overrides.env" ]; then
        echo -e "${YELLOW}📝 Found customer overrides${NC}"
        source configs/customer_overrides.env
        
        # Apply overrides if provided
        MODEL_NAME=${CUSTOMER_MODEL_NAME:-$MODEL_NAME}
        CTX_SIZE=${CUSTOMER_CTX_SIZE:-$CTX_SIZE}
        THREADS=${CUSTOMER_THREADS:-$THREADS}
        BATCH_SIZE=${CUSTOMER_BATCH_SIZE:-$BATCH_SIZE}
        GPU_LAYERS=${CUSTOMER_GPU_LAYERS:-$GPU_LAYERS}
        
        echo -e "${GREEN}✅ Applied customer overrides:${NC}"
        echo "   Model: $MODEL_NAME"
        echo "   Context Size: $CTX_SIZE"
        echo "   Threads: $THREADS"
        echo "   Batch Size: $BATCH_SIZE"
        echo "   GPU Layers: $GPU_LAYERS"
    else
        echo -e "${YELLOW}📝 No customer overrides found${NC}"
    fi
}

# Function to generate configuration files
generate_configs() {
    echo -e "${BLUE}📝 Generating configuration files...${NC}"
    
    # Create configs directory if it doesn't exist
    mkdir -p configs
    
    # Generate YAML config
    cat > "$CONFIG_FILE" << EOF
# Auto-generated configuration for $TIER tier
# Generated on: $(date)
# System: ${CPU_CORES} cores, ${RAM_GB}GB RAM, GPU: $([ "$GPU_AVAILABLE" = true ] && echo "Available" || echo "Not available")

llm:
  model: $MODEL_NAME
  backend: llama_cpp
  api_url: http://llama-cpp:8080/completion
  timeout: 60
  max_tokens: 256
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  repeat_penalty: 1.1
  context_size: $CTX_SIZE
  batch_size: $BATCH_SIZE
  threads: $THREADS
  gpu_layers: $GPU_LAYERS
  stream: true
  flash_attention: $([ "$GPU_AVAILABLE" = true ] && echo "true" || echo "false")

processing:
  chunk_size: 512
  chunk_overlap: 128
  max_workers: $((CPU_CORES / 2))
  embedding_batch_size: 32
  prompt_trimming: true
  max_prompt_tokens: 500
  max_chunks: 3

api:
  timeout: 60
  stream_responses: true
  enable_flash_attention: $([ "$GPU_AVAILABLE" = true ] && echo "true" || echo "false")

vector_store:
  max_results: 5
  max_chunks_per_query: 3
  similarity_threshold: 0.7

system:
  tier: $TIER
  cpu_cores: $CPU_CORES
  ram_gb: $RAM_GB
  gpu_available: $GPU_AVAILABLE
EOF

    # Generate environment file
    cat > "$ENV_FILE" << EOF
# Auto-generated environment configuration
# Generated on: $(date)

# Model configuration
MODEL_NAME=$MODEL_NAME
LLAMA_IMAGE=$([ "$GPU_AVAILABLE" = true ] && echo "ghcr.io/ggerganov/llama.cpp:server-cuda" || echo "ghcr.io/ggerganov/llama.cpp:server")

# Performance configuration
CTX_SIZE=$CTX_SIZE
THREADS=$THREADS
BATCH_SIZE=$BATCH_SIZE
GPU_LAYERS=$GPU_LAYERS

# GPU configuration (if available)
$([ "$GPU_AVAILABLE" = true ] && echo "GPU_COUNT=$GPU_COUNT" || echo "# GPU_COUNT=0")
$([ "$GPU_AVAILABLE" = true ] && echo "CUDA_VISIBLE_DEVICES=0" || echo "# CUDA_VISIBLE_DEVICES=")
$([ "$GPU_AVAILABLE" = true ] && echo "NVIDIA_VISIBLE_DEVICES=all" || echo "# NVIDIA_VISIBLE_DEVICES=")

# System information
HOST_RAM_GB=$RAM_GB
HOST_CPU_COUNT=$CPU_CORES
SYSTEM_TIER=$TIER
GPU_AVAILABLE=$GPU_AVAILABLE

# Quality settings
N_PREDICT=128
REPEAT_PENALTY=1.1
TEMP=0.7
TOP_P=0.9
TOP_K=40
EOF

    echo -e "${GREEN}✅ Generated configuration files:${NC}"
    echo "   YAML: $CONFIG_FILE"
    echo "   ENV: $ENV_FILE"
}

# Function to validate configuration
validate_config() {
    echo -e "${BLUE}🔍 Validating configuration...${NC}"
    
    # Check if model exists
    if [ ! -f "models/$MODEL_NAME" ]; then
        echo -e "${YELLOW}⚠️  Model $MODEL_NAME not found. Will be downloaded during startup.${NC}"
    else
        echo -e "${GREEN}✅ Model $MODEL_NAME found${NC}"
    fi
    
    # Validate parameters
    if [ $CTX_SIZE -lt 512 ] || [ $CTX_SIZE -gt 8192 ]; then
        echo -e "${RED}❌ Invalid context size: $CTX_SIZE${NC}"
        exit 1
    fi
    
    if [ $THREADS -lt 1 ] || [ $THREADS -gt 32 ]; then
        echo -e "${RED}❌ Invalid thread count: $THREADS${NC}"
        exit 1
    fi
    
    if [ $BATCH_SIZE -lt 32 ] || [ $BATCH_SIZE -gt 1024 ]; then
        echo -e "${RED}❌ Invalid batch size: $BATCH_SIZE${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Configuration validation passed${NC}"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --validate, -v      Only validate existing configuration"
    echo "  --force, -f         Force regeneration of configs"
    echo "  --tier TIER         Override tier detection (laptop|workstation|server|enterprise)"
    echo ""
    echo "Examples:"
    echo "  $0                  # Auto-detect and generate configs"
    echo "  $0 --tier laptop    # Force laptop tier configuration"
    echo "  $0 --validate       # Only validate existing configs"
    echo ""
    echo "Customer Overrides:"
    echo "  Create configs/customer_overrides.env with:"
    echo "    CUSTOMER_MODEL_NAME=your-model.gguf"
    echo "    CUSTOMER_CTX_SIZE=2048"
    echo "    CUSTOMER_THREADS=4"
    echo "    CUSTOMER_BATCH_SIZE=128"
    echo "    CUSTOMER_GPU_LAYERS=0"
}

# Main execution
main() {
    echo -e "${BLUE}🚀 Enterprise RAG Configuration Generator${NC}"
    echo "================================================"
    
    # Parse arguments
    VALIDATE_ONLY=false
    FORCE=false
    OVERRIDE_TIER=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_usage
                exit 0
                ;;
            --validate|-v)
                VALIDATE_ONLY=true
                shift
                ;;
            --force|-f)
                FORCE=true
                shift
                ;;
            --tier)
                OVERRIDE_TIER="$2"
                shift 2
                ;;
            *)
                echo -e "${RED}❌ Unknown option: $1${NC}"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Detect system
    detect_system
    
    # Override tier if specified
    if [ -n "$OVERRIDE_TIER" ]; then
        TIER="$OVERRIDE_TIER"
        echo -e "${YELLOW}📝 Using override tier: $TIER${NC}"
    fi
    
    # Get model recommendations
    get_model_recommendations "$TIER" "$GPU_AVAILABLE"
    
    # Apply customer overrides
    apply_overrides
    
    # Validate configuration
    validate_config
    
    # Generate configs if not validate-only
    if [ "$VALIDATE_ONLY" = false ]; then
        if [ "$FORCE" = true ] || [ ! -f "$CONFIG_FILE" ]; then
            generate_configs
        else
            echo -e "${YELLOW}📝 Configuration files already exist. Use --force to regenerate.${NC}"
        fi
    fi
    
    echo ""
    echo -e "${GREEN}🎉 Configuration generation complete!${NC}"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Review generated configs:"
    echo "      cat $CONFIG_FILE"
    echo "      cat $ENV_FILE"
    echo ""
    echo "   2. Start the system:"
    echo "      ./start.sh"
    echo ""
    echo "   3. Customize (optional):"
    echo "      cp $ENV_FILE configs/customer_overrides.env"
    echo "      # Edit configs/customer_overrides.env"
    echo "      ./scripts/generate_config.sh"
}

# Run main function
main "$@" 