#!/bin/bash

# Model Quantization Script
# Automatically downloads and quantizes models to Q4_K_M format for memory efficiency

echo "🔧 Model Quantization System"
echo "============================"

# Function to check if llama.cpp is available
check_llama_cpp() {
    if ! command -v llama-cpp &> /dev/null; then
        echo "❌ llama-cpp not found. Installing..."
        # Install llama-cpp-python with CUDA support if available
        pip install llama-cpp-python --force-reinstall --index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu118
    else
        echo "✅ llama-cpp found"
    fi
}

# Function to download and quantize a model
quantize_model() {
    local model_name=$1
    local model_url=$2
    local output_name=$3
    
    echo ""
    echo "🔧 Quantizing $model_name to Q4_K_M..."
    
    # Create models directory if it doesn't exist
    mkdir -p models
    
    # Check if model already exists
    if [ -f "models/$output_name" ]; then
        echo "✅ Model already exists: models/$output_name"
        ls -lh "models/$output_name"
        return 0
    fi
    
    # Download model if not present
    if [ ! -f "models/${model_name}.gguf" ]; then
        echo "📥 Downloading $model_name..."
        wget -O "models/${model_name}.gguf" "$model_url"
    fi
    
    # Quantize to Q4_K_M
    echo "🔧 Quantizing to Q4_K_M format..."
    llama-cpp --model "models/${model_name}.gguf" \
              --quantize "models/$output_name" \
              --quantize-type Q4_K_M \
              --threads $(nproc)
    
    if [ $? -eq 0 ]; then
        echo "✅ Quantization successful!"
        ls -lh "models/$output_name"
        
        # Remove original file to save space
        rm "models/${model_name}.gguf"
        echo "🗑️ Removed original file to save space"
    else
        echo "❌ Quantization failed"
        return 1
    fi
}

# Function to show quantization benefits
show_quantization_benefits() {
    echo ""
    echo "📊 Quantization Benefits (Q4_K_M):"
    echo "=================================="
    echo "  • Memory Usage: ~75% reduction"
    echo "  • Inference Speed: ~20-30% faster"
    echo "  • Quality Loss: Minimal (<5%)"
    echo "  • Compatibility: Works with llama-cpp"
    echo "  • File Size: ~25% of original"
    echo ""
}

# Function to quantize all supported models
quantize_all_models() {
    echo ""
    echo "🎯 Quantizing All Supported Models..."
    echo "===================================="
    
    # Model definitions with URLs and output names
    declare -A models=(
        ["qwen2-7b"]="https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct.gguf"
        ["qwen2.5-3b"]="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct.gguf"
        ["tinyllama"]="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    )
    
    declare -A output_names=(
        ["qwen2-7b"]="qwen2-7b-instruct-q4_k_m.gguf"
        ["qwen2.5-3b"]="qwen2.5-3b-instruct-q4_k_m.gguf"
        ["tinyllama"]="tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
    )
    
    local success_count=0
    local total_count=0
    
    for model in "${!models[@]}"; do
        total_count=$((total_count + 1))
        if quantize_model "$model" "${models[$model]}" "${output_names[$model]}"; then
            success_count=$((success_count + 1))
        fi
    done
    
    echo ""
    echo "📊 Quantization Summary:"
    echo "========================"
    echo "  ✅ Successful: $success_count/$total_count"
    echo "  ❌ Failed: $((total_count - success_count))/$total_count"
    
    if [ $success_count -eq $total_count ]; then
        echo "🎉 All models quantized successfully!"
    else
        echo "⚠️ Some models failed to quantize"
    fi
}

# Function to check current model sizes
check_model_sizes() {
    echo ""
    echo "📏 Current Model Sizes:"
    echo "======================="
    
    if [ -d "models" ]; then
        for model in models/*.gguf; do
            if [ -f "$model" ]; then
                size=$(ls -lh "$model" | awk '{print $5}')
                echo "  $(basename "$model"): $size"
            fi
        done
    else
        echo "  No models directory found"
    fi
}

# Function to estimate memory usage
estimate_memory_usage() {
    echo ""
    echo "💾 Memory Usage Estimation:"
    echo "=========================="
    echo "  Qwen2-7B Q4_K_M: ~4-5GB RAM"
    echo "  Qwen2.5-3B Q4_K_M: ~2-3GB RAM"
    echo "  TinyLlama Q4_K_M: ~1-2GB RAM"
    echo ""
    echo "  Recommended minimum RAM: 8GB"
    echo "  Recommended for production: 16GB+"
}

# Function to show usage
show_usage() {
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  all              - Quantize all supported models"
    echo "  qwen2-7b         - Quantize Qwen2-7B model"
    echo "  qwen2.5-3b       - Quantize Qwen2.5-3B model"
    echo "  tinyllama        - Quantize TinyLlama model"
    echo "  check            - Check current model sizes"
    echo "  memory           - Show memory usage estimates"
    echo "  benefits         - Show quantization benefits"
    echo "  help             - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 all                    # Quantize all models"
    echo "  $0 qwen2.5-3b            # Quantize specific model"
    echo "  $0 check                  # Check current sizes"
    echo ""
}

# Main script logic
case "${1:-help}" in
    "all")
        check_llama_cpp
        quantize_all_models
        ;;
    "qwen2-7b")
        check_llama_cpp
        quantize_model "qwen2-7b" \
            "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct.gguf" \
            "qwen2-7b-instruct-q4_k_m.gguf"
        ;;
    "qwen2.5-3b")
        check_llama_cpp
        quantize_model "qwen2.5-3b" \
            "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct.gguf" \
            "qwen2.5-3b-instruct-q4_k_m.gguf"
        ;;
    "tinyllama")
        check_llama_cpp
        quantize_model "tinyllama" \
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
            "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
        ;;
    "check")
        check_model_sizes
        ;;
    "memory")
        estimate_memory_usage
        ;;
    "benefits")
        show_quantization_benefits
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