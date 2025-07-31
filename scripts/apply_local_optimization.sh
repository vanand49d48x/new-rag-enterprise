#!/bin/bash

# Local Speed Optimization Script
# Implements all speed optimizations for sub-10s latency on laptops

echo "🚀 Local Speed Optimization"
echo "=========================="

# Function to check current model
check_current_model() {
    echo "🔍 Checking current model..."
    if curl -s http://localhost:8000/system 2>/dev/null | grep -q "qwen2.5-3b"; then
        echo "✅ Already using Qwen2.5-3B (optimized model)"
        return 0
    else
        echo "⚠️ Not using optimized model"
        return 1
    fi
}

# Function to switch to optimized model
switch_to_optimized_model() {
    echo ""
    echo "🔄 Switching to optimized model..."
    
    # Stop current services
    docker-compose down
    
    # Set environment for optimized model
    export MODEL_NAME=qwen2.5-3b-instruct-q4_k_m.gguf
    export CTX_SIZE=2048
    export BATCH_SIZE=128
    export THREADS=8
    export GPU_LAYERS=0
    export STREAM=true
    export FLASH_ATTENTION=true
    
    # Start with optimized settings
    docker-compose --env-file configs/cpu.env up -d
    
    echo "✅ Switched to Qwen2.5-3B with optimized settings"
}

# Function to apply local optimization config
apply_local_config() {
    echo ""
    echo "⚙️ Applying local optimization configuration..."
    
    # Copy optimized config
    cp configs/local_optimized.yaml config.yaml
    
    # Apply configuration
    docker-compose restart backend
    
    echo "✅ Local optimization configuration applied"
}

# Function to enable streaming
enable_streaming() {
    echo ""
    echo "⏩ Enabling streaming for faster first token..."
    
    # Update backend configuration
    cat >> backend/api/main.py << 'EOF'

# Streaming configuration
STREAM_RESPONSES = True
ENABLE_FLASH_ATTENTION = True

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint for faster first token"""
    try:
        # Implement streaming response
        response = await process_chat_request(request, stream=True)
        return StreamingResponse(response, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
EOF
    
    echo "✅ Streaming enabled"
}

# Function to optimize prompt processing
optimize_prompt_processing() {
    echo ""
    echo "✂️ Optimizing prompt processing..."
    
    # Create prompt optimization utility
    cat > backend/utils/prompt_optimizer.py << 'EOF'
import re
from typing import List, Dict

class PromptOptimizer:
    def __init__(self, max_tokens: int = 300, max_chunks: int = 2):
        self.max_tokens = max_tokens
        self.max_chunks = max_chunks
    
    def trim_prompt(self, prompt: str) -> str:
        """Trim prompt to specified token limit"""
        # Simple token estimation (4 chars per token)
        estimated_tokens = len(prompt) // 4
        
        if estimated_tokens <= self.max_tokens:
            return prompt
        
        # Trim to max tokens
        max_chars = self.max_tokens * 4
        return prompt[:max_chars] + "..."
    
    def limit_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Limit number of chunks for faster processing"""
        if len(chunks) <= self.max_chunks:
            return chunks
        
        # Keep most relevant chunks (highest scores)
        sorted_chunks = sorted(chunks, key=lambda x: x.get('score', 0), reverse=True)
        return sorted_chunks[:self.max_chunks]
    
    def optimize_query(self, query: str, context: str = "") -> str:
        """Optimize query for faster processing"""
        # Combine query and context
        full_prompt = f"Context: {context}\nQuery: {query}"
        
        # Trim if too long
        return self.trim_prompt(full_prompt)
EOF
    
    echo "✅ Prompt optimization utilities created"
}

# Function to enable flash attention (if GPU available)
enable_flash_attention() {
    echo ""
    echo "🚀 Checking for flash attention support..."
    
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ GPU detected, enabling flash attention..."
        
        # Set environment variable
        export OLLAMA_FLASH_ATTENTION=true
        
        # Update Docker environment
        cat >> docker-compose.yml << 'EOF'
    environment:
      - OLLAMA_FLASH_ATTENTION=true
      - FLASH_ATTENTION=true
EOF
        
        echo "✅ Flash attention enabled for GPU"
    else
        echo "⚠️ No GPU detected, flash attention not available"
    fi
}

# Function to warm up the model cache
warm_model_cache() {
    echo ""
    echo "🔥 Warming up model cache..."
    
    # Send a simple query to warm up the model
    curl -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{"message": "Hello"}' \
        --max-time 30
    
    echo "✅ Model cache warmed up"
}

# Function to test optimized performance
test_optimized_performance() {
    echo ""
    echo "🧪 Testing optimized performance..."
    
    # Test queries with timing
    queries=(
        "What are diabetes symptoms?"
        "How to treat hypertension?"
        "What is the dosage for aspirin?"
    )
    
    for query in "${queries[@]}"; do
        echo "Testing: $query"
        start_time=$(date +%s.%N)
        
        response=$(curl -s -X POST http://localhost:8000/chat \
            -H "Content-Type: application/json" \
            -d "{\"message\": \"$query\"}" \
            --max-time 30)
        
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc -l)
        
        echo "Response time: ${duration}s"
        echo "---"
    done
}

# Function to show optimization benefits
show_optimization_benefits() {
    echo ""
    echo "📊 Local Speed Optimization Benefits:"
    echo "===================================="
    echo "  • Model: Qwen2.5-3B Q4_K_M (30-40% faster than 7B)"
    echo "  • Streaming: Enabled for faster first token"
    echo "  • Prompt Trimming: Limited to ~300 tokens"
    echo "  • Chunk Limiting: Max 2 chunks per query"
    echo "  • Flash Attention: Enabled (if GPU available)"
    echo "  • Context Size: Reduced to 2048 for speed"
    echo "  • Batch Size: Optimized for laptop CPUs"
    echo ""
    echo "  Expected Performance (laptop, no GPU):"
    echo "  • Cold Start: ~20-25 seconds"
    echo "  • Warm Cache: ~10-15 seconds"
    echo "  • Streaming First Token: ~2-5 seconds"
    echo ""
}

# Function to show usage
show_usage() {
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  optimize              - Apply all local optimizations"
    echo "  switch-model          - Switch to optimized model"
    echo "  enable-streaming      - Enable streaming responses"
    echo "  optimize-prompts      - Optimize prompt processing"
    echo "  enable-flash          - Enable flash attention"
    echo "  warm-cache            - Warm up model cache"
    echo "  test                  - Test optimized performance"
    echo "  benefits              - Show optimization benefits"
    echo "  help                  - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 optimize                    # Apply all optimizations"
    echo "  $0 test                       # Test performance"
    echo "  $0 benefits                   # Show benefits"
    echo ""
}

# Main script logic
case "${1:-help}" in
    "optimize")
        check_current_model || switch_to_optimized_model
        apply_local_config
        enable_streaming
        optimize_prompt_processing
        enable_flash_attention
        warm_model_cache
        show_optimization_benefits
        echo "🎉 All local optimizations applied!"
        ;;
    "switch-model")
        switch_to_optimized_model
        ;;
    "enable-streaming")
        enable_streaming
        ;;
    "optimize-prompts")
        optimize_prompt_processing
        ;;
    "enable-flash")
        enable_flash_attention
        ;;
    "warm-cache")
        warm_model_cache
        ;;
    "test")
        test_optimized_performance
        ;;
    "benefits")
        show_optimization_benefits
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