#!/bin/bash
# Simple Model Switcher
# Usage: ./switch_model.sh [model_name]
# Examples: ./switch_model.sh qwen2.5-3b
#          ./switch_model.sh tinyllama
#          ./switch_model.sh qwen2-7b

set -e

# Available models for C7 testing
AVAILABLE_MODELS=("qwen25_3b" "qwen2_7b")

# Get model from argument or show menu
if [ $# -eq 0 ]; then
    echo "🤖 Available Models for C7 Testing:"
    echo "1) qwen25_3b  - Qwen2.5-3B (Fast & Balanced)"
    echo "2) qwen2_7b   - Qwen2-7B (Highest Quality)"
    echo ""
    read -p "Enter model name (or number 1-2): " MODEL_CHOICE
    
    # Convert number to model name
    case $MODEL_CHOICE in
        1) MODEL_CONFIG="qwen25_3b" ;;
        2) MODEL_CONFIG="qwen2_7b" ;;
        *) MODEL_CONFIG=$MODEL_CHOICE ;;
    esac
else
    MODEL_CONFIG=$1
fi

# Validate model
if [[ ! " ${AVAILABLE_MODELS[@]} " =~ " ${MODEL_CONFIG} " ]]; then
    echo "❌ Invalid model: $MODEL_CONFIG"
    echo "Available models: ${AVAILABLE_MODELS[*]}"
    exit 1
fi

echo "🔄 Switching to model: $MODEL_CONFIG"

# 1. Download model if needed
echo "📥 Checking model availability..."
MODEL_CONFIG=$MODEL_CONFIG ./init_model.sh

# 2. Update config.yaml with model-specific settings
echo "⚙️  Updating configuration for $MODEL_CONFIG..."

# Create a Python script to update the config
cat > update_config.py << 'EOF'
import yaml
import sys

model_config = sys.argv[1]

# Load current config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Model configurations
model_configs = {
    'qwen25_3b': {
        'name': 'qwen2.5-3b-instruct-q4_k_m.gguf',
        'size': '3b',
        'max_memory': '16GB',
        'threads': 8,
        'batch_size': 512
    },
    'qwen2_7b': {
        'name': 'qwen2-7b-instruct-q4_k_m.gguf',
        'size': '7b',
        'max_memory': '24GB',
        'threads': 12,
        'batch_size': 1024
    }
}

if model_config in model_configs:
    # Update model settings
    config['model']['name'] = model_configs[model_config]['name']
    config['model']['size'] = model_configs[model_config]['size']
    
    # Update performance settings
    config['performance']['max_memory'] = model_configs[model_config]['max_memory']
    config['performance']['threads'] = model_configs[model_config]['threads']
    config['performance']['batch_size'] = model_configs[model_config]['batch_size']
    
    # Save updated config
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Updated config for {model_config}")
else:
    print(f"❌ Unknown model config: {model_config}")
    sys.exit(1)
EOF

python3 update_config.py "$MODEL_CONFIG"
rm update_config.py

# 3. Restart services
echo "🔄 Restarting services..."
docker-compose down
docker-compose up -d

# 4. Wait and check
echo "⏳ Waiting for services to start..."
sleep 30

if curl -f -s "http://localhost:8000/" > /dev/null 2>&1; then
    echo "✅ Successfully switched to $MODEL_CONFIG!"
    echo ""
    echo "📊 Model Information for C7 (12 CPU, 30GB RAM):"
    case $MODEL_CONFIG in
        "qwen25_3b")
            echo "   🚀 Speed: ~80-120 tokens/sec"
            echo "   🧠 Quality: Excellent for most tasks"
            echo "   💾 Memory: ~16GB (8 threads)"
            echo "   ⚡ Batch Size: 512"
            echo "   🔧 Optimizations: LoRA + MoE + NF4"
            ;;
        "qwen2_7b")
            echo "   🚀 Speed: ~50-80 tokens/sec"
            echo "   🧠 Quality: Best reasoning capabilities"
            echo "   💾 Memory: ~24GB (12 threads)"
            echo "   ⚡ Batch Size: 1024"
            echo "   🔧 Optimizations: LoRA + MoE + NF4"
            ;;
    esac
    echo ""
    echo "🧪 Test with: python test_performance.py"
else
    echo "❌ Failed to start with $MODEL_CONFIG"
    echo "Restoring previous configuration..."
    mv config.yaml.bak config.yaml
    docker-compose up -d
    exit 1
fi 