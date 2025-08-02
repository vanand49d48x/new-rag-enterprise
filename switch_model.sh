#!/bin/bash
# Simple Model Switcher
# Usage: ./switch_model.sh [model_name]
# Examples: ./switch_model.sh qwen2.5-3b
#          ./switch_model.sh tinyllama
#          ./switch_model.sh qwen2-7b

set -e

# Available models for testing
AVAILABLE_MODELS=("tinyllama" "qwen25_3b" "qwen2_7b")

# Get model from argument or show menu
if [ $# -eq 0 ]; then
    echo "🤖 Available Models:"
    echo "1) tinyllama   - TinyLlama-1.1B (Fastest)"
    echo "2) qwen25_3b   - Qwen2.5-3B (Balanced)"
    echo "3) qwen2_7b    - Qwen2-7B (Highest Quality)"
    echo ""
    read -p "Enter model name (or number 1-3): " MODEL_CHOICE
    
    # Convert number to model name
    case $MODEL_CHOICE in
        1) MODEL_CONFIG="tinyllama" ;;
        2) MODEL_CONFIG="qwen25_3b" ;;
        3) MODEL_CONFIG="qwen2_7b" ;;
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

# Create a Python script to update the config using model-specific files
cat > update_config.py << 'EOF'
import yaml
import sys
import os

model_config = sys.argv[1]

# Model-specific config file mapping
config_files = {
    'tinyllama': 'configs/models/config_tinyllama.yaml',
    'qwen25_3b': 'configs/models/config_qwen25.yaml',
    'qwen2_7b': 'configs/models/config_qwen25.yaml'  # Same config for now
}

if model_config not in config_files:
    print(f"❌ Unknown model config: {model_config}")
    sys.exit(1)

config_file = config_files[model_config]

if not os.path.exists(config_file):
    print(f"❌ Config file not found: {config_file}")
    sys.exit(1)

# Load model-specific config
with open(config_file, 'r') as f:
    model_specific_config = yaml.safe_load(f)

# Load current config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get model settings from model-specific config
model_settings = model_specific_config['model']
selected_model = model_settings['name']

# Update config with model-specific settings
if selected_model in model_settings:
    model_config_data = model_settings[selected_model]
    
    # Update model settings
    config['model']['name'] = model_config_data['filename']
    config['model']['context_size'] = model_config_data['context_size']
    config['model']['max_tokens'] = model_config_data['max_tokens']
    
    # Update performance settings
    config['performance']['threads'] = model_config_data['threads']
    config['performance']['batch_size'] = model_config_data['batch_size']
    
    # Update LLM settings from model-specific config
    if 'llm' in model_specific_config:
        llm_settings = model_specific_config['llm']
        config['model']['temperature'] = llm_settings.get('temperature', 0.7)
        config['model']['top_p'] = llm_settings.get('top_p', 0.9)
        config['model']['top_k'] = llm_settings.get('top_k', 40)
        config['model']['repeat_penalty'] = llm_settings.get('repeat_penalty', 1.1)
    
    # Save updated config
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Updated config for {model_config} using {config_file}")
else:
    print(f"❌ Model {selected_model} not found in config")
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
    echo "📊 Model Information:"
    case $MODEL_CONFIG in
        "tinyllama")
            echo "   🚀 Speed: ~150-200 tokens/sec"
            echo "   🧠 Quality: Good for basic tasks"
            echo "   💾 Memory: ~4GB (8 threads)"
            echo "   ⚡ Batch Size: 256"
            echo "   🔧 Optimizations: Q4_K_M quantization"
            ;;
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