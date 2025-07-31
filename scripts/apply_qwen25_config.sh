#!/bin/bash
# Apply Qwen2.5-3B Configuration Script
# Switches to Qwen2.5-3B for better speed while maintaining quality

echo "⚡ Applying Qwen2.5-3B Configuration..."

# 1. Backup current config
echo "📋 Backing up current configuration..."
cp config.yaml config.yaml.backup

# 2. Apply Qwen2.5-3B configuration
echo "🔧 Applying Qwen2.5-3B configuration..."
cp config_qwen25.yaml config.yaml

# 3. Check if Qwen2.5-3B model exists
echo "🔍 Checking Qwen2.5-3B model..."
if [ ! -f "models/qwen2.5-3b-instruct-q4_k_m.gguf" ] || [ ! -s "models/qwen2.5-3b-instruct-q4_k_m.gguf" ]; then
    echo "📥 Qwen2.5-3B model not found or empty. Downloading..."
    MODEL_CONFIG="qwen2.5-3b" ./init_model.sh
else
    echo "✅ Qwen2.5-3B model found"
fi

# 4. Stop current services
echo "🛑 Stopping current services..."
docker-compose down

# 5. Restart with Qwen2.5-3B configuration
echo "🔄 Restarting with Qwen2.5-3B configuration..."
docker-compose up -d --build

# 6. Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# 7. Health check
echo "🔍 Checking system health..."
if curl -f -s "http://localhost:8000/" > /dev/null 2>&1; then
    echo "✅ System is ready with Qwen2.5-3B!"
    echo ""
    echo "📊 Configuration Applied:"
    echo "   ✅ Switched to Qwen2.5-3B (3B parameters)"
    echo "   ⚡ Expected 2-3x faster than Qwen2-7B"
    echo "   🧠 Good quality for medical queries"
    echo "   💾 Lower memory usage (~2GB vs 4GB)"
    echo "   ⚙️  Optimized settings for enterprise hardware"
    echo ""
    echo "🎯 Expected Performance Improvements:"
    echo "   - 20-30 seconds average response time (vs 66s)"
    echo "   - Better concurrent processing"
    echo "   - Maintained quality for medical queries"
    echo "   - Lower resource usage"
    echo ""
    echo "🧪 Test with: python test_performance.py"
else
    echo "❌ System failed to start properly"
    echo "Restoring backup configuration..."
    cp config.yaml.backup config.yaml
    docker-compose up -d
    exit 1
fi 