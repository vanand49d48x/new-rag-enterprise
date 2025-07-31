#!/bin/bash
# Apply Fast Configuration Script
# This script applies optimized settings for faster RAG performance

echo "🚀 Applying Fast Configuration..."

# 1. Backup current config
echo "📋 Backing up current configuration..."
cp config.yaml config.yaml.backup

# 2. Apply fast configuration
echo "⚡ Applying fast configuration..."
cp config_fast.yaml config.yaml

# 3. Stop current services
echo "🛑 Stopping current services..."
docker-compose down

# 4. Set environment variable for tinyllama model
export MODEL_CONFIG="tinyllama"

# 5. Restart with fast configuration
echo "🔄 Restarting with fast configuration..."
docker-compose up -d --build

# 6. Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# 7. Health check
echo "🔍 Checking system health..."
if curl -f -s "http://localhost:8000/" > /dev/null 2>&1; then
    echo "✅ System is ready with fast configuration!"
    echo ""
    echo "📊 Performance Optimizations Applied:"
    echo "   - Using TinyLlama (1.1B) instead of Qwen2-7B (7B)"
    echo "   - Increased threads from 8 to 16"
    echo "   - Reduced temperature from 0.7 to 0.5"
    echo "   - Reduced top_p from 0.9 to 0.8"
    echo "   - Reduced top_k from 40 to 20"
    echo "   - Smaller chunk size (300 vs 400)"
    echo "   - More workers (8 vs 4)"
    echo "   - Faster timeout (60s vs 180s)"
    echo ""
    echo "🎯 Expected Performance Improvements:"
    echo "   - ~5-10x faster response times"
    echo "   - Lower memory usage"
    echo "   - Better concurrent processing"
    echo ""
    echo "🧪 Test with: python test_simple_medical.py"
else
    echo "❌ System failed to start properly"
    echo "Restoring backup configuration..."
    cp config.yaml.backup config.yaml
    docker-compose up -d
    exit 1
fi 