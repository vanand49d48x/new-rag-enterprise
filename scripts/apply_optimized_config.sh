#!/bin/bash
# Apply Optimized Configuration Script
# Keeps Qwen2-7B but applies performance optimizations

echo "⚡ Applying Optimized Configuration (Qwen2-7B + Speed Optimizations)..."

# 1. Backup current config
echo "📋 Backing up current configuration..."
cp config.yaml config.yaml.backup

# 2. Apply optimized configuration
echo "🔧 Applying optimized configuration..."
cp config_optimized.yaml config.yaml

# 3. Stop current services
echo "🛑 Stopping current services..."
docker-compose down

# 4. Restart with optimized configuration
echo "🔄 Restarting with optimized configuration..."
docker-compose up -d --build

# 5. Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# 6. Health check
echo "🔍 Checking system health..."
if curl -f -s "http://localhost:8000/" > /dev/null 2>&1; then
    echo "✅ System is ready with optimized configuration!"
    echo ""
    echo "📊 Performance Optimizations Applied:"
    echo "   ✅ Kept Qwen2-7B model (powerful and accurate)"
    echo "   ⚡ Increased threads from 8 to 16"
    echo "   ⚡ Reduced temperature from 0.7 to 0.6"
    echo "   ⚡ Reduced top_p from 0.9 to 0.85"
    echo "   ⚡ Reduced top_k from 40 to 30"
    echo "   ⚡ Reduced repeat_penalty from 1.1 to 1.08"
    echo "   ⚡ Reduced chunk_size from 400 to 350"
    echo "   ⚡ Increased max_workers from 4 to 6"
    echo "   ⚡ Reduced timeout from 180s to 120s"
    echo ""
    echo "🎯 Expected Performance Improvements:"
    echo "   - 20-30% faster response times"
    echo "   - Better parallel processing"
    echo "   - More focused responses"
    echo "   - Maintained model quality"
    echo ""
    echo "🧪 Test with: python test_performance.py"
else
    echo "❌ System failed to start properly"
    echo "Restoring backup configuration..."
    cp config.yaml.backup config.yaml
    docker-compose up -d
    exit 1
fi 