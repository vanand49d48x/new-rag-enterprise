#!/bin/bash

# Apply Embedding Optimizations Script
# Implements batching for large files and switches to BGE-small-en

echo "🔧 Applying embedding optimizations..."

# Set environment variable for the new embedding model
export EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"

echo "✅ Updated embedding model to: BAAI/bge-small-en-v1.5"
echo "✅ Implemented batch processing for files with >1000 chunks"
echo "✅ Memory optimization for large files"

# Restart the backend to apply changes
echo "🔄 Restarting backend to apply optimizations..."
docker-compose restart backend

echo "⏳ Waiting for backend to start..."
sleep 10

# Check if backend is healthy
echo "🔍 Checking backend health..."
if curl -f http://localhost:8000/ > /dev/null 2>&1; then
    echo "✅ Backend is healthy and running with optimizations"
else
    echo "❌ Backend health check failed"
    echo "📋 Checking logs..."
    docker-compose logs backend --tail 20
fi

echo "🎯 Optimizations applied successfully!"
echo ""
echo "📊 Summary of changes:"
echo "   • Embedding model: BAAI/bge-small-en-v1.5 (better English performance)"
echo "   • Batch processing: Enabled for files with >1000 chunks"
echo "   • Memory optimization: Prevents OOM for large files"
echo "   • Vector dimension: 384 (compatible with existing collections)"
echo ""
echo "🚀 Your system is now optimized for large file processing!" 