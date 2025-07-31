# 🚀 Enterprise RAG System - Paperspace C7 Deployment

## Overview

This is a complete Enterprise RAG (Retrieval-Augmented Generation) system optimized for Paperspace C7 instances (12 CPU cores, 30GB RAM). The system supports both Qwen2.5-3B and Qwen2-7B models with all advanced optimizations enabled.

## 🎯 Features

- **Dual Model Support**: Qwen2.5-3B (fast) and Qwen2-7B (high quality)
- **Advanced Optimizations**: LoRA + MoE + NF4 quantization
- **C7 Optimized**: Configured for 12 CPU cores and 30GB RAM
- **Easy Model Switching**: Seamless transitions between models
- **Comprehensive Testing**: Automated performance benchmarking
- **Production Ready**: Docker-based deployment with monitoring

## 📋 System Requirements

- **Paperspace Instance**: C7 or better (12+ CPU cores, 30GB+ RAM)
- **OS**: Ubuntu 20.04+ (automatically handled by Paperspace)
- **Storage**: 50GB+ free space
- **Network**: Stable internet connection for model downloads

## 🚀 Quick Deployment

### Step 1: Create Paperspace Instance

1. Go to [Paperspace](https://www.paperspace.com/)
2. Create a new Notebook with C7 specifications:
   - **Machine**: C7 (12 CPU, 30GB RAM)
   - **Storage**: 50GB+
   - **OS**: Ubuntu 20.04+

### Step 2: Upload and Deploy

```bash
# 1. Upload the compressed project to Paperspace
# (Upload the rag-enterprise.tar.gz file)

# 2. Extract the project
tar -xzf rag-enterprise.tar.gz
cd rag-enterprise

# 3. Run the deployment script
./deploy_paperspace.sh
```

### Step 3: Verify Deployment

```bash
# Check service status
docker ps

# Test LLM inference
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "n_predict": 20}'

# Test model switching
./switch_model.sh qwen25_3b
./switch_model.sh qwen2_7b

# Run comprehensive tests
python3 test_models.py
```

## 📊 Expected Performance (C7)

| Model | Speed | Memory | Quality | Use Case |
|-------|-------|--------|---------|----------|
| **Qwen2.5-3B** | 80-120 tokens/sec | 16GB | Excellent | Most tasks |
| **Qwen2-7B** | 50-80 tokens/sec | 24GB | Best | Complex reasoning |

## 🔧 Configuration

### Model Switching

```bash
# Switch to Qwen2.5-3B (faster)
./switch_model.sh qwen25_3b

# Switch to Qwen2-7B (higher quality)
./switch_model.sh qwen2_7b
```

### Advanced Optimizations

All optimizations are enabled by default:
- ✅ **LoRA Fine-tuning**: Adaptive model adaptation
- ✅ **MoE (Mixture of Experts)**: Multi-expert routing
- ✅ **NF4 Quantization**: 4-bit precision for efficiency
- ✅ **NUMA Awareness**: Optimized for multi-core systems
- ✅ **Cache Optimization**: Prefetching and memory management

## 🧪 Testing

### Quick Tests

```bash
# Test LLM inference
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "n_predict": 100}'

# Test RAG pipeline
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

### Comprehensive Testing

```bash
# Run full test suite
python3 test_models.py

# Monitor system resources
docker stats

# Check service logs
docker-compose logs -f
```

## 📈 Monitoring

### Service URLs

- **Backend API**: http://localhost:8000
- **LLM Server**: http://localhost:8080
- **Vector DB**: http://localhost:6334
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### Monitoring Commands

```bash
# View service status
docker ps

# Monitor resource usage
docker stats

# View logs
docker-compose logs -f

# Check system resources
htop
free -h
```

## 🔍 Troubleshooting

### Common Issues

1. **Docker Permission Error**
   ```bash
   # Logout and login again
   exit
   # Reconnect to Paperspace
   ```

2. **Model Download Issues**
   ```bash
   # Manual model download
   wget -O models/qwen2.5-3b-instruct-q4_k_m.gguf \
     "https://huggingface.co/TheBloke/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
   ```

3. **Service Not Starting**
   ```bash
   # Restart services
   docker-compose down
   docker-compose up -d
   
   # Check logs
   docker-compose logs -f
   ```

### Performance Optimization

```bash
# Check CPU usage
htop

# Monitor memory
free -h

# Check Docker resource usage
docker stats

# Optimize for your specific workload
# Edit config.yaml for custom settings
```

## 📁 Project Structure

```
rag-enterprise/
├── deploy_paperspace.sh      # Paperspace deployment script
├── requirements.txt          # Python dependencies
├── config.yaml              # System configuration
├── start.sh                 # Service startup script
├── switch_model.sh          # Model switching script
├── test_models.py           # Comprehensive testing
├── docker-compose.yml       # Docker services
├── backend/                 # Backend API code
├── models/                  # Model files (downloaded)
├── scripts/                 # Utility scripts
└── docs/                    # Documentation
```

## 🎯 Use Cases

### Qwen2.5-3B (Recommended for most tasks)
- **Speed**: 80-120 tokens/sec
- **Memory**: 16GB
- **Use Cases**: General chat, document Q&A, basic reasoning

### Qwen2-7B (For complex tasks)
- **Speed**: 50-80 tokens/sec
- **Memory**: 24GB
- **Use Cases**: Complex reasoning, detailed analysis, research

## 🔄 Model Comparison

| Feature | Qwen2.5-3B | Qwen2-7B |
|---------|-------------|-----------|
| **Speed** | ⚡⚡⚡⚡⚡ | ⚡⚡⚡ |
| **Quality** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Memory** | 💾💾💾 | 💾💾💾💾 |
| **Best For** | General use | Complex tasks |

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review service logs: `docker-compose logs -f`
3. Monitor system resources: `htop` and `docker stats`

## 🚀 Next Steps

After deployment:
1. Test both models: `./switch_model.sh qwen25_3b` and `./switch_model.sh qwen2_7b`
2. Run performance tests: `python3 test_models.py`
3. Upload your documents for RAG testing
4. Monitor performance with Grafana dashboard

---

**🎉 Your Enterprise RAG System is ready for production use on Paperspace C7!** 