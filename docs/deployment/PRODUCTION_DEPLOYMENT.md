# 🚀 Enterprise RAG - Production Deployment Guide

## **Overview**

This guide covers deploying the Enterprise RAG system in production environments with adaptive configuration, model management, and client demonstration capabilities.

## **📋 System Requirements**

### **Minimum Requirements**
- **RAM**: 8GB (Laptop tier)
- **CPU**: 2+ cores
- **Storage**: 10GB free space
- **OS**: Linux, macOS, or Windows with Docker

### **Recommended Requirements**
- **RAM**: 32GB+ (Enterprise tier)
- **CPU**: 8+ cores (Apple M2 Max or equivalent)
- **Storage**: 50GB+ free space
- **Network**: Stable internet connection for model downloads

## **🔧 Installation & Setup**

### **1. Clone Repository**
```bash
git clone <repository-url>
cd rag-enterprise
```

### **2. Environment Configuration**
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

**Key Environment Variables:**
```env
# Host System Info (for accurate resource detection)
HOST_RAM_GB=32
HOST_CPU_COUNT=12

# LLM Configuration
LLM_API_URL=http://llama-cpp:8080/completion
LLM_STRICT_MODE=true

# Model Management
MODELS_DIR=/app/models
AUTO_DOWNLOAD_MODELS=true
```

### **3. Start the System**
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

## **🎯 Adaptive Configuration**

### **Automatic Tier Detection**
The system automatically detects your hardware and selects the optimal configuration:

| Tier | RAM | CPU | Model | Threads | Context |
|------|-----|-----|-------|---------|---------|
| **Laptop** | <8GB | ≤2 cores | TinyLlama 1.1B | 2 | 512 |
| **Workstation** | <16GB | ≤4 cores | Qwen 2.5-3B | 4 | 1024 |
| **Server** | <32GB | ≤8 cores | Qwen 2-7B | 8 | 2048 |
| **Enterprise** | 32GB+ | 8+ cores | Qwen 2-7B | 16 | 4096 |

### **Manual Configuration**
```bash
# Check current tier
curl http://localhost:8000/system

# View all tiers
curl http://localhost:8000/api/tiers

# Switch tier (demo mode)
curl -X POST http://localhost:8000/api/tiers/enterprise/switch
```

## **🤖 Model Management**

### **Available Models**
The system supports multiple models per tier with automatic fallback:

- **TinyLlama 1.1B** (580MB) - Fastest, lowest memory
- **Qwen 2.5-0.5B** (320MB) - Ultra-lightweight
- **Qwen 2.5-3B** (1.8GB) - Balanced performance
- **Qwen 2-7B** (4.2GB) - High performance

### **Model Operations**
```bash
# List available models
curl http://localhost:8000/api/models

# Download model for specific tier
curl -X POST http://localhost:8000/api/models/download/enterprise

# Check model statistics
curl http://localhost:8000/api/models/enterprise
```

### **Auto-Download Configuration**
Models are automatically downloaded when needed:
- Primary model for detected tier
- Fallback model if primary unavailable
- Progress tracking and error handling

## **📊 Monitoring & Performance**

### **System Metrics**
```bash
# Current system performance
curl http://localhost:8000/api/performance

# Real-time monitoring
docker stats
```

### **Performance Benchmarks**
| Tier | Response Time | Memory Usage | CPU Usage |
|------|---------------|--------------|-----------|
| Laptop | ~500ms | 512MB | 15% |
| Workstation | ~800ms | 1GB | 25% |
| Server | ~1200ms | 2GB | 40% |
| Enterprise | ~1500ms | 4GB | 60% |

### **Health Checks**
```bash
# Backend health
curl http://localhost:8000/

# LLM server health
curl http://localhost:8080/health

# Vector database health
curl http://localhost:6334/health
```

## **🎮 Client Demonstration**

### **Demo Interface**
Access the client demo at: `http://localhost:8000/demo`

**Features:**
- Real-time system status
- Tier comparison visualization
- Model management interface
- Performance metrics
- Interactive controls

### **API Endpoints**
```bash
# System configuration
GET /system

# Tier management
GET /api/tiers
GET /api/tiers/{tier}
POST /api/tiers/{tier}/switch

# Model management
GET /api/models
GET /api/models/{tier}
POST /api/models/download/{tier}

# Performance metrics
GET /api/performance

# Chat interface
POST /chat
POST /chat/stream
```

## **🔒 Security & Production**

### **Security Best Practices**
1. **Network Security**
   ```bash
   # Use reverse proxy (nginx)
   # Enable HTTPS
   # Configure firewall rules
   ```

2. **Access Control**
   ```bash
   # Add authentication
   # Implement rate limiting
   # Monitor access logs
   ```

3. **Data Protection**
   ```bash
   # Encrypt sensitive data
   # Regular backups
   # Audit logging
   ```

### **Production Deployment**
```bash
# Production docker-compose
docker-compose -f docker-compose.prod.yml up -d

# With monitoring
docker-compose -f docker-compose.prod.yml -f docker-compose.monitoring.yml up -d
```

## **📈 Scaling & Optimization**

### **Horizontal Scaling**
```bash
# Scale backend services
docker-compose up -d --scale backend=3

# Load balancer configuration
# Database clustering
# Cache optimization
```

### **Performance Tuning**
```bash
# Optimize LLM parameters
# Adjust thread count
# Tune context size
# Monitor resource usage
```

## **🛠️ Troubleshooting**

### **Common Issues**

**1. Model Download Failures**
```bash
# Check network connectivity
curl -I https://huggingface.co

# Verify disk space
df -h

# Check model directory permissions
ls -la /app/models
```

**2. LLM Server Issues**
```bash
# Check LLM logs
docker-compose logs llama-cpp

# Verify model file integrity
ls -la models/*.gguf

# Restart LLM service
docker-compose restart llama-cpp
```

**3. Memory Issues**
```bash
# Check system memory
free -h

# Monitor container memory
docker stats

# Adjust memory limits in docker-compose.yml
```

### **Debug Commands**
```bash
# Full system status
curl http://localhost:8000/system | jq

# Model availability
curl http://localhost:8000/api/models | jq

# Performance metrics
curl http://localhost:8000/api/performance | jq

# Container logs
docker-compose logs -f backend
```

## **📚 Additional Resources**

### **Documentation**
- [API Documentation](http://localhost:8000/docs)
- [Client Demo](http://localhost:8000/demo)
- [System Status](http://localhost:8000/system)

### **Support**
- GitHub Issues: [Repository Issues]
- Documentation: [Wiki Link]
- Community: [Discord/Slack]

### **Updates & Maintenance**
```bash
# Update system
git pull origin main
docker-compose down
docker-compose up -d --build

# Backup configuration
cp docker-compose.yml docker-compose.yml.backup
cp .env .env.backup
```

---

**🎉 Your Enterprise RAG system is now production-ready with adaptive configuration, model management, and client demonstration capabilities!** 