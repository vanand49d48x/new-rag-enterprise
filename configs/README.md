# ⚙️ Configuration Files Documentation

This folder contains all configuration files for the Medical RAG System, organized by purpose and model.

## 📁 Folder Structure

```
configs/
├── models/              # Model-specific configurations
│   └── config_qwen25.yaml
├── optimized/           # Performance-optimized configurations
│   ├── config_fast.yaml
│   ├── config_optimized.yaml
│   ├── config_qwen25_optimized.yaml
│   └── config_tinyllama_optimized.yaml
├── production/          # Production-ready configurations
│   └── config_production.yaml
└── README.md           # This file
```

## 🎯 Configuration Categories

### **models/** - Model-Specific Configs
- **Purpose**: Base configurations for specific LLM models
- **Usage**: Starting point for model deployment
- **Examples**: Qwen2.5-3B, TinyLlama, Qwen2-7B

### **optimized/** - Performance-Optimized Configs
- **Purpose**: Configurations tuned for speed/quality balance
- **Usage**: Production deployment with specific optimizations
- **Features**: Enhanced parameters for better performance

### **production/** - Production-Ready Configs
- **Purpose**: Stable, tested configurations for production
- **Usage**: Live deployment with proven settings
- **Features**: Conservative, reliable settings

## 🔧 Configuration Parameters

### **Model Settings**
```yaml
model:
  name: "model-name"
  context_size: 2048      # Context window size
  threads: 16             # CPU threads
  batch_size: 512         # Batch processing size
  max_tokens: 256         # Maximum response length
```

### **API Settings**
```yaml
api:
  timeout: 120            # Request timeout (seconds)
```

### **LLM Settings**
```yaml
llm:
  temperature: 0.6        # Response creativity (0.0-1.0)
  top_p: 0.85            # Nucleus sampling
  top_k: 30              # Top-k sampling
  repeat_penalty: 1.08    # Repetition penalty
```

### **Processing Settings**
```yaml
processing:
  chunk_size: 400         # Document chunk size
  chunk_overlap: 50       # Chunk overlap
  max_workers: 6          # Parallel processing workers
```

## 🚀 Quick Configuration Commands

### Apply Model Configuration
```bash
# Apply Qwen2.5-3B config
cp configs/models/config_qwen25.yaml config.yaml
docker-compose down && docker-compose up -d

# Apply optimized config
cp configs/optimized/config_qwen25_optimized.yaml config.yaml
docker-compose down && docker-compose up -d

# Apply production config
cp configs/production/config_production.yaml config.yaml
docker-compose down && docker-compose up -d
```

### Model Switching
```bash
# Switch to Qwen2.5-3B with optimized settings
./switch_model.sh qwen2.5-3b
cp configs/optimized/config_qwen25_optimized.yaml config.yaml
docker-compose down && docker-compose up -d

# Switch to TinyLlama with optimized settings
./switch_model.sh tinyllama
cp configs/optimized/config_tinyllama_optimized.yaml config.yaml
docker-compose down && docker-compose up -d
```

## 📊 Configuration Comparison

### **Performance vs Quality Trade-offs**

| Configuration | Speed | Quality | Memory | Use Case |
|---------------|-------|---------|--------|----------|
| **config_fast.yaml** | ⚡ Fast | 🟡 Good | 💾 Low | High-traffic scenarios |
| **config_optimized.yaml** | 🚀 Balanced | 🟢 Excellent | 💾 Medium | Production deployment |
| **config_qwen25_optimized.yaml** | 🚀 Balanced | 🟢 Excellent | 💾 Medium | Medical RAG production |
| **config_tinyllama_optimized.yaml** | ⚡ Fast | 🟡 Good | 💾 Low | Speed-critical scenarios |

### **Model-Specific Optimizations**

#### **Qwen2.5-3B Optimized**
- **Temperature**: 0.3 (focused responses)
- **Context Size**: 4096 (larger context)
- **Max Tokens**: 256 (detailed answers)
- **Medical Focus**: Enhanced medical terminology

#### **TinyLlama Optimized**
- **Temperature**: 0.2 (very focused)
- **Context Size**: 2048 (balanced)
- **Max Tokens**: 256 (adequate detail)
- **Speed Focus**: Optimized for fast responses

## 🎯 Best Practices

### **Configuration Selection**
1. **Development**: Use model-specific configs
2. **Testing**: Use optimized configs
3. **Production**: Use production configs
4. **High-traffic**: Use fast configs

### **Performance Tuning**
1. **Speed**: Lower temperature, smaller context
2. **Quality**: Higher temperature, larger context
3. **Memory**: Adjust batch_size and max_workers
4. **Reliability**: Conservative timeout settings

### **Medical RAG Specific**
1. **Accuracy**: Lower temperature for precise answers
2. **Completeness**: Larger max_tokens for detailed responses
3. **Professional Tone**: Higher repeat_penalty for coherent text
4. **Medical Terms**: Enhanced medical terminology recognition

## 🔄 Configuration Workflow

### **Development Phase**
1. Start with model-specific config
2. Test performance and quality
3. Apply optimizations as needed
4. Document changes and results

### **Production Phase**
1. Use production config as baseline
2. Monitor performance metrics
3. Apply optimizations based on usage
4. Regular configuration reviews

### **Optimization Process**
1. **Baseline**: Run tests with current config
2. **Modify**: Adjust parameters based on needs
3. **Test**: Run performance and quality tests
4. **Deploy**: Apply if improvements are significant
5. **Monitor**: Track metrics in production

## 📋 Configuration Checklist

### **Before Deployment**
- [ ] Model compatibility verified
- [ ] Performance benchmarks run
- [ ] Quality tests passed
- [ ] Memory usage acceptable
- [ ] Timeout settings appropriate
- [ ] Error handling configured

### **Production Monitoring**
- [ ] Response time tracking
- [ ] Quality score monitoring
- [ ] Error rate tracking
- [ ] Resource usage monitoring
- [ ] User feedback collection

## 🛠️ Troubleshooting

### **Common Issues**
1. **High Memory Usage**: Reduce batch_size or context_size
2. **Slow Response**: Increase threads or reduce max_tokens
3. **Poor Quality**: Increase temperature or max_tokens
4. **Timeouts**: Increase api.timeout or optimize processing

### **Configuration Validation**
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Test configuration
python test/performance/test_performance.py
python test/quality/test_model_quality.py
```

---

**Last Updated**: $(date)
**RAG System Version**: Enterprise RAG v1.0
**Configuration Version**: v1.0 