# 🎯 RAG Improvement Guide

## 📋 Overview

This guide explains how to use the comprehensive RAG improvement system for optimizing TinyLlama and Qwen2.5-3B models. The system reuses existing infrastructure and provides configurable model switching and parameter optimization.

## 🏗️ System Architecture

### **Reused Components**
- **Existing Test Infrastructure**: `test/quality/test_model_quality.py`
- **Configuration Management**: `apply_config.sh` and `configs/` structure
- **Model Switching**: `switch_model.sh`
- **Quality Evaluation**: Enhanced from existing quality tests

### **New Components**
- **RAG Improvement Tester**: `test/quality/test_rag_improvement.py`
- **Optimization Script**: `scripts/rag_optimizer.sh`
- **Configurable Parameters**: Multiple optimization profiles per model

## 🎯 Optimization Strategies

### **1. Quality Focus**
- **Purpose**: Maximum response quality for medical queries
- **Parameters**: Lower temperature, higher top_p, larger context
- **Use Case**: Production deployment where accuracy is critical
- **Trade-off**: Slower response times

### **2. Speed Focus**
- **Purpose**: Fast response times for real-time applications
- **Parameters**: Higher temperature, lower top_p, smaller context
- **Use Case**: Interactive applications requiring quick responses
- **Trade-off**: Reduced quality and detail

### **3. Balanced Approach**
- **Purpose**: Optimal balance of quality and speed
- **Parameters**: Moderate settings across all parameters
- **Use Case**: General production deployment
- **Trade-off**: Compromise between quality and speed

## 🔧 Configuration Parameters

### **Qwen2.5-3B Optimizations**

#### **Quality Focus**
```yaml
temperature: 0.2          # Low randomness for consistency
top_p: 0.95              # High nucleus sampling
top_k: 50                # More token options
repeat_penalty: 1.15     # Prevent repetition
max_tokens: 320          # Longer responses
context_size: 4096       # Larger context window
chunk_size: 450          # Larger document chunks
chunk_overlap: 60        # More overlap for coherence
```

#### **Speed Focus**
```yaml
temperature: 0.4          # Higher randomness for speed
top_p: 0.85              # Lower nucleus sampling
top_k: 30                # Fewer token options
repeat_penalty: 1.05     # Less repetition penalty
max_tokens: 192          # Shorter responses
context_size: 2048       # Smaller context window
chunk_size: 350          # Smaller document chunks
chunk_overlap: 40        # Less overlap for speed
```

#### **Balanced**
```yaml
temperature: 0.3          # Moderate randomness
top_p: 0.9               # Balanced nucleus sampling
top_k: 40                # Moderate token options
repeat_penalty: 1.1      # Moderate repetition penalty
max_tokens: 256          # Moderate response length
context_size: 4096       # Large context for quality
chunk_size: 400          # Balanced chunk size
chunk_overlap: 50        # Balanced overlap
```

### **TinyLlama Optimizations**

#### **Quality Focus**
```yaml
temperature: 0.1          # Very low randomness
top_p: 0.95              # High nucleus sampling
top_k: 50                # More token options
repeat_penalty: 1.2      # Strong repetition penalty
max_tokens: 320          # Longer responses
context_size: 2048       # Moderate context
chunk_size: 400          # Larger chunks
chunk_overlap: 50        # More overlap
```

#### **Speed Focus**
```yaml
temperature: 0.3          # Higher randomness
top_p: 0.85              # Lower nucleus sampling
top_k: 30                # Fewer token options
repeat_penalty: 1.1      # Light repetition penalty
max_tokens: 192          # Shorter responses
context_size: 1024       # Small context
chunk_size: 300          # Small chunks
chunk_overlap: 30        # Minimal overlap
```

#### **Balanced**
```yaml
temperature: 0.2          # Low randomness
top_p: 0.9               # Balanced nucleus sampling
top_k: 40                # Moderate token options
repeat_penalty: 1.15     # Moderate repetition penalty
max_tokens: 256          # Moderate response length
context_size: 2048       # Moderate context
chunk_size: 350          # Balanced chunk size
chunk_overlap: 40        # Balanced overlap
```

## 🚀 Usage Guide

### **1. Interactive Mode**
```bash
# Start interactive optimization
./scripts/rag_optimizer.sh interactive

# Or simply run without arguments
./scripts/rag_optimizer.sh
```

### **2. Command Line Mode**

#### **Optimize Specific Model**
```bash
# Optimize Qwen2.5-3B with balanced approach
./scripts/rag_optimizer.sh optimize qwen2.5-3b balanced

# Optimize TinyLlama for quality
./scripts/rag_optimizer.sh optimize tinyllama quality_focus

# Optimize Qwen2.5-3B for speed
./scripts/rag_optimizer.sh optimize qwen2.5-3b speed_focus
```

#### **Apply Best Configuration**
```bash
# Find and apply best config for Qwen2.5-3B
./scripts/rag_optimizer.sh apply-best qwen2.5-3b

# Find and apply best config for TinyLlama
./scripts/rag_optimizer.sh apply-best tinyllama
```

#### **Quick Testing**
```bash
# Quick test current Qwen2.5-3B setup
./scripts/rag_optimizer.sh quick-test qwen2.5-3b

# Quick test current TinyLlama setup
./scripts/rag_optimizer.sh quick-test tinyllama
```

#### **Model Comparison**
```bash
# Compare TinyLlama vs Qwen2.5-3B
./scripts/rag_optimizer.sh compare
```

### **3. Direct Python Usage**

#### **Run Optimization**
```bash
cd test/quality
python test_rag_improvement.py optimize qwen2.5-3b
python test_rag_improvement.py optimize tinyllama
```

#### **Test Specific Configuration**
```bash
cd test/quality
python test_rag_improvement.py test qwen2.5-3b balanced
python test_rag_improvement.py test tinyllama quality_focus
```

## 📊 Quality Evaluation Metrics

### **Enhanced Quality Scoring**
The system evaluates responses using multiple criteria:

1. **Medical Terminology (25%)**: Presence of medical terms
2. **Keyword Coverage (25%)**: Expected keywords found
3. **Structure Score (20%)**: Organized, coherent response
4. **Professional Tone (15%)**: Medical professional language
5. **Completeness (10%)**: Comprehensive coverage
6. **Length Score (5%)**: Appropriate detail level

### **Medical Terms Recognized**
- **Symptoms**: symptoms, diagnosis, patient
- **Treatment**: treatment, medication, therapy, prescription
- **Medical Context**: doctor, hospital, healthcare, professional
- **Technical Terms**: dosage, side effects, interactions, contraindications
- **Professional Language**: recommended, consult, advise, monitoring

### **Structure Indicators**
- **Numbered Lists**: "1.", "2.", "3."
- **Bullet Points**: "-", "•"
- **Sequential Markers**: "First", "Second", "Third"
- **Connectors**: "Additionally", "Furthermore"

## 🔄 Workflow Examples

### **Example 1: Production Deployment**
```bash
# 1. Start with Qwen2.5-3B
./switch_model.sh qwen2.5-3b

# 2. Run comprehensive optimization
./scripts/rag_optimizer.sh optimize qwen2.5-3b balanced

# 3. Apply best configuration
./scripts/rag_optimizer.sh apply-best qwen2.5-3b

# 4. Verify with quick test
./scripts/rag_optimizer.sh quick-test qwen2.5-3b
```

### **Example 2: Speed Optimization**
```bash
# 1. Switch to TinyLlama for speed
./switch_model.sh tinyllama

# 2. Optimize for speed
./scripts/rag_optimizer.sh optimize tinyllama speed_focus

# 3. Apply speed configuration
./scripts/rag_optimizer.sh apply-best tinyllama

# 4. Test performance
./scripts/rag_optimizer.sh quick-test tinyllama
```

### **Example 3: Quality Comparison**
```bash
# 1. Compare both models
./scripts/rag_optimizer.sh compare

# 2. Analyze results and choose best
# 3. Apply chosen configuration
./scripts/rag_optimizer.sh apply-best qwen2.5-3b
```

## 📈 Performance Benchmarks

### **Expected Results**

#### **Qwen2.5-3B**
- **Quality Focus**: 0.85+ quality, 25-35s response
- **Speed Focus**: 0.70+ quality, 15-25s response
- **Balanced**: 0.80+ quality, 20-30s response

#### **TinyLlama**
- **Quality Focus**: 0.75+ quality, 20-30s response
- **Speed Focus**: 0.60+ quality, 10-20s response
- **Balanced**: 0.70+ quality, 15-25s response

### **Quality Thresholds**
- **Excellent**: 0.85+ (Production ready)
- **Good**: 0.75-0.84 (Acceptable for production)
- **Fair**: 0.65-0.74 (Needs optimization)
- **Poor**: <0.65 (Requires significant improvement)

## 🛠️ Troubleshooting

### **Common Issues**

#### **System Not Running**
```bash
# Start the system
./start.sh

# Wait for services to be ready
sleep 30

# Then run optimization
./scripts/rag_optimizer.sh optimize qwen2.5-3b balanced
```

#### **Configuration Not Applied**
```bash
# Check current configuration
curl -s http://localhost:8000/system

# Manually apply configuration
./apply_config.sh optimized config_qwen25_balanced

# Restart services
docker-compose down && docker-compose up -d
```

#### **Test Failures**
```bash
# Check API health
curl -s http://localhost:8000/health

# Check logs
docker-compose logs llama-cpp

# Restart services
docker-compose restart
```

### **Debug Mode**
```bash
# Run with verbose output
cd test/quality
python test_rag_improvement.py optimize qwen2.5-3b 2>&1 | tee optimization.log

# Check detailed results
cat optimization.log
```

## 🎯 Best Practices

### **1. Optimization Strategy**
- **Start with balanced** configuration for initial testing
- **Use quality focus** for production medical applications
- **Use speed focus** for interactive applications
- **Test thoroughly** before production deployment

### **2. Model Selection**
- **Qwen2.5-3B**: Better quality, larger model, slower inference
- **TinyLlama**: Faster inference, smaller model, lower quality
- **Choose based on** your specific requirements

### **3. Configuration Management**
- **Always backup** before applying new configurations
- **Test thoroughly** after configuration changes
- **Monitor performance** in production
- **Document changes** and their impact

### **4. Quality Assurance**
- **Run comprehensive tests** before deployment
- **Compare multiple configurations** to find optimal settings
- **Monitor quality metrics** in production
- **Iterate and improve** based on real-world usage

## 🔗 Related Documentation

- **Configuration Management**: `configs/README.md`
- **Testing Framework**: `test/README.md`
- **Model Registry**: `backend/utils/model_registry.py`
- **Quality Testing**: `test/quality/test_model_quality.py`
- **Performance Testing**: `test/performance/test_performance.py`

---

**RAG Improvement System Version**: v1.0
**Last Updated**: $(date)
**Compatible Models**: TinyLlama, Qwen2.5-3B
**Optimization Types**: Quality Focus, Speed Focus, Balanced 