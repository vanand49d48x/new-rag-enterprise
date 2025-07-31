# 🧪 Test Suite Documentation

This folder contains comprehensive tests for the Medical RAG System, organized by category.

## 📁 Folder Structure

```
test/
├── performance/          # Performance and speed tests
│   └── test_performance.py
├── quality/             # Quality and accuracy tests
│   └── test_model_quality.py
├── benchmark/           # Benchmark reports and analysis
│   └── production_benchmark_report.md
├── medical/             # Medical-specific RAG tests
│   └── test_medical_rag.py
└── README.md           # This file
```

## 🚀 Performance Tests

### `performance/test_performance.py`
- **Purpose**: Measures response times and performance metrics
- **Usage**: `python test/performance/test_performance.py`
- **Metrics**: Average, min, max response times, success rates
- **Rating**: EXCELLENT (<10s), GOOD (<20s), SLOW (<30s), VERY SLOW (>30s)

## 🧠 Quality Tests

### `quality/test_model_quality.py`
- **Purpose**: Evaluates response quality across different models
- **Usage**: `python test/quality/test_model_quality.py`
- **Metrics**: Medical terminology, structure, professional tone, completeness
- **Scoring**: 0.0-1.0 scale with detailed breakdown

## 📊 Benchmark Reports

### `benchmark/production_benchmark_report.md`
- **Purpose**: Comprehensive production deployment analysis
- **Content**: Model comparisons, architecture recommendations, production readiness
- **Recommendations**: Qwen2.5-3B for balanced performance

## 🏥 Medical Tests

### `medical/test_medical_rag.py`
- **Purpose**: Medical-specific RAG functionality testing
- **Usage**: `python test/medical/test_medical_rag.py`
- **Queries**: Diabetes, hypertension, asthma, depression, medication interactions
- **Focus**: Medical accuracy and relevance

## 🎯 Quick Test Commands

### Performance Testing
```bash
# Run performance test
python test/performance/test_performance.py

# Expected output: Response times and performance rating
```

### Quality Testing
```bash
# Run quality test
python test/quality/test_model_quality.py

# Expected output: Quality scores and detailed breakdown
```

### Medical RAG Testing
```bash
# Run medical RAG test
python test/medical/test_medical_rag.py

# Expected output: Medical query results and accuracy metrics
```

## 📈 Test Results Interpretation

### Performance Ratings
- **🟢 EXCELLENT**: < 10 seconds
- **🟡 GOOD**: 10-20 seconds
- **🟠 SLOW**: 20-30 seconds
- **🔴 VERY SLOW**: > 30 seconds

### Quality Ratings
- **🟢 EXCELLENT**: > 0.8
- **🟡 GOOD**: 0.6-0.8
- **🟠 FAIR**: 0.4-0.6
- **🔴 POOR**: < 0.4

### Production Readiness
- **🟢 EXCELLENT**: > 0.8 (Speed + Quality + Reliability)
- **🟡 GOOD**: 0.6-0.8
- **🟠 FAIR**: 0.4-0.6
- **🔴 POOR**: < 0.4

## 🔧 Configuration

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install requests statistics
```

### API Configuration
- **Base URL**: `http://localhost:8000`
- **Timeout**: 120-180 seconds for complex queries
- **Top-k**: 5 for retrieval

## 📋 Test Categories

### 1. **Performance Tests**
- Response time measurement
- Throughput analysis
- Load testing capabilities
- Performance regression detection

### 2. **Quality Tests**
- Medical terminology accuracy
- Response structure evaluation
- Professional tone assessment
- Completeness scoring

### 3. **Benchmark Tests**
- Model comparison analysis
- Production readiness assessment
- Architecture recommendations
- Cost-benefit analysis

### 4. **Medical Tests**
- Medical query accuracy
- Clinical relevance
- Safety and reliability
- Domain-specific validation

## 🎯 Best Practices

### Running Tests
1. **Ensure system is running**: `./start.sh`
2. **Check API health**: `curl http://localhost:8000/health`
3. **Run specific test**: `python test/[category]/test_[name].py`
4. **Review results**: Check performance and quality metrics

### Interpreting Results
1. **Performance**: Focus on average response times
2. **Quality**: Look at medical terminology and structure scores
3. **Reliability**: Check success rates and error handling
4. **Production**: Consider speed/quality balance for deployment

### Continuous Testing
- **Daily**: Run performance tests
- **Weekly**: Run quality and medical tests
- **Monthly**: Run full benchmark suite
- **Before deployment**: Run all tests with production config

## 🔄 Model Testing Workflow

1. **Switch Model**: `./switch_model.sh [model_name]`
2. **Wait for Startup**: 30-60 seconds
3. **Run Performance Test**: `python test/performance/test_performance.py`
4. **Run Quality Test**: `python test/quality/test_model_quality.py`
5. **Compare Results**: Against baseline metrics
6. **Document Findings**: Update benchmark reports

## 📊 Metrics Dashboard

### Key Performance Indicators
- **Response Time**: Target < 30s average
- **Success Rate**: Target > 95%
- **Quality Score**: Target > 0.8
- **Medical Accuracy**: Target > 85%

### Quality Metrics
- **Medical Terminology**: Presence of clinical terms
- **Structure**: Organized, coherent responses
- **Professional Tone**: Healthcare-appropriate language
- **Completeness**: Coverage of query aspects

---

**Last Updated**: $(date)
**Test Environment**: macOS, 16GB RAM, 8 CPU cores
**RAG System Version**: Enterprise RAG v1.0 