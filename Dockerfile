# Multi-architecture Dockerfile for Enterprise RAG System
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for multi-modal processing
RUN apt-get update && apt-get install -y \
    # OCR and image processing
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    # OpenCV dependencies (architecture-agnostic)
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Audio processing
    ffmpeg \
    libsndfile1 \
    # Video processing
    libavcodec-extra \
    libavformat-dev \
    libswscale-dev \
    # General utilities
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    libmagic1 \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /tmp /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
