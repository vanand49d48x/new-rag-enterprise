# backend/Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y gcc libmagic-dev poppler-utils && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Set PYTHONPATH so "backend" is discoverable as a top-level module
ENV PYTHONPATH=/app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]



