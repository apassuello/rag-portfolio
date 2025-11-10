# RAG Portfolio Project 1 - Docker Package with Ollama
# This Docker image includes all dependencies including Ollama and models

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data cache logs

# Set environment variables
ENV PYTHONPATH=/app
ENV RAG_DATA_DIR=/app/data
ENV RAG_CACHE_DIR=/app/cache
ENV OLLAMA_HOST=0.0.0.0

# Expose ports
EXPOSE 8000 8501 11434

# Create startup script
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting RAG Portfolio System"
echo "================================"

# Start Ollama service in background
echo "📦 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
sleep 10

# Pull required model
echo "📥 Pulling llama3.2:3b model..."
ollama pull llama3.2:3b

# Start Streamlit in background if enabled
if [ "$ENABLE_STREAMLIT" = "true" ]; then
    echo "🎨 Starting Streamlit demo..."
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
    STREAMLIT_PID=$!
    echo "  - Streamlit Demo: http://localhost:8501"
fi

echo "✅ System ready!"
echo ""
echo "Available endpoints:"
echo "  - RAG API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
if [ "$ENABLE_STREAMLIT" = "true" ]; then
    echo "  - Streamlit Demo: http://localhost:8501"
fi
echo ""

# Keep container running and handle shutdown
trap "echo 'Shutting down...'; kill $OLLAMA_PID; [ ! -z $STREAMLIT_PID ] && kill $STREAMLIT_PID; exit 0" SIGTERM SIGINT

# Start the main application
exec "$@"
EOF

RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:11434/api/tags || exit 1

# Default command - start both API server and Streamlit
CMD ["/app/start.sh", "python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]