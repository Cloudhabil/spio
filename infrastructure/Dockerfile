# Sovereign PIO - Main Application Container
FROM python:3.11-slim

LABEL maintainer="Sovereign PIO Team"
LABEL version="1.618.2"
LABEL description="Sovereign PIO - Personal Intelligent Operator"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY extensions/ ./extensions/

# Install dependencies
RUN pip install --no-cache-dir -e ".[server]"

# Create data directories
RUN mkdir -p /app/data/npu /app/data/memories

# Environment
ENV PYTHONPATH=/app/src
ENV SOVEREIGN_PIO_DATA=/app/data

# Expose ports
# 8009 - Substrate Gateway API
EXPOSE 8009

# Health check on the actual gateway port
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8009/api/status')" || exit 1

# Default command - run substrate gateway
CMD ["python", "-m", "server.substrate_gateway"]
