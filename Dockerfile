# Backend as complete API service
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and Azure CLI
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    ca-certificates \
    lsb-release \
    gnupg \
    && curl -sL https://packages.microsoft.com/keys/microsoft.asc | \
       gpg --dearmor | \
       tee /etc/apt/trusted.gpg.d/microsoft.gpg > /dev/null \
    && echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ bookworm main" | \
       tee /etc/apt/sources.list.d/azure-cli.list \
    && apt-get update \
    && apt-get install azure-cli -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire backend service (includes data)
COPY . .

# Backend service creates its own data directories
RUN mkdir -p data/{raw,processed,indices} logs

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run container startup script (ensures data availability, then starts API)
CMD ["python", "/app/scripts/container_startup.py"]
