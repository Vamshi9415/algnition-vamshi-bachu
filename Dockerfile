# =============================================================================
# AIgnition Forecast Studio — Production Docker Image
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# System deps for Prophet / LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libgomp1 curl bash \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create runtime directories
RUN mkdir -p data/raw output reports logs pickle

# Make run.sh executable
RUN chmod +x run.sh

# Expose FastAPI port
EXPOSE 8000

# Default: start API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Override for batch mode:
# docker run -e DATA_DIR=/data -e OUTPUT_PATH=/output/predictions.csv \
#   -v $(pwd)/data:/data -v $(pwd)/output:/output \
#   aignition bash run.sh
