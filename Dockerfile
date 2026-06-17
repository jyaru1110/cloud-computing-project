FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.32.0 \
    scikit-learn==1.6.1 \
    numpy==1.26.4 \
    scipy==1.14.1 \
    joblib==1.4.2 \
    requests==2.32.3 \
    pydantic==2.9.2 \
    pandas>=2.2.0 \
    google-cloud-storage>=2.14.0 \
    google-cloud-secret-manager>=2.20.0

# Copy source code (generic -- no model baked in)
COPY src/ /app/src/

# Create empty model dir (will be populated from GCS at startup)
RUN mkdir -p /app/model

# Copy serving entrypoint (supports --mode train and --mode serve)
COPY src/serving/train.py /app/entrypoint.py

# Environment
ENV MODEL_DIR=/app/model
ENV PORT=8080
ENV PYTHONPATH=/app
# GCS_MODEL_URI must be set at deploy time, e.g.:
#   gs://mlops-toxic-comments-ml/model
ENV GCS_MODEL_URI=""
ENV PROJECT_ID=""

EXPOSE 8080

ENTRYPOINT ["python", "/app/entrypoint.py"]
CMD ["--mode", "serve"]
