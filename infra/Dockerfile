FROM python:3.11-slim

# Set working directory
WORKDIR /mlflow

RUN chmod -R 777 /mlflow

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    mlflow==3.1.1 \
    psycopg2-binary \
    boto3 \
    sqlalchemy \
    pymysql

# Set environment variables (customize these as needed)
ENV MLFLOW_PORT=5000
ENV MLFLOW_HOST=0.0.0.0
ENV BACKEND_STORE_URI=sqlite:///mlflow.db
ENV ARTIFACT_ROOT=/tmp/mlruns

# Expose MLflow UI port
EXPOSE 5000

# Start MLflow server
CMD mlflow server \
    --host ${MLFLOW_HOST} \
    --port ${MLFLOW_PORT} \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${ARTIFACT_ROOT}