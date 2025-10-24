# MLOps Assignment: Productionizing a Classification Model

Github Link: https://github.com/prasadzende/mlops_assignment_2

## Table of Contents

* [Project Structure](#project-structure)

* [1. Pipeline Design and Artifact Tracking](#1-pipeline-design-and-artifact-tracking)

* [2. Containerized Inference API](#2-containerized-inference-api)

* [3. Basic Monitoring and Logging](#3-basic-monitoring-and-logging)

* [Scope and Production Scalability](#scope-and-production-scalability)

* [Key Design Decisions and Trade-offs](#key-design-decisions-and-trade-offs)

This repository contains the solution for the MLOps skill assignment. The goal was to productionize a simple ML classification model by implementing a basic CI/CD pipeline for training, artifact tracking, containerized inference API deployment, and basic monitoring/logging.

The entire workflow is managed via a `Makefile` and automated through a GitHub Actions workflow for seamless execution and reproducibility.

## Project Structure

The project is organized to separate model training logic from the serving API deployment logic.

```
MLOPS_ASSIGNMENT_2/
├── .github/
│   └── workflows/
│       └── main.yaml     # CI/CD pipeline (Training -> Artifacts -> Docker Build)
├── infra/
│   └── Dockerfile        # Dockerfile for the mlflow server
├── src/
│   ├── api/
│   │   └── app.py        # FastAPI application for inference
│   └── model_src/
│       └── train.py      # Original data scientist training script (modified for MLflow)
├── Makefile              # Commands for local execution (build, run, train, test)
├── requirements.txt      # Python dependencies (generated via pdm export)
└── README.md             # This document

```

## 1. Pipeline Design and Artifact Tracking

The ML pipeline is implemented using a combination of a **`Makefile`** (for local execution) and **GitHub Actions** (for CI/CD automation). **MLflow** is used for artifact tracking, ensuring reproducibility and centralized metadata storage.

### Technology Choices:

| Component | Tool | Rationale | 
 | ----- | ----- | ----- | 
| **Pipeline Orchestration** | GitHub Actions | Simple, free, and integrated CI/CD for automating the training run upon code changes. | 
| **Artifact Tracking** | MLflow | Industry-standard tool for logging run parameters, metrics (accuracy, loss), and model binaries, fulfilling the reproducibility and artifact tracking requirements. | 
| **Code Management** | `Makefile` | Simplifies complex commands into easy-to-run targets (e.g., `make train`, `make build`), providing a clean local developer experience. |

### How to Run the Training Pipeline (Local)

1. **Set up MLflow:** Ensure you have an MLflow tracking server running or configure the local path where MLflow will store run data (the default is a local `mlruns` folder).

2. **Execute the `train` target:** This runs `src/model_src/train.py`, which wraps the training logic with MLflow tracking.

#### Primary command: Assumes MLflow Tracking Server is configured and supports model registration.

```
make train
```

**Alternative for Local Setup:** If an external MLflow Tracking Server is not configured, use the following target to avoid errors related to model registration permissions/connection:

#### Failsafe command: Runs the script using local MLflow, skipping explicit model registration.

```
make train-no-registration
```

#### **CI/CD Workflow Summary (`.github/workflows/main.yaml`)**

The full automated pipeline, triggered on push and pull request, executes the following sequence:

* **Setup:** Checks out code, sets up Python 3.11, and installs all dependencies (`make setup`, `make install`).

* **Training & Artifacts:** Runs `make train-no-registration` to train the model and log artifacts locally (within the runner).

* **Containerization:** Builds the inference Docker image (`make build-docker`) using the trained model artifact.

* **Testing:** Starts the container (`make run-docker`) and performs an end-to-end integration test against the `/predict` endpoint (`make test-endpoint`).

* **Cleanup:** Stops and removes the Docker container (`make stop-docker`) regardless of test success/failure.

* **Deployment (Conditional):** If running on the `main` branch, logs into Docker Hub and pushes the final, tested image (`make push-docker`).

**What is Tracked:**

* **Parameters:** Hyperparameters like `random_state`.

* **Metrics:** `Accuracy` score on the test set.

* **Artifacts:** The serialized model file (`iris_model.pkl`) is logged and stored by MLflow.

## 2. Containerized Inference API

The trained model is deployed as a high-performance REST API using **FastAPI** and **Uvicorn**, which is then packaged into a production-ready **Docker** container.

### Technology Choices:

| Component | Tool | Rationale | 
 | ----- | ----- | ----- | 
| **Web Framework** | FastAPI | Provides extremely high performance (asynchronous), automatic interactive documentation (Swagger UI), and native data validation using Pydantic, which improves data contract clarity. | 
| **Containerization** | Docker | Ensures the inference environment is consistent, isolated, and easily scalable across different environments (local, staging, production). | 

### How to Build and Run the Inference Container

The Inference API uses the model artifact generated in the training step. For local simulation, we assume `iris_model.pkl` is available in the `src/api/` directory (or can be manually copied from the MLflow artifact store).

#### 1. Build the Docker Image

To build the image, navigate to the src/api/ directory and execute the docker build command:
```
cd src/api/
docker build -t iris-app:latest .
```

#### 2. Run the Container
This command starts the container and maps the internal port 8000 (used by FastAPI) to the host's port 8000.
```
docker run -d -p 8000:8000 iris-app:latest
```

The API will be available at `http://localhost:8000/docs` (for Swagger UI) and the prediction endpoint will be at `http://localhost:8000/predict`.

### Prediction Request Example (cURL):
```
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}'
```

## 3. Basic Monitoring and Logging

Basic monitoring is implemented directly within the **FastAPI** inference application (`src/api/app.py`) using structured logging.

### What is Monitored and How:

| Type of Monitoring | Implementation Method | Location/Format | 
 | ----- | ----- | ----- | 
| **Request/Response Logging** | Structured JSON logging in `app.py`'s prediction route. | Standard Output (`stdout`) of the Docker container. | 
| **Latency Tracking** | Timestamps are logged on request start and finish. | Included in the JSON log structure. | 
| **Simulated Drift Tracking** | Logging **input summary statistics** (mean, std dev) for the current batch of prediction features. | JSON log structure under the `input_stats` key. | 

### Example Log Entry (Simulated)

In this project the application logs the input request and output to local `prediction.log` file.


In a real-world scenario, The application logs a JSON object to `stdout` for every prediction request. This log stream would be ingested by a centralized logging system (e.g., Elastic Stack, Datadog, Prometheus/Loki).

```
{
  "timestamp": "2025-10-24 15:26:57,895",
  "level": "info",
  "message": "Inference successful.",
  "module": "app",
  "endpoint": "/predict",
  "status": "success",
  "latency_ms": 0.36,
  "request_id": "1761299817895314",
  "input": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  },
  "prediction": {
    "class": 0,
    "confidence": 0.9766
  },
  "input_stats": {
    "sepal_length": {
      "mean": 5.84,
      "std": 0.82
    },
    "sepal_width": {
      "mean": 3.05,
      "std": 0.43
    },
    "petal_length": {
      "mean": 3.76,
      "std": 1.76
    },
    "petal_width": {
      "mean": 1.2,
      "std": 0.76
    }
  }
}

```

### Scope and Production Scalability

This project leverages `open-source` and `local-first` tools (MLflow with local artifact storage, GitHub Actions, Docker on a local host) to meet the proof-of-concept requirements for the MLOps pipeline.

For a true, high-volume production environment, the following scalable infrastructure would be implemented:

- **Model Registry/Tracking**: MLflow deployed with a centralized PostgreSQL/MySQL backend and S3/GCS/Azure Blob Storage for artifact storage.

- **Orchestration/CI/CD**: Using a managed service like Cloud Build/Azure Pipelines/AWS CodePipeline for CI, and Kubernetes (EKS/GKE/AKS) orchestrated by tools like Argo CD for CD and serving.

- **Inference Serving**: Deploying the Docker container to a managed Kubernetes cluster with Knative for autoscaling, or using a serverless option like AWS Lambda (with containers) or Google Cloud Run.

- **Monitoring/Logging**: Integrating structured logs into a centralized system like ELK Stack/Grafana Loki, and using Prometheus/Grafana for metric scraping and dashboarding.

### Key Design Decisions and Trade-offs

- #### Model Artifact Separation

**Decision**: The training script (train.py) is placed in model_src, and the inference code (app.py) is in api.

**Benefit**: This separation aligns with MLOps best practices, where the training environment (often a larger, GPU-enabled VM) is distinct from the low-latency, small serving environment. The serving container only needs the final .pkl file, not all the training dependencies.

- #### CI/CD Simplification

**Decision**: Using shell scripts/Makefile targets combined with GitHub Actions instead of a full orchestration tool (Airflow/Prefect).

**Trade-off**: This is simpler and quicker to set up for a proof-of-concept. For a full-scale pipeline with dependency-based scheduling, retries, and complex fan-out/fan-in tasks, a dedicated orchestrator would be the correct choice.
