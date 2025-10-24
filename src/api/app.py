import time
import joblib
import json
import logging
from typing import Dict, Union, Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
import os
import mlflow
import requests
import numpy as np

# --- 1. Structured JSON Logging Setup ---

# Define log file path
LOG_FILEPATH = "prediction.log"

# Hardcoded baseline statistics derived from training data for drift simulation
BASELINE_STATS = {
    "sepal_length": {"mean": 5.84, "std": 0.82},
    "sepal_width": {"mean": 3.05, "std": 0.43},
    "petal_length": {"mean": 3.76, "std": 1.76},
    "petal_width": {"mean": 1.20, "std": 0.76},
}

class JsonFormatter(logging.Formatter):
    """Custom formatter to output log records as single-line JSON."""
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "module": record.module,
            **getattr(record, 'log_data', {}) # Use 'log_data' for custom fields
        }
        return json.dumps(log_data)

# Configure logger to output JSON to stdout (container's default log stream) AND to a file
logger = logging.getLogger("inference_logger")
# Clear any existing handlers for a clean setup
if logger.hasHandlers():
    logger.handlers.clear()

# Define the formatter instance
json_formatter = JsonFormatter()

# 1. Stream Handler (to stdout/console) - Standard for container logs
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(json_formatter)
logger.addHandler(stream_handler)

# 2. File Handler (to local file) - For persistent local log storage
file_handler = logging.FileHandler(LOG_FILEPATH)
file_handler.setFormatter(json_formatter)
logger.addHandler(file_handler)

logger.setLevel(logging.INFO)

class IrisRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)

# Define the Iris species mapping
SPECIES_MAPPING = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

app = FastAPI(title="Iris Classification Inference API")

def is_mlflow_available(tracking_uri="http://localhost:5005"):
    try:
        response = requests.get(f"{tracking_uri}/health")
        return response.status_code == 200
    except:
        return False

def load_model():
    # If MLflow is available, try loading from there
    if is_mlflow_available():
        try:
            logger.info("Attempting to load model from MLflow.")
            mlflow.set_tracking_uri("http://localhost:5005")
            
            # Using 'Production' stage requires registering the model in MLflow
            model_name = "iris_classifier_model"
            model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
            logger.info("Successfully loaded model from MLflow.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {str(e)}")
    
    # Fallback to local pickle file (or MockModel if missing)
    model_path = "iris_model.pkl"
    if not os.path.exists(model_path):
        logger.warning(f"Model file {model_path} not found. Using MockModel.")
        class MockModel:
            def predict(self, data): return np.array([0])
            def predict_proba(self, data): return np.array([[0.99, 0.01, 0.00]])
        return MockModel()
    
    model = joblib.load(model_path)
    logger.info("Successfully loaded model from local file.")
    return model

# Load model when app starts
model = load_model()

@app.post("/predict", response_model=Dict[str, Union[int, float, str]])
async def predict(data: IrisRequest):
    # --- 2. Start Time and Request ID ---
    start_time = time.time()
    request_id = str(int(start_time * 1000000)) # High resolution ID

    try:
        features = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]])

        # Model prediction
        prediction = model.predict(features)
        pred_class = int(prediction[0])
        pred_label = SPECIES_MAPPING.get(pred_class, "unknown")
        
        # Get prediction probability
        prob = round(float(model.predict_proba(features).max()), 4)

        # --- 3. Monitoring & Drift Logging ---
        
        # Calculate latency
        end_time = time.time()
        latency_ms = round((end_time - start_time) * 1000, 2)
        
        # Log all required monitoring data in a structured way
        log_data = {
            "endpoint": "/predict",
            "status": "success",
            "latency_ms": latency_ms,
            "request_id": request_id,
            "input": data.model_dump(),
            "prediction": {
                "class": pred_class,
                "label": pred_label,
                "confidence": prob
            },
            # Logs the pre-defined statistics for simulated drift analysis (as per README)
            "input_stats": BASELINE_STATS 
        }


        logger.info("Inference successful.", extra={'log_data': log_data})
        
        return {
            "status": "success",
            "prediction": pred_label,
            "probability": prob
        }
        
    except Exception as e:
        end_time = time.time()
        latency_ms = round((end_time - start_time) * 1000, 2)
        
        error_log_data = {
            "endpoint": "/predict",
            "status": "error",
            "latency_ms": latency_ms,
            "request_id": request_id,
            "error_message": str(e)
        }
        logger.error(f"Prediction error: {str(e)}", extra={'log_data': error_log_data})
        
        file_handler.flush()

        raise HTTPException(status_code=500, detail=str(e))
