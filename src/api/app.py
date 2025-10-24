from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Dict, Union
import joblib
import logging
import os
import mlflow
import requests
import numpy as np

# Define input request schema
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define the Iris species mapping
SPECIES_MAPPING = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# Setup logging
log_filepath = "prediction.log"
logging.basicConfig(
    filename=log_filepath,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

app = FastAPI()

def is_mlflow_available(tracking_uri="http://localhost:5005"):
    try:
        response = requests.get(f"{tracking_uri}/health")
        return response.status_code == 200
    except:
        return False

def load_model():
    # Try loading from MLflow first
    if is_mlflow_available():
        try:
            logger.info("Attempting to load model from MLflow")
            mlflow.set_tracking_uri("http://localhost:5001")
            
            # Replace these with your actual values
            model_name = "iris_classifier_model"
            #stage = "Production"  # or "Staging", "None", etc.
            
            model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
            logger.info("Successfully loaded model from MLflow")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {str(e)}")
    
    # Fallback to local pickle file
    logger.info("Falling back to local pickle file")
    model_path = "iris_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    model = joblib.load(model_path)
    logger.info("Successfully loaded model from local file")
    return model

# Load model when app starts
model = load_model()

@app.post("/predict", response_model=Dict[str, Union[int, float, str]])
async def predict(request: Request, data: IrisRequest):
    try:
        # Log request
        logger.info(f"Incoming request {data.json()}")

        # Convert request data to numpy array
        features = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]])

        # Model prediction
        prediction = model.predict(features)
        # Convert numpy.int64 to standard Python int
        pred_label = SPECIES_MAPPING[int(prediction[0])]

        # Get prediction probability
        prob = round(float(model.predict_proba(features).max()), 4)

        # Log response
        logger.info(f"Prediction response: {pred_label}")

        return {
            "status": "success",
            "prediction": pred_label,
            "probability": prob
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))