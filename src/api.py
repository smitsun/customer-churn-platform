import os
import logging
import joblib
import json
import pandas as pd
from typing import List, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add parent dir to path for local runs
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing import clean_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

# Paths
MODELS_DIR = "models"
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# Global variables to store model & preprocessor
preprocessor = None
best_model = None
metadata = None

def startup_event():
    """Loads preprocessor, model, and metadata on API startup."""
    global preprocessor, best_model, metadata
    
    logger.info("Initializing API: Loading model and preprocessing assets...")
    
    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        logger.info("Loaded preprocessor successfully.")
    else:
        logger.warning(f"Preprocessor not found at {PREPROCESSOR_PATH}. Train models first.")
        
    if os.path.exists(BEST_MODEL_PATH):
        best_model = joblib.load(BEST_MODEL_PATH)
        logger.info("Loaded best model successfully.")
    else:
        logger.warning(f"Best model not found at {BEST_MODEL_PATH}. Train models first.")
        
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metadata = json.load(f)
        logger.info("Loaded metrics metadata successfully.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_event()
    yield

app = FastAPI(
    title="Customer Churn Prediction Platform API",
    description="FastAPI backend for predicting customer churn probability and class.",
    version="1.0.0",
    lifespan=lifespan
)

# Input Pydantic Model
class CustomerPayload(BaseModel):
    Gender: str = Field(..., json_schema_extra={"example": "Female"})
    SeniorCitizen: int = Field(..., json_schema_extra={"example": 0})
    Partner: str = Field(..., json_schema_extra={"example": "Yes"})
    Dependents: str = Field(..., json_schema_extra={"example": "No"})
    Tenure: int = Field(..., json_schema_extra={"example": 1})
    PhoneService: str = Field(..., json_schema_extra={"example": "No"})
    MultipleLines: str = Field(..., json_schema_extra={"example": "No phone service"})
    InternetService: str = Field(..., json_schema_extra={"example": "DSL"})
    OnlineSecurity: str = Field(..., json_schema_extra={"example": "No"})
    OnlineBackup: str = Field(..., json_schema_extra={"example": "Yes"})
    DeviceProtection: str = Field(..., json_schema_extra={"example": "No"})
    TechSupport: str = Field(..., json_schema_extra={"example": "No"})
    StreamingTV: str = Field(..., json_schema_extra={"example": "No"})
    StreamingMovies: str = Field(..., json_schema_extra={"example": "No"})
    Contract: str = Field(..., json_schema_extra={"example": "Month-to-month"})
    PaperlessBilling: str = Field(..., json_schema_extra={"example": "Yes"})
    PaymentMethod: str = Field(..., json_schema_extra={"example": "Electronic check"})
    MonthlyCharges: float = Field(..., json_schema_extra={"example": 29.85})
    TotalCharges: float = Field(..., json_schema_extra={"example": 29.85})

class PredictionResult(BaseModel):
    churn_probability: float
    churn_prediction: int
    churn_label: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResult]

@app.get("/health")
def health():
    """Health check endpoint that returns API status and model metadata."""
    is_ready = preprocessor is not None and best_model is not None
    
    return {
        "status": "healthy" if is_ready else "not_configured",
        "best_model_name": metadata.get("best_model_name", "None") if metadata else "None",
        "best_model_roc_auc": metadata.get("best_score_roc_auc", 0.0) if metadata else 0.0,
        "preprocessor_ready": preprocessor is not None,
        "model_ready": best_model is not None
    }

@app.post("/predict", response_model=BatchPredictionResponse)
def predict(payload: List[CustomerPayload]):
    """
    Predicts churn probability and labels for a batch of customers.
    """
    global preprocessor, best_model
    
    if preprocessor is None or best_model is None:
        # Try loading on demand in case they were generated post-startup
        startup_event()
        if preprocessor is None or best_model is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not trained or loaded. Run model training pipeline first."
            )
            
    try:
        # 1. Convert payload list to pandas DataFrame
        records = [item.model_dump() for item in payload]
        df_input = pd.DataFrame(records)
        
        # 2. Clean input
        df_clean = clean_data(df_input)
        
        # 3. Transform using preprocessor
        # Note: drop columns that are not features if they exist
        X_processed = preprocessor.transform(df_clean)
        
        # 4. Run prediction
        probs = best_model.predict_proba(X_processed)[:, 1]
        preds = best_model.predict(X_processed)
        
        # 5. Format results
        results = []
        for prob, pred in zip(probs, preds):
            results.append(PredictionResult(
                churn_probability=float(prob),
                churn_prediction=int(pred),
                churn_label="Yes" if pred == 1 else "No"
            ))
            
        return BatchPredictionResponse(predictions=results)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
