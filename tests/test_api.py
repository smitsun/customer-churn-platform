import os
import sys
from fastapi.testclient import TestClient

# Add parent dir to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.api import app

client = TestClient(app)

def test_health_endpoint():
    """Tests that the health endpoint returns a successful status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "preprocessor_ready" in data
    assert "model_ready" in data

def test_prediction_endpoint():
    """
    Tests the prediction endpoint. Note: requires preprocessor and best_model to be loaded.
    If models are not trained yet, it might return 503, which is also an expected API code for config issues.
    """
    sample_payload = [
        {
            "Gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "Tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.0,
            "TotalCharges": 840.0
        }
    ]
    
    response = client.post("/predict", json=sample_payload)
    
    # If trained, should return predictions
    # If not trained, should return 503
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        pred = data["predictions"][0]
        assert "churn_probability" in pred
        assert "churn_prediction" in pred
        assert "churn_label" in pred
        assert pred["churn_label"] in ["Yes", "No"]
        assert 0.0 <= pred["churn_probability"] <= 1.0
