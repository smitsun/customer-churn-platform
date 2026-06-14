import os
import json
import logging
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

# Add parent dir to path for local runs
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_ingestion import ingest_data
from src.preprocessing import fit_and_save_pipeline, clean_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def train_and_evaluate_models(models_dir: str = "models") -> dict:
    """
    Trains and compares multiple machine learning models on customer churn,
    saving the best performing model.
    
    Args:
        models_dir (str): Directory where preprocessing pipeline and models will be saved.
        
    Returns:
        dict: Test metrics for all evaluated models.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Ingest Data
    raw_df = ingest_data()
    
    # 2. Clean & Preprocess
    # Preprocessor fits and saves itself inside fit_and_save_pipeline
    X_processed, y, preprocessor = fit_and_save_pipeline(raw_df, save_dir=models_dir)
    
    # 3. Train-Test Split (stratified for class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Split data into train ({X_train.shape[0]} rows) and test ({X_test.shape[0]} rows)")
    
    # 4. Define models to train
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }
    
    metrics_summary = {}
    best_score = -1.0
    best_model_name = ""
    best_model_obj = None
    
    # 5. Train & Evaluate each
    for name, model in models.items():
        logger.info(f"Training model: {name}...")
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        metrics = {
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "ROC-AUC": float(auc)
        }
        metrics_summary[name] = metrics
        
        logger.info(
            f"{name} Results - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, ROC-AUC: {auc:.4f}"
        )
        
        # Save individual models for safety/reference
        model_filename = f"{name.lower()}_model.joblib"
        joblib.dump(model, os.path.join(models_dir, model_filename))
        
        # Best model selection is based on ROC-AUC (standard metric for classification probability rank)
        if auc > best_score:
            best_score = auc
            best_model_name = name
            best_model_obj = model
            
    logger.info(f"Best Model Selected: {best_model_name} with ROC-AUC = {best_score:.4f}")
    
    # 6. Save best model and metadata
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model_obj, best_model_path)
    logger.info(f"Saved best model to {best_model_path}")
    
    metadata = {
        "best_model_name": best_model_name,
        "best_score_roc_auc": best_score,
        "metrics": metrics_summary
    }
    
    metadata_path = os.path.join(models_dir, "metrics.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved metrics comparison to {metadata_path}")
    
    return metadata

if __name__ == "__main__":
    try:
        results = train_and_evaluate_models()
        print("\nFinal Model Comparison:")
        print(json.dumps(results, indent=2))
    except Exception as e:
        logger.error(f"Training run failed: {e}")
