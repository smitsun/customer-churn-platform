import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Add parent dir to path for local runs
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_ingestion import ingest_data
from src.preprocessing import clean_data, get_feature_names

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_evaluation_visualizations(models_dir: str = "models") -> None:
    """
    Loads trained models and data, evaluates their performance,
    and generates evaluation charts (Confusion Matrix, ROC Curve, and Feature Importance).
    """
    logger.info("Starting model evaluation and visualization generation...")
    
    # 1. Load data
    raw_df = ingest_data()
    df_clean = clean_data(raw_df)
    
    # Load preprocessor
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found. Please train models first.")
        
    preprocessor = joblib.load(preprocessor_path)
    
    # Preprocess
    X = df_clean.drop(columns=["Churn", "CustomerId"], errors="ignore")
    y = df_clean["Churn"]
    X_processed = preprocessor.transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get feature names
    feature_names = get_feature_names(preprocessor)
    
    # 2. Plot ROC Curves Comparison
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    
    model_names = ["LogisticRegression", "RandomForest", "XGBoost"]
    colors = {"LogisticRegression": "#1f77b4", "RandomForest": "#2ca02c", "XGBoost": "#ff7f0e"}
    
    best_model_name = "XGBoost" # Fallback if not specified in JSON
    # Try to load metadata
    metrics_path = os.path.join(models_dir, "metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, "r") as f:
            meta = json.load(f)
            best_model_name = meta.get("best_model_name", "XGBoost")
            
    for name in model_names:
        model_path = os.path.join(models_dir, f"{name.lower()}_model.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            label = f"{name} (AUC = {roc_auc:.3f})"
            if name == best_model_name:
                label += " [BEST]"
                
            plt.plot(fpr, tpr, color=colors[name], lw=2.5, label=label)
            
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    
    roc_plot_path = os.path.join(models_dir, "roc_curve_comparison.png")
    plt.tight_layout()
    plt.savefig(roc_plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved ROC comparison plot to {roc_plot_path}")
    
    # 3. Load best model and generate Confusion Matrix
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    if os.path.exists(best_model_path):
        best_model = joblib.load(best_model_path)
        y_pred = best_model.predict(X_test)
        
        # Calculate Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"],
            annot_kws={"size": 14, "weight": "bold"}
        )
        plt.xlabel("Predicted Labels", fontsize=12)
        plt.ylabel("True Labels", fontsize=12)
        plt.title(f"Confusion Matrix ({best_model_name})", fontsize=14, fontweight="bold")
        
        cm_plot_path = os.path.join(models_dir, "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_plot_path, dpi=150)
        plt.close()
        logger.info(f"Saved Confusion Matrix to {cm_plot_path}")
        
        # 4. Generate Feature Importance for Best Model (if supported)
        has_importance = hasattr(best_model, "feature_importances_") or hasattr(best_model, "coef_")
        
        if has_importance:
            if hasattr(best_model, "feature_importances_"):
                importances = best_model.feature_importances_
            else:
                # Use absolute coefficients for Logistic Regression
                importances = np.abs(best_model.coef_[0])
                
            # Create a DataFrame of feature importances
            fi_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)
            
            # Save CSV file for Streamlit to parse
            fi_df.to_csv(os.path.join(models_dir, "feature_importances.csv"), index=False)
            
            # Plot top 15 features
            top_n = min(15, len(fi_df))
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=fi_df.head(top_n), x="Importance", y="Feature", 
                hue="Feature", legend=False, palette="viridis"
            )
            plt.title(f"Top {top_n} Feature Importance ({best_model_name})", fontsize=14, fontweight="bold")
            plt.xlabel("Importance Score", fontsize=12)
            plt.ylabel("Feature", fontsize=12)
            
            fi_plot_path = os.path.join(models_dir, "feature_importance.png")
            plt.tight_layout()
            plt.savefig(fi_plot_path, dpi=150)
            plt.close()
            logger.info(f"Saved Feature Importance plot to {fi_plot_path}")
            
    else:
        logger.warning("Best model file not found; skipping confusion matrix and feature importance plot.")
        
    logger.info("Evaluation visualizations generated successfully.")

if __name__ == "__main__":
    try:
        generate_evaluation_visualizations()
    except Exception as e:
        logger.error(f"Evaluation visualization generation failed: {e}")
