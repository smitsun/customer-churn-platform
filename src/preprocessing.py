import os
import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Define features for training
NUMERICAL_FEATURES = ["Tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "Gender", "SeniorCitizen", "Partner", "Dependents", 
    "PhoneService", "MultipleLines", "InternetService", 
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
    "TechSupport", "StreamingTV", "StreamingMovies", 
    "Contract", "PaperlessBilling", "PaymentMethod"
]
TARGET_COLUMN = "Churn"

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw customer data: converts fields to correct types, 
    replaces empty/whitespace characters with NaN, and encodes target.
    """
    df = df.copy()
    
    # 1. Clean TotalCharges
    # Replace empty spaces with NaN
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace(r'^\s*$', np.nan, regex=True)
        # Convert to float
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Impute missing TotalCharges with median or MonthlyCharges * Tenure
        median_total = df["TotalCharges"].median()
        if pd.isna(median_total):
            median_total = 0.0
        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["Tenure"])
        df["TotalCharges"] = df["TotalCharges"].fillna(median_total)
        
    # 2. Clean/Convert SeniorCitizen to object/categorical representation if needed
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
        
    # 3. Encode target Churn: Yes -> 1, No -> 0
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        
    return df

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Constructs the ColumnTransformer pipeline for numerical scaling and categorical encoding.
    """
    # Numerical preprocessor
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical preprocessor
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERICAL_FEATURES),
            ("cat", cat_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop" # Drop customer ID and other unneeded columns
    )
    
    return preprocessor

def get_feature_names(column_transformer: ColumnTransformer) -> list:
    """
    Extracts feature names from fitted ColumnTransformer.
    """
    feature_names = []
    
    # Loop over transformers
    for name, trans, cols in column_transformer.transformers_:
        if name == "remainder" or trans == "drop":
            continue
        # If pipeline, get the last step
        if isinstance(trans, Pipeline):
            step = trans.steps[-1][1]
        else:
            step = trans
            
        if isinstance(step, OneHotEncoder):
            # Get one-hot encoded categories
            names = step.get_feature_names_out(cols)
            feature_names.extend(names)
        else:
            # Numerical features or scaled features
            feature_names.extend(cols)
            
    return feature_names

def fit_and_save_pipeline(df: pd.DataFrame, save_dir: str = None) -> tuple:
    """
    Fits preprocessing pipeline and saves it.
    
    Returns:
        tuple: (preprocessed_features_array, targets_array, fitted_preprocessor)
    """
    if save_dir is None:
        save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    
    preprocessor = build_preprocessing_pipeline()
    
    # Clean the dataset
    df_clean = clean_data(df)
    
    X = df_clean.drop(columns=[TARGET_COLUMN, "CustomerId"], errors="ignore")
    y = df_clean[TARGET_COLUMN] if TARGET_COLUMN in df_clean.columns else None
    
    logger.info("Fitting the preprocessing pipeline...")
    X_processed = preprocessor.fit_transform(X)
    
    # Save the pipeline
    pipeline_path = os.path.join(save_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, pipeline_path)
    logger.info(f"Saved fitted preprocessing pipeline to {pipeline_path}")
    
    return X_processed, y.values if y is not None else None, preprocessor

if __name__ == "__main__":
    # Quick module test
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.data_ingestion import ingest_data
    
    logging.basicConfig(level=logging.INFO)
    try:
        raw_df = ingest_data()
        X_p, y_p, prep = fit_and_save_pipeline(raw_df)
        print(f"Preprocessed X shape: {X_p.shape}")
        print(f"Preprocessed y shape: {y_p.shape}")
        features = get_feature_names(prep)
        print(f"First 10 feature names: {features[:10]}")
    except Exception as e:
        print(f"Preprocessing test failed: {e}")
