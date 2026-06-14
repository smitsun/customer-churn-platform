import os
import pandas as pd
import sqlite3
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def ingest_data(csv_path: str = None, db_path: str = None) -> pd.DataFrame:
    """
    Ingests customer churn data by loading a demographics CSV and a services SQLite database,
    and joining them on the common key 'CustomerId'.
    
    Args:
        csv_path (str, optional): Path to demographics CSV file. Defaults to data/raw/demographics.csv.
        db_path (str, optional): Path to services SQLite database. Defaults to data/raw/services.db.
        
    Returns:
        pd.DataFrame: Merged dataset containing both demographic and services information.
    """
    if csv_path is None:
        csv_path = os.path.join("data", "raw", "demographics.csv")
    if db_path is None:
        db_path = os.path.join("data", "raw", "services.db")
        
    logger.info(f"Starting data ingestion process.")
    
    # 1. Load demographics CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Demographics CSV file not found at: {csv_path}")
        
    logger.info(f"Loading demographics CSV from: {csv_path}")
    df_demographics = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df_demographics)} demographic records. Shape: {df_demographics.shape}")
    
    # 2. Load services DB
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Services database file not found at: {db_path}")
        
    logger.info(f"Loading services database from: {db_path}")
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM services"
        df_services = pd.read_sql_query(query, conn)
        conn.close()
        logger.info(f"Loaded {len(df_services)} service records from DB. Shape: {df_services.shape}")
    except Exception as e:
        logger.error(f"Error reading services database: {str(e)}")
        raise
        
    # 3. Merge data
    logger.info("Merging demographic and service records on 'CustomerId'")
    try:
        merged_df = pd.merge(df_demographics, df_services, on="CustomerId", how="inner")
        logger.info(f"Merged dataset successfully. Total records: {len(merged_df)}, Features count: {merged_df.shape[1]}")
        
        # Verify alignment
        if len(merged_df) < max(len(df_demographics), len(df_services)):
            missing_count = abs(len(df_demographics) - len(merged_df))
            logger.warning(f"Incomplete merge: {missing_count} records did not find matching keys across sources.")
            
        return merged_df
    except Exception as e:
        logger.error(f"Failed to merge data sources: {str(e)}")
        raise

if __name__ == "__main__":
    # Test execution
    try:
        df = ingest_data()
        print("\nIngested DataFrame head:")
        print(df.head(2))
    except Exception as err:
        print(f"Ingestion test failed: {err}")
