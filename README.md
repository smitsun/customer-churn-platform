# Customer Churn Prediction Platform

This is a modular, production-grade Customer Churn Prediction Platform. It provides a complete end-to-end machine learning pipeline, a RESTful API backend built with FastAPI, and an interactive business analytics dashboard built with Streamlit.

## Platform Features

1. **Multi-Source Data Ingestion**: Loads and merges customer demographics from CSV and service details from a SQLite database.
2. **Robust Preprocessing & Feature Engineering**: Handles missing values, performs one-hot encoding for categorical factors, scales numeric columns, and saves the pipeline transformer via joblib.
3. **Multi-Model Training & Optimization**: Compares Logistic Regression, Random Forest, and XGBoost models, evaluating accuracy, precision, recall, and ROC-AUC. Saves the overall best-performing model.
4. **Diagnostic Metrics**: Automatically generates confusion matrices, ROC curve charts, and feature importance analyses.
5. **FastAPI Inference Server**: Exposes endpoints for high-throughput prediction requests (`/predict`) and service readiness checking (`/health`).
6. **Streamlit Analytics Dashboard**: Displays high-level KPIs, predictive insight charts, and an interactive interface for predicting churn risk on individual customer profiles.
7. **Docker Containerization**: Full Docker and Docker Compose files to deploy the backend and frontend seamlessly.

---

## Folder Architecture

```
├── data/
│   ├── raw/                 # Demographics CSV & Services SQLite database
│   └── processed/           # Processed datasets (optional cache)
├── models/                  # Joblib files for preprocessor, best model, and charts
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py   # Merges CSV and database tables
│   ├── preprocessing.py    # Standard scaling and hot encoding pipelines
│   ├── train.py            # Executes train-test splits and fits models
│   ├── evaluate.py         # Visualizes metrics (ROC, Confusion Matrix, importance)
│   ├── api.py              # FastAPI endpoints
│   └── dashboard.py        # Streamlit graphical interface
├── scripts/
│   └── generate_data.py    # Synthetic customer database generator
├── tests/
│   └── test_api.py         # Pytest API verification suite
├── Dockerfile               # Multi-port application dockerfile
├── docker-compose.yml       # Orchestrates FastAPI & Streamlit services
├── requirements.txt         # Core dependencies
└── README.md                # System documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Make sure to tick **"Add python.exe to PATH"** during installation)

### 1. Create a Virtual Environment and Install Dependencies
Open your terminal in this directory (`c:\Projects\1`) and run:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows Powershell:
.\venv\Scripts\Activate.ps1
# On Windows CMD:
.\venv\Scripts\activate.bat
# On macOS/Linux:
source venv/bin/activate

# Install required libraries
pip install -r requirements.txt
```

---

## How to Run the Platform

### Step 1: Generate Synthetic Data
First, run the data generator to create a synthetic raw customer dataset distributed across a CSV file and a SQLite database:
```bash
python scripts/generate_data.py
```

### Step 2: Train the Models & Generate Visuals
Execute the training script to train Logistic Regression, Random Forest, and XGBoost classifiers. This will automatically select the best model, export it, and write performance metrics:
```bash
python src/train.py
```

Next, run the evaluation script to generate evaluation charts (ROC Curves, Confusion Matrix, and Feature Importance):
```bash
python src/evaluate.py
```

### Step 3: Launch the FastAPI Backend
Start the FastAPI server using Uvicorn:
```bash
uvicorn src.api:app --reload
```
The API documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Step 4: Run the Streamlit Dashboard
In a separate terminal window (with the virtual environment activated), start the dashboard:
```bash
streamlit run src/dashboard.py
```
Your browser should open to [http://localhost:8501](http://localhost:8501) (or check output if port is redirected).

---

## Running with Docker Compose

If you have Docker Desktop installed, you can skip local installations and start both the backend API and frontend dashboard with a single command:

```bash
# Start container cluster
docker-compose up --build
```

- **FastAPI API** will be accessible at: [http://localhost:8000](http://localhost:8000)
- **Streamlit Dashboard** will be accessible at: [http://localhost:8505](http://localhost:8505)

---

## Running Automated Tests

Run the integration test suite using `pytest`:
```bash
pytest tests/
```
