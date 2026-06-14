FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for xgboost and other compiled packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose API (8000) and Streamlit Dashboard (8501) ports
EXPOSE 8000
EXPOSE 8501

# Default startup command (runs the API)
# Can be overridden for dashboard or other scripts
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
