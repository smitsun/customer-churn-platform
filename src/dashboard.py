import os
import sys
# Add parent dir to path for local runs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import pandas as pd
import requests
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction Portal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism UI)
st.markdown("""
<style>
    .main {
        background-color: #0F111A;
        color: #E2E8F0;
    }
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .kpi-card h3 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .kpi-card p {
        margin: 5px 0 0 0;
        color: #94A3B8;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .churn-alert-high {
        background-color: rgba(239, 68, 68, 0.2);
        border: 1px solid rgb(239, 68, 68);
        color: #FCA5A5;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
        font-weight: bold;
    }
    .churn-alert-low {
        background-color: rgba(16, 185, 129, 0.2);
        border: 1px solid rgb(16, 185, 129);
        color: #A7F3D0;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Configurations
API_URL = "http://localhost:8000"
MODELS_DIR = "models"
DB_PATH = os.path.join("data", "raw", "services.db")
CSV_PATH = os.path.join("data", "raw", "demographics.csv")

@st.cache_data
def load_raw_stats():
    """Loads database and CSV to calculate statistics for KPI cards."""
    try:
        if os.path.exists(DB_PATH) and os.path.exists(CSV_PATH):
            conn = sqlite3.connect(DB_PATH)
            df_s = pd.read_sql_query("SELECT * FROM services", conn)
            conn.close()
            df_d = pd.read_csv(CSV_PATH)
            df = pd.merge(df_d, df_s, on="CustomerId")
            
            # Compute stats
            total_customers = len(df)
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(r'^\s*$', "0", regex=True), errors="coerce").fillna(0.0)
            avg_monthly = df["MonthlyCharges"].mean()
            avg_tenure = df["Tenure"].mean()
            
            # Churn rate
            churn_count = (df["Churn"] == "Yes").sum()
            churn_rate = (churn_count / total_customers) * 100
            
            return {
                "total_customers": total_customers,
                "churn_rate": churn_rate,
                "avg_monthly": avg_monthly,
                "avg_tenure": avg_tenure,
                "df": df
            }
    except Exception as e:
        st.warning(f"Could not load database statistics: {e}")
    return None

def check_api_health():
    """Checks if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=1.5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

def predict_churn_api(payload):
    """Sends prediction request to FastAPI backend."""
    try:
        response = requests.post(f"{API_URL}/predict", json=[payload], timeout=3)
        if response.status_code == 200:
            res_data = response.json()
            return res_data["predictions"][0]
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"API predict error: {e}")
    return None

def predict_churn_local(payload):
    """Fallback: Predicts using local joblib models directly if API is down."""
    try:
        # Load local model and preprocessor
        prep_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
        model_path = os.path.join(MODELS_DIR, "best_model.joblib")
        
        if os.path.exists(prep_path) and os.path.exists(model_path):
            preprocessor = joblib.load(prep_path)
            model = joblib.load(model_path)
            
            # Format payload as dataframe
            df = pd.DataFrame([payload])
            
            # Preprocess
            from src.preprocessing import clean_data
            df_clean = clean_data(df)
            X_processed = preprocessor.transform(df_clean)
            
            # Predict
            prob = model.predict_proba(X_processed)[0, 1]
            pred = model.predict(X_processed)[0]
            
            return {
                "churn_probability": float(prob),
                "churn_prediction": int(pred),
                "churn_label": "Yes" if pred == 1 else "No"
            }
    except Exception as e:
        st.error(f"Local fallback predictor error: {e}")
    return None


# Sidebar Setup
st.sidebar.markdown("<h1 style='text-align: center; color: #6366F1;'>ChurnPortal 📊</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go To",
    ["Overview & Insights", "Single Customer Predictor", "Model Diagnostic Hub"]
)

api_status = check_api_health()
st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
if api_status:
    st.sidebar.success(f"FastAPI Backend: Online")
    st.sidebar.info(f"Active Model: **{api_status.get('best_model_name')}**")
else:
    st.sidebar.warning("FastAPI Backend: Offline\n(Running in local fallback mode)")

# Load stats
stats = load_raw_stats()

# ==============================================================================
# Page 1: Overview & Insights
# ==============================================================================
if page == "Overview & Insights":
    st.markdown("<h1 style='color: #6366F1;'>Customer Churn Insights</h1>", unsafe_allow_html=True)
    st.write("Understand the key indicators driving customer churn across the business.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    if stats:
        # 1. KPI Cards Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #6366F1;">
                <p>TOTAL CUSTOMERS</p>
                <h3 style="color: #818CF8;">{stats['total_customers']:,}</h3>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #EF4444;">
                <p>CHURN RATE</p>
                <h3 style="color: #F87171;">{stats['churn_rate']:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #10B981;">
                <p>AVG MONTHLY CHARGE</p>
                <h3 style="color: #34D399;">${stats['avg_monthly']:.2f}</h3>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #F59E0B;">
                <p>AVG TENURE (MONTHS)</p>
                <h3 style="color: #FBBF24;">{stats['avg_tenure']:.1f}m</h3>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # 2. Charts Row
        c1, c2 = st.columns(2)
        df = stats["df"]
        
        with c1:
            st.subheader("Contract Type vs. Customer Churn")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0F111A')
            ax.set_facecolor('#1E293B')
            
            sns.countplot(data=df, x="Contract", hue="Churn", palette={"Yes": "#EF4444", "No": "#10B981"}, ax=ax)
            ax.set_title("Churn count by Contract Type", color="white", fontweight="bold")
            ax.set_xlabel("Contract Type", color="white")
            ax.set_ylabel("Customer Count", color="white")
            ax.tick_params(colors="white")
            ax.legend(title="Churned?", labelcolor="white", facecolor="#1E293B")
            
            st.pyplot(fig)
            st.caption("Month-to-month contracts exhibit disproportionately high churn rates compared to long-term commitments.")
            
        with c2:
            st.subheader("Distribution of Tenure by Churn Status")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0F111A')
            ax.set_facecolor('#1E293B')
            
            sns.kdeplot(data=df, x="Tenure", hue="Churn", fill=True, common_norm=False, palette={"Yes": "#EF4444", "No": "#10B981"}, alpha=0.5, ax=ax)
            ax.set_title("Tenure Kernel Density Estimate", color="white", fontweight="bold")
            ax.set_xlabel("Tenure (months)", color="white")
            ax.set_ylabel("Density", color="white")
            ax.tick_params(colors="white")
            
            st.pyplot(fig)
            st.caption("New customers (low tenure < 12 months) represent the highest risk bracket for customer churn.")

    else:
        st.info("No raw database/CSV records found. Run synthetic data generator to load dashboard charts.")

# ==============================================================================
# Page 2: Single Customer Predictor
# ==============================================================================
elif page == "Single Customer Predictor":
    st.markdown("<h1 style='color: #6366F1;'>Predict Churn Risk</h1>", unsafe_allow_html=True)
    st.write("Enter details of an individual customer to predict their risk of churning.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Form layout
    with st.form("churn_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            
            st.subheader("Account Details")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer", "Credit card"
            ])
            
        with col2:
            st.subheader("Services Subscribed")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_sec = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
            online_bak = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
            dev_prot = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
            tech_sup = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
            streaming_mov = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
            
        with col3:
            st.subheader("Charges & Engagement")
            tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
            monthly = st.slider("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=70.0, step=0.5)
            # Calculated total charges
            total = monthly * tenure
            st.info(f"Total Charges (Calculated): **${total:.2f}**")
            
        submit_btn = st.form_submit_button("Compute Churn Probability")
        
    if submit_btn:
        # Prepare JSON payload
        payload = {
            "Gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": partner,
            "Dependents": dependents,
            "Tenure": int(tenure),
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bak,
            "DeviceProtection": dev_prot,
            "TechSupport": tech_sup,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_mov,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly),
            "TotalCharges": float(total)
        }
        
        # Run prediction (try API first, then local fallback)
        res = None
        if api_status:
            with st.spinner("Calling API model..."):
                res = predict_churn_api(payload)
        
        if res is None:
            with st.spinner("Processing locally..."):
                res = predict_churn_local(payload)
                
        if res:
            prob = res["churn_probability"]
            label = res["churn_label"]
            
            st.markdown("<br>---", unsafe_allow_html=True)
            st.subheader("Prediction Result")
            
            # Display score beautifully
            c_score, c_msg = st.columns([1, 2])
            
            with c_score:
                st.metric("Churn Probability", f"{prob * 100:.1f}%")
                # Custom CSS progress bar
                color = "#10B981" if prob < 0.3 else ("#F59E0B" if prob < 0.7 else "#EF4444")
                st.markdown(f"""
                <div style="background-color: #1E293B; border-radius: 10px; width: 100%; height: 20px;">
                    <div style="background-color: {color}; width: {prob * 100:.1f}%; height: 100%; border-radius: 10px;"></div>
                </div>
                """, unsafe_allow_html=True)
                
            with c_msg:
                if prob >= 0.5:
                    st.markdown(f"""
                    <div class="churn-alert-high">
                        ⚠️ High Risk Customer (Probability: {prob*100:.1f}%) <br>
                        This customer is highly likely to churn. Recommend proactive outreach, exclusive offers, or contract renewal incentives.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="churn-alert-low">
                        ✅ Low Risk Customer (Probability: {prob*100:.1f}%) <br>
                        This customer is stable. Continue standard retention policies and monitor usage metrics.
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Could not run prediction. Please ensure model training has been executed successfully.")

# ==============================================================================
# Page 3: Model Diagnostic Hub
# ==============================================================================
elif page == "Model Diagnostic Hub":
    st.markdown("<h1 style='color: #6366F1;'>Model Diagnostic Hub</h1>", unsafe_allow_html=True)
    st.write("Compare trained models and inspect diagnostic metrics.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Load comparison metrics
    metrics_json_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r") as f:
            meta = json.load(f)
            
        st.subheader("Model Comparison Matrix")
        metrics_df = pd.DataFrame(meta["metrics"]).T
        st.table(metrics_df.style.highlight_max(axis=0, color="#1E3A8A"))
        st.write(f"🏆 Best selected model: **{meta['best_model_name']}** (based on ROC-AUC)")
        
        st.markdown("<br>---", unsafe_allow_html=True)
        
        # Display saved plot images
        st.subheader("Performance Curves")
        c_roc, c_cm = st.columns(2)
        
        with c_roc:
            roc_path = os.path.join(MODELS_DIR, "roc_curve_comparison.png")
            if os.path.exists(roc_path):
                st.image(roc_path, caption="Receiver Operating Characteristic (ROC) curves showing performance of all candidates.")
            else:
                st.info("ROC Curve Comparison plot not found. Run evaluations script first.")
                
        with c_cm:
            cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, caption=f"Confusion Matrix on hold-out test set for {meta['best_model_name']}.")
            else:
                st.info("Confusion Matrix plot not found. Run evaluations script first.")
                
        st.markdown("<br>---", unsafe_allow_html=True)
        
        # Feature importance
        st.subheader("Feature Importances")
        fi_path = os.path.join(MODELS_DIR, "feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Top features driving churn predictions.")
        else:
            st.info("Feature importance plot not found. Run evaluations script first.")
            
    else:
        st.info("No trained models or metrics found. Please execute the model training script first.")
