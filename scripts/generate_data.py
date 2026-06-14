import csv
import sqlite3
import random
import os
import math

def main():
    print("Generating synthetic customer churn dataset...")
    
    # Paths
    raw_data_dir = os.path.join("data", "raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    
    csv_path = os.path.join(raw_data_dir, "demographics.csv")
    db_path = os.path.join(raw_data_dir, "services.db")
    
    num_customers = 1500
    
    # 1. Generate demographics CSV data
    demographics_data = []
    
    # Seed for reproducibility
    random.seed(42)
    
    genders = ["Male", "Female"]
    yes_no = ["Yes", "No"]
    
    customer_ids = [f"CUST-{1000 + i}" for i in range(num_customers)]
    
    for cust_id in customer_ids:
        gender = random.choice(genders)
        senior = 1 if random.random() < 0.16 else 0
        partner = random.choice(yes_no)
        dependents = "Yes" if random.random() < 0.3 else "No"
        
        demographics_data.append({
            "CustomerId": cust_id,
            "Gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents
        })
        
    # Write CSV
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["CustomerId", "Gender", "SeniorCitizen", "Partner", "Dependents"])
        writer.writeheader()
        for row in demographics_data:
            writer.writerow(row)
            
    print(f"Saved demographics CSV to {csv_path}")
    
    # 2. Generate services and churn SQL database
    # Connect to SQLite
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS services (
        CustomerId TEXT PRIMARY KEY,
        Tenure INTEGER,
        PhoneService TEXT,
        MultipleLines TEXT,
        InternetService TEXT,
        OnlineSecurity TEXT,
        OnlineBackup TEXT,
        DeviceProtection TEXT,
        TechSupport TEXT,
        StreamingTV TEXT,
        StreamingMovies TEXT,
        Contract TEXT,
        PaperlessBilling TEXT,
        PaymentMethod TEXT,
        MonthlyCharges REAL,
        TotalCharges TEXT,
        Churn TEXT
    )
    """)
    
    contracts = ["Month-to-month", "One year", "Two year"]
    payment_methods = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    internet_services = ["DSL", "Fiber optic", "No"]
    
    services_data = []
    
    for i, cust_id in enumerate(customer_ids):
        # Determine contract type and tenure
        contract = random.choices(contracts, weights=[0.55, 0.22, 0.23], k=1)[0]
        
        if contract == "Month-to-month":
            tenure = int(random.lognormvariate(1.5, 0.8)) + 1
            tenure = min(max(tenure, 1), 72)
        elif contract == "One year":
            tenure = int(random.uniform(12, 36))
        else: # Two year
            tenure = int(random.uniform(24, 72))
            
        phone_service = "Yes" if random.random() < 0.90 else "No"
        
        if phone_service == "Yes":
            multiple_lines = "Yes" if random.random() < 0.42 else "No"
        else:
            multiple_lines = "No phone service"
            
        internet = random.choices(internet_services, weights=[0.35, 0.45, 0.20], k=1)[0]
        
        if internet != "No":
            online_sec = "Yes" if random.random() < 0.35 else "No"
            online_bak = "Yes" if random.random() < 0.40 else "No"
            dev_prot = "Yes" if random.random() < 0.38 else "No"
            tech_sup = "Yes" if random.random() < 0.36 else "No"
            st_tv = "Yes" if random.random() < 0.45 else "No"
            st_mov = "Yes" if random.random() < 0.45 else "No"
        else:
            online_sec = "No internet service"
            online_bak = "No internet service"
            dev_prot = "No internet service"
            tech_sup = "No internet service"
            st_tv = "No internet service"
            st_mov = "No internet service"
            
        paperless = "Yes" if random.random() < 0.60 else "No"
        pay_method = random.choice(payment_methods)
        
        # Calculate MonthlyCharges based on services
        charges = 20.0 # Base rate
        if phone_service == "Yes":
            charges += 10.0
        if multiple_lines == "Yes":
            charges += 15.0
            
        if internet == "DSL":
            charges += 30.0
            if online_sec == "Yes": charges += 8.0
            if online_bak == "Yes": charges += 7.0
            if dev_prot == "Yes": charges += 7.0
            if tech_sup == "Yes": charges += 9.0
            if st_tv == "Yes": charges += 12.0
            if st_mov == "Yes": charges += 12.0
        elif internet == "Fiber optic":
            charges += 55.0
            if online_sec == "Yes": charges += 10.0
            if online_bak == "Yes": charges += 8.0
            if dev_prot == "Yes": charges += 9.0
            if tech_sup == "Yes": charges += 11.0
            if st_tv == "Yes": charges += 15.0
            if st_mov == "Yes": charges += 15.0
            
        # Add slight variation
        charges = round(charges + random.uniform(-3, 3), 2)
        
        # Calculate TotalCharges
        tot_charges = charges * tenure
        
        # Simulate missing values for clean data demonstration (around 0.5% of rows)
        if random.random() < 0.005:
            total_charges_str = " "
        else:
            total_charges_str = f"{round(tot_charges, 2)}"
            
        # Determine Churn based on risk factors (probabilistic)
        # Log-odds of churning
        log_odds = -0.5
        
        # Contract risk
        if contract == "Month-to-month":
            log_odds += 1.8
        elif contract == "Two year":
            log_odds -= 2.2
            
        # Tenure benefit
        log_odds -= 0.06 * tenure
        
        # Internet service risk
        if internet == "Fiber optic":
            log_odds += 0.8
        elif internet == "No":
            log_odds -= 0.8
            
        # Technical support benefit
        if tech_sup == "No" and internet != "No":
            log_odds += 0.6
        elif tech_sup == "Yes":
            log_odds -= 0.5
            
        # Senior citizen risk
        if demographics_data[i]["SeniorCitizen"] == 1:
            log_odds += 0.4
            
        # Payment method risk
        if pay_method == "Electronic check":
            log_odds += 0.7
            
        # Convert log-odds to probability
        prob = 1.0 / (1.0 + math.exp(-log_odds))
        
        churn_val = "Yes" if random.random() < prob else "No"
        
        services_data.append((
            cust_id, tenure, phone_service, multiple_lines, internet,
            online_sec, online_bak, dev_prot, tech_sup, st_tv, st_mov,
            contract, paperless, pay_method, charges, total_charges_str, churn_val
        ))
        
    # Insert data
    cursor.executemany("""
    INSERT INTO services VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, services_data)
    
    conn.commit()
    conn.close()
    
    print(f"Saved services and churn to SQLite DB at {db_path}")
    print("Synthetic data generation completed successfully!")

if __name__ == "__main__":
    main()
