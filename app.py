import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(page_title="Telco Churn Predictor", page_icon="📊", layout="centered")

# Load Saved Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    columns = joblib.load('feature_columns.joblib')
    return model, scaler, columns

model, scaler, feature_columns = load_artifacts()

# App Header
st.title("📊 Telco Customer Churn Predictor")
st.write("Enter customer attributes below to predict the probability of churn.")

st.divider()

# Input Fields
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Predict Button
if st.button("🔮 Predict Churn Risk", use_container_width=True):
    # Construct Raw DataFrame matching encoding structure
    input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Assign Numeric Values
    if 'tenure' in input_data.columns: input_data['tenure'] = tenure
    if 'MonthlyCharges' in input_data.columns: input_data['MonthlyCharges'] = monthly_charges
    if 'TotalCharges' in input_data.columns: input_data['TotalCharges'] = total_charges
    
    # Scale Numeric Features
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
    
    # One-Hot Encoding Mappings
    contract_col = f"Contract_{contract}"
    if contract_col in input_data.columns: input_data[contract_col] = 1
        
    internet_col = f"InternetService_{internet_service}"
    if internet_col in input_data.columns: input_data[internet_col] = 1
        
    payment_col = f"PaymentMethod_{payment_method}"
    if payment_col in input_data.columns: input_data[payment_col] = 1

    # Generate Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    
    # Display Result
    if prediction == 1:
        st.error(f"⚠️ **High Churn Risk!** Probability: **{probability*100:.1f}%**")
        st.write("👉 **Action:** Consider offering contract incentives or discounts.")
    else:
        st.success(f"✅ **Low Churn Risk.** Probability: **{probability*100:.1f}%**")
        st.write("👉 **Action:** Customer is likely retained.")
