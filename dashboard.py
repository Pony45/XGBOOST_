import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="XGBoost M&V", layout="wide")
st.title("🚀 XGBoost M&V Dashboard")
st.markdown("*AI-based Measurement & Verification using XGBoost Regressor*")

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    model = joblib.load('models/xgboost_model.pkl')
    with open('models/xgboost_features.txt', 'r') as f:
        features = [line.strip() for line in f.readlines()]
    return model, features

model, FEATURES = load_model()
st.success("✅ XGBoost model loaded!")

# ==========================================
# LOAD METRICS
# ==========================================
@st.cache_resource
def load_metrics():
    if os.path.exists('models/xgboost_metrics.json'):
        with open('models/xgboost_metrics.json', 'r') as f:
            return json.load(f)
    return None

metrics = load_metrics()

# Show metrics in sidebar
if metrics:
    with st.sidebar:
        st.header("📊 Model Performance")
        st.metric("R² Score", f"{metrics['r2_score']:.4f}")
        st.metric("MAE", f"{metrics['mae']:.2f} kWh")
        st.metric("RMSE", f"{metrics['rmse']:.2f} kWh")
        st.caption(f"MAPE: {metrics.get('mape', 0):.1f}%")

# ==========================================
# SCALING
# ==========================================
SCALING = 56

# ==========================================
# SIDEBAR INPUTS
# ==========================================
with st.sidebar:
    st.header("📋 Building Parameters")
    st.markdown("---")
    
    temp = st.slider("🌡️ Temperature (°C)", 22, 35, 28)
    humidity = st.slider("💧 Humidity (%)", 60, 95, 80)
    hour = st.slider("⏰ Hour", 0, 23, 14)
    day = st.selectbox("📅 Day", range(7), format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
    month = st.selectbox("📆 Month", range(1,13), format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
    area = st.number_input("🏠 Floor Area (m²)", 50, 300, 120)
    occ = st.number_input("👥 Occupants", 1, 8, 4)
    retro = st.selectbox("🔧 Retrofit", [0,1], format_func=lambda x: "✅ Yes (Retrofitted)" if x else "❌ No (Baseline)")

# ==========================================
# FEATURE ENGINEERING
# ==========================================
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
is_weekend = 1 if day >= 5 else 0
temp_hum = temp * humidity / 100
occ_area = occ / area

X = np.array([[
    temp, humidity, hour, day, month, area, occ, retro,
    hour_sin, hour_cos, month_sin, month_cos, is_weekend, temp_hum, occ_area
]])

# ==========================================
# PREDICTION
# ==========================================
if st.button("🔮 Predict Energy", type="primary", use_container_width=True):
    pred = model.predict(X)[0] / SCALING
    st.metric("⚡ Predicted Energy", f"{pred:.2f} kWh")
    
    if retro == 1:
        X_base = X.copy()
        X_base[0][7] = 0
        base = model.predict(X_base)[0] / SCALING
        savings = base - pred
        pct = (savings / base) * 100 if base > 0 else 0
        st.success(f"💰 Savings: {savings:.2f} kWh ({pct:.1f}%)")
        st.info(f"💰 Monthly Savings: RM {savings * 24 * 30 * 0.52:.2f}")
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(['Baseline\n(No Retrofit)', 'Retrofitted'], [base, pred], 
               color=['#e74c3c', '#2ecc71'], edgecolor='black')
        ax.set_ylabel('Energy (kWh)')
        ax.set_title('XGBoost: Retrofit Impact', fontweight='bold')
        st.pyplot(fig)
    else:
        X_retro = X.copy()
        X_retro[0][7] = 1
        retro_pred = model.predict(X_retro)[0] / SCALING
        potential = pred - retro_pred
        pct = (potential / pred) * 100 if pred > 0 else 0
        st.info(f"💡 If retrofitted: Save ~{potential:.2f} kWh ({pct:.1f}%)")
        
        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(['Current\n(No Retrofit)', 'If Retrofitted'], [pred, retro_pred],
               color=['#e74c3c', '#2ecc71'], edgecolor='black')
        ax.set_ylabel('Energy (kWh)')
        ax.set_title('Potential Retrofit Impact', fontweight='bold')
        st.pyplot(fig)

st.markdown("---")
st.caption("🚀 XGBoost M&V System | Thesis Project | Scaled for Malaysian Residential Buildings")
