# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image
st.set_page_config(layout="centered", page_title="Water Potability Predictor")
model = joblib.load("best_pipeline.pkl")
st.title("💧 Water Potability Predictor")
with st.expander("Dataset info"):
    st.write("Features: ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity")
st.sidebar.header("Input parameters")
ph = st.sidebar.slider("pH", 0.0, 14.0, 7.0, 0.1)
Hardness = st.sidebar.number_input("Hardness", min_value=0.0, value=200.0, step=0.1)
Solids = st.sidebar.number_input("Solids (TDS)", min_value=0.0, value=20000.0, step=1.0)
Chloramines = st.sidebar.number_input("Chloramines", min_value=0.0, value=7.0, step=0.01)
Sulfate = st.sidebar.number_input("Sulfate", min_value=0.0, value=333.0, step=0.1)
Conductivity = st.sidebar.number_input("Conductivity", min_value=0.0, value=420.0, step=0.1)
Organic_carbon = st.sidebar.number_input("Organic Carbon", min_value=0.0, value=14.0, step=0.1)
Trihalomethanes = st.sidebar.number_input("Trihalomethanes", min_value=0.0, value=66.0, step=0.1)
Turbidity = st.sidebar.number_input("Turbidity", min_value=0.0, value=4.0, step=0.01)
input_df = pd.DataFrame([[ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity]],
                        columns=['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity'])
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else None
    if pred == 1:
        st.success("✅ Predicted: Potable (Safe to drink)")
    else:
        st.error("❌ Predicted: Not Potable (Not safe to drink)")
    if prob is not None:
        st.write("Potability probability:", f"{prob:.3f}")
st.markdown("---")
st.header("Model insights")
try:
    img = Image.open("shap_summary.png")
    st.image(img, use_column_width=True)
except:
    st.write("SHAP summary not available")
st.markdown("---")
st.header("Quick WHO-like thresholds (informational)")
st.write("""
- pH: 6.5 - 8.5 recommended range (approx)
- Turbidity: lower is better
- High Trihalomethanes often indicate byproduct of chlorination
""")
