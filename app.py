import streamlit as st
import pickle
import numpy as np

st.title("🍷 Wine Quality Prediction App")

# Load the trained model from pickle file
try:
    with open("wine_quality_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found! Please ensure wine_quality_model.pkl is in the same directory as this app.")
    st.stop()

st.write("Enter wine chemical properties:")

# Input fields - exactly matching your dataset columns (11 features)
fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.7)
citric_acid = st.number_input("Citric Acid", value=0.0)
residual_sugar = st.number_input("Residual Sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
density = st.number_input("Density", value=0.9978)
pH = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.4)

# Predict button
if st.button("Predict Wine Quality"):
    sample_data = np.array([
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]).reshape(1, -1)

    prediction = model.predict(sample_data)
    st.success(f"🍷 Predicted Wine Quality Class: {prediction[0]}")