import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

# Load dataset (replace this with your own dataset if needed)
data = load_wine()
X = data.data
y = data.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

st.title("🍷 Wine Quality Prediction App")

st.write("Enter wine chemical properties:")

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", value=7.9)
volatile_acidity = st.number_input("Volatile Acidity", value=0.35)
citric_acid = st.number_input("Citric Acid", value=0.46)
residual_sugar = st.number_input("Residual Sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34)
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
