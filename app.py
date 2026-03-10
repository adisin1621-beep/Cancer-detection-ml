import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

st.title("Breast Cancer Prediction App")

st.write("Enter tumor measurements to predict if it is benign or malignant.")

# User inputs
mean_radius = st.number_input("Mean Radius")
mean_texture = st.number_input("Mean Texture")
mean_area = st.number_input("Mean Area")

if st.button("Predict"):
    
    input_data = [[mean_radius, mean_texture, mean_area] + [0]*(X.shape[1]-3)]
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("Benign Tumor")
    else:
        st.error("Malignant Tumor")