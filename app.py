import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = pickle.load(open("food_delivery_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# Title of the app
st.title("Food Delivery Time Prediction")

# Input fields
age = st.number_input("Delivery Person Age", min_value=18, max_value=100, value=25)
ratings = st.number_input("Delivery Person Ratings", min_value=1.0, max_value=5.0, value=4.5)
distance = st.number_input("Total Distance (km)", min_value=1.0, max_value=100.0, value=5.0)

# User inputs as features
features = np.array([[age, ratings, distance]])

# Scaling the features
features_scaled = scaler.transform(features)

# Prediction
if st.button("Predict Delivery Time"):
    prediction = model.predict(features_scaled)
    st.write(f"Predicted Delivery Time: {prediction[0]} minutes")
