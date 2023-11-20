import streamlit as st
import pandas as pd
import joblib

model = joblib.load('gradient_boosting_model.joblib')


# Streamlit App
st.title("Palm Pressure Prediction App")

# User Input
posture_options = ['1', '2', '3', '4', '5', '6', '7']
posture = st.selectbox("Select Posture:", posture_options)

bmi = st.number_input("Enter BMI:")
gender = st.radio("Select Gender:", ['Female', 'Male'])
dominant_hand = st.radio("Select Dominant Hand:", ['Left', 'Right'])

# Convert user input to numerical format
gender = 0 if gender == 'Female' else 1
dominant_hand = 0 if dominant_hand == 'Left' else 1

# Predict Pressure
user_data = pd.DataFrame({
    'BMI': [bmi],
    'Gender': [gender],
    'Dominant_Hand': [dominant_hand],
    'Posture': [int(posture)]
})

predicted_pressure = model.predict(user_data)[0]

# Display Prediction
st.subheader("Predicted Pressure:")
st.write(predicted_pressure)
