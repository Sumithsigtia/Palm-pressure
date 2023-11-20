import streamlit as st
import pandas as pd
import pickle

# Load the model
model = pickle.load(open('gradient_boosting_model.sav', 'rb'))

# Streamlit App
st.title("Palm Pressure Prediction App")

# User Input Sidebar
st.sidebar.header("User Input")

# Posture Selection with Descriptions
posture_options = {
    '1': 'In Standing - Full body weight distribution (without bending the elbow and elevated shoulder).',
    '2': 'Only arm weight distribution (without body weight).',
    '3': 'Forward loading - Body weight shifting towards the loading arm forward - pushing the subject front.',
    '4': 'Backwards unloading - Body weight shifting from the loading arm backward - pulling the subject back.',
    '5': 'Sideways loading - Body weight shifting toward the loading arm - pushing the subject towards the loading arm sideways.',
    '6': 'Sideways unloading - Body weight shifting away from the loading arm - pulling the subject away from the loading arm sideways.',
    '7': 'In sitting - Elbow straight without bending, without elevating the shoulder. Apply constant pressure on the device.',
}

posture = st.sidebar.selectbox("Select Posture:", list(posture_options.keys()), format_func=lambda x: posture_options[x])

# BMI Input with Information
st.sidebar.subheader("BMI (Body Mass Index)")
st.sidebar.info("BMI is a measure of body fat based on height and weight.")
bmi = st.sidebar.number_input("Enter BMI:")

# Gender Selection with Emoji
st.sidebar.subheader("Gender")
gender = st.sidebar.radio("Select Gender:", ['👩 Female', '👨 Male'])

# Dominant Hand Selection with Emoji
st.sidebar.subheader("Dominant Hand")
dominant_hand = st.sidebar.radio("Select Dominant Hand:", ['🤚 Left', '✋ Right'])

# Button to Display Prediction
if st.sidebar.button("Predict Pressure"):
    # Convert user input to numerical format
    gender_code = 0 if "Female" in gender else 1
    dominant_hand_code = 0 if "Left" in dominant_hand else 1

    # Predict Pressure
    user_data = pd.DataFrame({
        'BMI': [bmi],
        'Gender': [gender_code],
        'Dominant_Hand': [dominant_hand_code],
        'Posture': [int(posture)]
    })

    predicted_pressure = model.predict(user_data)[0]

    # Display Prediction with larger font size and box
    st.subheader("Predicted Pressure:")
    st.markdown(f"<div style='font-size:24px; padding: 10px; border: 2px solid #f63366; border-radius: 10px;'>"
                f"<strong>{predicted_pressure:.2f}</strong></div>", unsafe_allow_html=True)

# Posture Visualization
st.markdown("### Posture Visualization")
posture_images = {
    '1': 'Posture1.jpg',
    '2': 'Posture2.jpg',
    '3': 'Posture3.jpg',
    '4': 'Posture4.jpg',
    '5': 'Posture5.jpg',
    '6': 'Posture6.jpg',
    '7': 'Posture7.jpg',
}

selected_posture_image = posture_images.get(posture, 'default_image_url')
st.image(selected_posture_image, caption=f"Posture {posture}", use_column_width=True)

# Information about ML Prediction and Experimental Diagnosis
st.markdown("### Information")
st.write("This is a predicted pressure using Machine Learning. "
         "Experimentally, palm pressure can be measured using FBG sensor or load cell. "
         "If the patient disagrees with the predicted values, further diagnosis can be done, and "
         "it may help in identifying diseases impacting palm pressure.")

# Instructions on Computing BMI
st.markdown("### BMI Calculation")
st.write("BMI is calculated as weight (kg) divided by the square of height (m). "
         "The formula is BMI = weight / (height * height). "
         "BMI provides an indication of body fat and helps assess health risks associated with weight.")

# BMI Chart Image
st.markdown("### BMI Chart")
bmi_chart_image = 'bmi.png'  # Replace with the actual file path or URL for the BMI chart image
st.image(bmi_chart_image, caption="BMI Chart", use_column_width=True)

# Information about Diseases and Cure
st.markdown("### Diseases and Cure")
disease_image = 'disease.png'  # Replace with the actual file path or URL for the BMI chart image
st.image(disease_image, caption="Palm Impairments", use_column_width=True)
st.markdown("""
1. **Carpal Tunnel Syndrome (CTS):**
   Symptoms: Numbness, tingling, and weakness in the hand due to compression of the median nerve in the wrist.
   Treatment: Rest, wrist splints, medications, and in severe cases, surgery.

2. **Rheumatoid Arthritis (RA):**
   Symptoms: Joint pain, swelling, and stiffness, often affecting the hands and wrists.
   Treatment: Medications, physical therapy, and in some cases, surgery.

3. **Wrist Arthroplasty:**
   Description: Surgical procedure involving the replacement of the wrist joint with an artificial joint.
   Purpose: To relieve pain and improve function, often used in cases of severe arthritis.

4. **Swan Neck Deformity:**
   Description: Deformity of the finger, characterized by hyperextension of the proximal interphalangeal (PIP) joint and flexion of the distal interphalangeal (DIP) joint.
   Treatment: Varied and may include splints, exercises, or surgery depending on severity.

5. **Leprosy:**
   Symptoms: Skin lesions, nerve damage, and muscle weakness.
   Treatment: Multi-drug therapy (MDT) is effective in curing leprosy.

6. **Frozen Shoulder (Adhesive Capsulitis):**
   Symptoms: Gradual onset of shoulder pain and stiffness, limiting range of motion.
   Treatment: Physical therapy, medications, and in some cases, joint distension or surgery.
""")
