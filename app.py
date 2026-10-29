
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="DC Motor Monitor",
    page_icon="⚙️",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("motor_model.pkl")

model = load_model()

st.title("⚙️ DC Motor Monitoring System")
st.write("ML-Based Motor Condition Prediction")
st.info("Software simulation only — no physical motor is connected.")

st.subheader("Enter Simulated Sensor Values")

current = st.slider(
    "Current (A)", 0.1, 2.0, 0.6, 0.1
)

temperature = st.slider(
    "Temperature (°C)", 20, 85, 35, 1
)

vibration = st.slider(
    "Vibration", 0.1, 2.5, 0.3, 0.1
)

if st.button("Check Motor Condition", type="primary"):
    readings = pd.DataFrame({
        "Current": [current],
        "Temperature": [temperature],
        "Vibration": [vibration]
    })

    prediction = str(model.predict(readings)[0]).strip().lower()

    st.divider()

    if prediction == "abnormal":
        st.error("⚠️ MOTOR STATUS: ABNORMAL")
        st.warning(
            "Abnormal condition predicted. "
            "Please inspect the motor in a real deployment."
        )
    else:
        st.success("✅ MOTOR STATUS: NORMAL")
        st.write("The ML model predicts normal condition.")

    st.caption(
        "Prediction is based on the trained model "
        "and simulated input values."
    )
