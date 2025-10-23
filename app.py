import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="eVTOL Anomaly Detector",
    page_icon="✈️",
    layout="wide"  # Use wide layout for better spacing
)

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the pre-trained Random Forest model."""
    try:
        model = joblib.load('evtol_anomaly_detector.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'evtol_anomaly_detector.pkl' is in the same directory.")
        return None

model = load_model()

# --- Application UI ---
st.title("✈️ eVTOL Communication Link Anomaly Detector")
st.write(
    "This dashboard uses a pre-trained Random Forest model to predict the health of an eVTOL's communication link in real-time. "
    "Enter the precise telemetry values in the sidebar to simulate a sensor reading and see the model's prediction."
)

# --- NEW: Model Performance Section ---
with st.expander("View Champion Model Performance Metrics"):
    st.write("These are the performance metrics of the final Random Forest model, tuned to maximize the detection of anomalies (Recall).")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Overall Accuracy",
            value="87.72%",
            help="The percentage of all predictions (both normal and anomalous) that the model got correct."
        )
    with col2:
        st.metric(
            label="Anomaly Precision",
            value="94.00%",
            help="When the model flags an anomaly, this is the percentage of time it is correct. High precision means fewer false alarms."
        )
    with col3:
        st.metric(
            label="Anomaly Recall",
            value="81.03%",
            help="**Our most important metric.** This is the percentage of all real anomalies that the model successfully detected. High recall means fewer missed failures."
        )


st.sidebar.header("Live Telemetry Inputs")

# --- UPDATED FUNCTION with Explanations ---
def user_input_features():
    """Creates number input fields with explanations in the sidebar."""
    
    snr = st.sidebar.number_input(
        'Signal-to-Noise Ratio (SNR)', 
        min_value=-5.0, max_value=30.0, value=25.0, 
        step=0.01, format="%.4f"
    )
    st.sidebar.caption("Higher is better. Represents the clarity of the signal over background noise.")

    rssi = st.sidebar.number_input(
        'Received Signal Strength (RSSI)', 
        min_value=-50.0, max_value=-30.0, value=-35.0, 
        step=0.01, format="%.4f"
    )
    st.sidebar.caption("Higher (less negative) is better. Represents the total power of the received signal.")

    ber = st.sidebar.number_input(
        'Bit Error Rate (BER)', 
        min_value=0.0, max_value=0.5, value=0.0, 
        step=1e-6, format="%.6f"
    )
    st.sidebar.caption("Lower is better (0 is perfect). The percentage of data bits that were corrupted during transmission.")

    evm = st.sidebar.number_input(
        'Error Vector Magnitude (EVM %)', 
        min_value=0.0, max_value=40.0, value=1.0, 
        step=0.01, format="%.4f"
    )
    st.sidebar.caption("Lower is better. Measures the 'messiness' of the signal. A great early warning sign of link degradation.")

    phase_offset = st.sidebar.number_input(
        'Phase Offset', 
        min_value=-2.0, max_value=3.0, value=0.0, 
        step=1e-6, format="%.6f"
    )
    st.sidebar.caption("Closer to 0 is better. A large value indicates a receiver synchronization problem.")
    
    frequency_offset = st.sidebar.number_input(
        'Frequency Offset', 
        min_value=-1.0, max_value=1.5, value=0.0, 
        step=1e-9, format="%.9f"
    )
    st.sidebar.caption("Closer to 0 is better. A large value indicates a receiver synchronization problem.")
    
    data = {
        'SNR': snr, 'RSSI': rssi, 'BER': ber, 'EVM': evm,
        'Phase_Offset': phase_offset, 'Frequency_Offset': frequency_offset
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user's selected inputs in a table
st.subheader("Current Input Parameters")
st.dataframe(input_df) # Use st.dataframe for a nicer table display

# --- Prediction Logic ---
if model is not None:
    if st.button("Analyze Link Health", type="primary"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("Anomaly Detected! 🚨", icon="⚠️")
            st.write(f"**Confidence:** {prediction_proba[0][1]*100:.2f}%")
            st.write("The model has detected a high probability of link degradation or failure. This could be due to low signal strength, high noise, or receiver synchronization issues. Immediate attention is recommended.")
        else:
            st.success("Link is Normal ✅", icon="✔️")
            st.write(f"**Confidence:** {prediction_proba[0][0]*100:.2f}%")
            st.write("The communication link appears to be stable and healthy based on the current telemetry data.")
else:
    st.warning("Model is not loaded. Please check the file path.")