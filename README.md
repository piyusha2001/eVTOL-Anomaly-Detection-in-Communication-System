# eVTOL Communication Link Anomaly Detector

**A machine learning-powered web application to detect anomalies in real-time for safety-critical eVTOL communication links.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://evtol-anomaly-detection-in-communication-system-mby5qjbwaee3en.streamlit.app/)

---

## 🚀 Project Overview

This project addresses the critical need for reliable communication in electric Vertical Takeoff and Landing (eVTOL) aircraft. The entire command and control system depends on a stable communication link, where even short disruptions can pose significant safety risks.

This application uses a trained Random Forest model to analyze simulated real-time telemetry data and predict whether the communication link is healthy or experiencing an anomaly. The goal is to provide an early warning system that can detect link degradation *before* a catastrophic failure occurs.

## ✨ Key Features

-   **Real-Time Anomaly Prediction:** Enter live telemetry values and get an instant prediction on the link's health.
-   **Interactive Dashboard:** A user-friendly interface built with Streamlit for simulating various link conditions.
-   **High-Recall Model:** The underlying Random Forest model was specifically tuned to maximize **Recall (81.0%)**, prioritizing the detection of every potential anomaly.
-   **Model Transparency:** The dashboard displays the champion model's key performance metrics (Precision, Recall, Accuracy).
-   **Feature Explanations:** Clear captions for each input explain its significance in communication systems.

## 🛠️ How It Was Built (Tech Stack)

-   **Data Generation & Analysis:** Python, NumPy, Pandas
-   **Machine Learning:** Scikit-learn, LightGBM (for model comparison), Joblib (for model serialization)
-   **Frontend Dashboard:** Streamlit
-   **Deployment:** Streamlit Community Cloud & GitHub

## 🤖 The Machine Learning Workflow

The model was developed through a systematic, end-to-end machine learning process:

1.  **Synthetic Data Generation:** A high-quality, balanced dataset of 40,000 data points was programmatically generated to simulate five distinct communication scenarios:
    -   Normal Operation (Healthy Link)
    -   Gradual Degradation (e.g., increasing distance)
    -   Sudden Interference (e.g., jamming)
    -   Intermittent Faults (e.g., multi-path fading)
    -   Synchronization Issues (e.g., receiver fault)

2.  **Model Bake-Off:** Three different models (Logistic Regression, Random Forest, LightGBM) were trained and evaluated to find the most promising candidate. The Random Forest was selected for its robust performance.

3.  **Hyperparameter Tuning:** Scikit-learn's `GridSearchCV` was used to fine-tune the Random Forest model. Critically, the model was optimized specifically for **Recall**, ensuring it was as vigilant as possible in catching failures.

4.  **Feature Importance:** Analysis of the final model revealed that **EVM (Error Vector Magnitude)** and **SNR (Signal-to-Noise Ratio)** were the most important features, confirming that the model's logic aligns with communications engineering principles.

## 💻 How to Run Locally

To run this application on your own machine:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/evtol-anomaly-dashboard.git
    cd evtol-anomaly-dashboard
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create --name evtol_env python=3.9
    conda activate evtol_env
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

---