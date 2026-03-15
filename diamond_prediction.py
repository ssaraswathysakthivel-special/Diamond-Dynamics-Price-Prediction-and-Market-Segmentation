# streamlit_diamond_app.py

import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib  # For clustering model

# -------------------------------
# Load models
# -------------------------------
# Regression model (ANN)
reg_model = load_model(r"C:\Users\ADMIN\Documents\mini_project_guvi\project_diamond\diamond_price_model.keras")

# Clustering model (e.g., KMeans)
cluster_model = joblib.load(r"C:\Users\ADMIN\Documents\mini_project_guvi\project_diamond\best_kmeans_model.pkl")

# Cluster mapping (optional, human-readable names)
cluster_names = {
    0: "Premium Heavy Diamonds",
    1: "Affordable Small Diamonds",
    2: "Mid-range Standard Diamonds"
}

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Diamond Price & Market Predictor", layout="wide")
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #ff69b4, #8a2be2);  /* changed gradient */
        padding: 0.8rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 8px;
    ">
        <h1 style="color:#ffffff; font-size:2.5rem; margin:0;">  <!-- changed text color -->
            💎 Diamond Price & Market Predictor
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption(":blue[Predict diamond price and market segment based on its attributes]")
# -------------------------------
# Input form in 3 columns
# -------------------------------
st.subheader("Enter Diamond Attributes:")

with st.form(key="diamond_form"):
    
    # Create 3 columns
    col1, col2, col3 = st.columns(3)

    # Column 1: Numeric inputs
    with col1:
        carat = st.number_input("Carat", min_value=0.01, max_value=5.0, value=0.5, step=0.01)
        x_dim = st.number_input("X (length in mm)", min_value=0.0, max_value=10.0, value=5.0, step=0.01)
    
    # Column 2: Numeric inputs continued
    with col2:
        y_dim = st.number_input("Y (width in mm)", min_value=0.0, max_value=10.0, value=5.0, step=0.01)
        z_dim = st.number_input("Z (depth in mm)", min_value=0.0, max_value=10.0, value=3.0, step=0.01)
    
    # Column 3: Categorical inputs
    with col3:
        cut = st.selectbox("Cut", ["Ideal", "Premium", "Good", "Very Good", "Fair"])
        color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
        clarity = st.selectbox("Clarity", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])

    # Submit buttons (spanning all columns)
    predict_price_btn = st.form_submit_button("Predict Price")
    predict_cluster_btn = st.form_submit_button("Predict Cluster")# -------------------------------
# Preprocessing helpers
# -------------------------------
def preprocess_inputs(carat, x_dim, y_dim, z_dim, cut, color, clarity):
    """
    Convert user inputs to model-ready format.
    Assumes same encoding/scaling used during model training.
    """
    # Example: encode categorical variables numerically
    cut_map = {"Fair":0, "Good":1, "Very Good":2, "Premium":3, "Ideal":4}
    color_map = {"D":0, "E":1, "F":2, "G":3, "H":4, "I":5, "J":6}
    clarity_map = {"I1":0, "SI2":1, "SI1":2, "VS2":3, "VS1":4, "VVS2":5, "VVS1":6, "IF":7}

    # Numeric array for model
    features = np.array([
        carat,
        x_dim,
        y_dim,
        z_dim,
        cut_map[cut],
        color_map[color],
        clarity_map[clarity]
    ]).reshape(1, -1)

    # Optional: scale features if used during ANN training
    # from sklearn.preprocessing import StandardScaler
    # features = scaler.transform(features)

    return features

# -------------------------------
# Price Prediction
# -------------------------------
if predict_price_btn:
    inputs = preprocess_inputs(carat, x_dim, y_dim, z_dim, cut, color, clarity)
    price_pred = reg_model.predict(inputs)[0][0]
    st.success(f"💰 Predicted Diamond Price: ₹{price_pred:,.2f}")

# -------------------------------
# Cluster Prediction
# -------------------------------
if predict_cluster_btn:
    inputs = preprocess_inputs(carat, x_dim, y_dim, z_dim, cut, color, clarity)
    cluster_pred = cluster_model.predict(inputs)[0]
    cluster_name = cluster_names.get(cluster_pred, f"Cluster {cluster_pred}")
    st.info(f"🔹 Predicted Market Segment: {cluster_name} (Cluster {cluster_pred})")