import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io

# ------------------------------
# Load dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")
    # Standardize column names (match training pipeline)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

# ------------------------------
# Load model from Hugging Face
# ------------------------------
@st.cache_resource
def load_model_from_hf(repo_id, filename):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    response = requests.get(url)
    if response.status_code == 200:
        return joblib.load(io.BytesIO(response.content))
    else:
        st.error(f"Error loading model: {response.status_code}")
        return None

# ------------------------------
# App
# ------------------------------
st.title("🏡 Airbnb Price Prediction Dashboard")

df = load_data()

# Load models
rf_model = load_model_from_hf("UjjwalKaushik/Airbnb_model", "best_random_forest.pkl")
xgb_model = load_model_from_hf("UjjwalKaushik/Airbnb_model", "best_xgboost.pkl")

# Sidebar filters
st.sidebar.header("Filter Options")

room_type = st.sidebar.selectbox("Room Type", df["room_type"].unique())
neighbourhood_group = st.sidebar.selectbox("Neighbourhood Group", df["neighbourhood_group"].unique())
minimum_nights = st.sidebar.slider("Minimum Nights", 1, 30, 3)
number_of_reviews = st.sidebar.slider("Number of Reviews", 0, 500, 10)

# Prepare input data
input_data = pd.DataFrame({
    "room_type": [room_type],
    "neighbourhood_group": [neighbourhood_group],
    "minimum_nights": [minimum_nights],
    "number_of_reviews": [number_of_reviews]
})

# ✅ Ensure input_data has all features model expects
if rf_model is not None:
    expected_cols = rf_model.feature_names_in_
    for col in expected_cols:
        if col not in input_data.columns:
            input_data[col] = 0  # fill missing features with default
    input_data = input_data[expected_cols]  # reorder to match training

    rf_pred = rf_model.predict(input_data)[0]
    st.subheader("Random Forest Prediction")
    st.write(f"💰 Predicted Price: **${rf_pred:.2f}**")

if xgb_model is not None:
    expected_cols = xgb_model.feature_names_in_
    for col in expected_cols:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[expected_cols]

    xgb_pred = xgb_model.predict(input_data)[0]
    st.subheader("XGBoost Prediction")
    st.write(f"💰 Predicted Price: **${xgb_pred:.2f}**")

# ------------------------------
# Dataset Preview
# ------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())
