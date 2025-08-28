import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io

# ---------------------------
# Load Models from HuggingFace
# ---------------------------
@st.cache_resource
def load_model_from_hf(repo_id, filename):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    response = requests.get(url)
    response.raise_for_status()
    return joblib.load(io.BytesIO(response.content))

rf_model = load_model_from_hf("UjjwalKaushik/Airbnb_model", "best_random_forest.pkl")
xgb_model = load_model_from_hf("UjjwalKaushik/Airbnb_model", "best_xgboost.pkl")

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")

    # ✅ Rename columns to match training pipeline
    df.rename(columns={
        "NAME": "name",
        "host id": "host_id",
        "neighbourhood group": "neighbourhood_group",
        "lat": "latitude",
        "long": "longitude",
        "country code": "country_code",
        "room type": "room_type",
        "Construction year": "construction_year",
        "service fee": "service_fee",
        "minimum nights": "minimum_nights",
        "number of reviews": "number_of_reviews",
        "reviews per month": "reviews_per_month",
        "review rate number": "review_rate_number",
        "calculated host listings count": "calculated_host_listings_count",
        "availability 365": "availability_365"
    }, inplace=True)

    return df

df = load_data()

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("🏡 Airbnb Price Prediction")
st.write("This app predicts **Airbnb listing price** using Random Forest and XGBoost models.")

# Sidebar inputs
st.sidebar.header("Input Features")

neighbourhood_group = st.sidebar.selectbox("Neighbourhood Group", df["neighbourhood_group"].unique())
room_type = st.sidebar.selectbox("Room Type", df["room_type"].unique())
minimum_nights = st.sidebar.number_input("Minimum Nights", min_value=1, max_value=500, value=3)
number_of_reviews = st.sidebar.number_input("Number of Reviews", min_value=0, max_value=1000, value=10)
reviews_per_month = st.sidebar.number_input("Reviews per Month", min_value=0.0, max_value=30.0, value=1.0)
availability_365 = st.sidebar.number_input("Availability (days)", min_value=0, max_value=365, value=180)

# ---------------------------
# Prepare Input for Prediction
# ---------------------------
input_data = pd.DataFrame({
    "neighbourhood_group": [neighbourhood_group],
    "room_type": [room_type],
    "minimum_nights": [minimum_nights],
    "number_of_reviews": [number_of_reviews],
    "reviews_per_month": [reviews_per_month],
    "availability_365": [availability_365]
})

# ---------------------------
# Make Predictions
# ---------------------------
rf_pred = rf_model.predict(input_data)[0]
xgb_pred = xgb_model.predict(input_data)[0]

# ---------------------------
# Show Results
# ---------------------------
st.subheader("💡 Predicted Price")
st.write(f"🌲 Random Forest Prediction: **${rf_pred:.2f}**")
st.write(f"🚀 XGBoost Prediction: **${xgb_pred:.2f}**")

st.markdown("---")
st.caption("Built with Streamlit, scikit-learn & XGBoost")

