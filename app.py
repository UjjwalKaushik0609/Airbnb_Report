import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ------------------------------
# Load Model from Hugging Face Hub
# ------------------------------
@st.cache_resource
def load_model(model_choice):
    if model_choice == "Random Forest":
        model_path = hf_hub_download(
            repo_id="UjjwalKaushik/Airbnb_model",
            filename="best_random_forest.pkl"
        )
    else:
        model_path = hf_hub_download(
            repo_id="UjjwalKaushik/Airbnb_model",
            filename="best_xgboost.pkl"
        )

    return joblib.load(model_path)

# ------------------------------
# Streamlit App
# ------------------------------
st.title("🏡 Airbnb Price Prediction")
st.markdown("Predict Airbnb prices using trained ML models (Random Forest / XGBoost).")

# Choose model
model_choice = st.selectbox("Choose a model:", ["Random Forest", "XGBoost"])

# Load selected model
model = load_model(model_choice)

# ------------------------------
# User Inputs
# ------------------------------
st.subheader("Enter Airbnb Listing Details")

room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
neighbourhood_group = st.selectbox("Neighbourhood Group", ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"])
minimum_nights = st.number_input("Minimum Nights", min_value=1, value=1)
number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=0)
reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, value=0.0, step=0.1)
calculated_host_listings_count = st.number_input("Host Listings Count", min_value=1, value=1)
availability_365 = st.slider("Availability (days per year)", 0, 365, 180)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        "room_type": [room_type],
        "neighbourhood_group": [neighbourhood_group],
        "minimum_nights": [minimum_nights],
        "number_of_reviews": [number_of_reviews],
        "reviews_per_month": [reviews_per_month],
        "calculated_host_listings_count": [calculated_host_listings_count],
        "availability_365": [availability_365]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Price: **${prediction:.2f}**")

