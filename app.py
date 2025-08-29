import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# -------------------------
# Load dataset with exception handling
# -------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/UjjwalKaushik0609/Airbnb_Report/refs/heads/main/cleaned_dataset.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        return pd.DataFrame()  # fallback

# -------------------------
# Load models with Hugging Face Hub
# -------------------------
@st.cache_resource
def load_models():
    try:
        rf_path = hf_hub_download(repo_id="UjjwalKaushik/Airbnb_model", filename="best_random_forest.pkl")
        xgb_path = hf_hub_download(repo_id="UjjwalKaushik/Airbnb_model", filename="best_xgboost.pkl")

        rf_model = joblib.load(rf_path)
        xgb_model = joblib.load(xgb_path)

        return rf_model, xgb_model
    except Exception as e:
        st.error(f"❌ Failed to load models from Hugging Face: {e}")
        return None, None

# -------------------------
# Sidebar input function
# -------------------------
def user_input(df):
    st.sidebar.header("Filter Options")

    # --- Host Identity Verified ---
    host_identity_verified = st.sidebar.selectbox("Host Identity Verified", ["Yes", "No"])
    host_identity_verified = 1 if host_identity_verified == "Yes" else 0

    # --- Instant Bookable ---
    instant_bookable = st.sidebar.selectbox("Instant Bookable", ["Yes", "No"])
    instant_bookable = 1 if instant_bookable == "Yes" else 0

    # --- Neighbourhood Group & Neighbourhood ---
    neighbourhood_group = st.sidebar.selectbox("Neighbourhood Group", df["neighbourhood group"].unique())
    neighbourhoods = df[df["neighbourhood group"] == neighbourhood_group]["neighbourhood"].unique()
    neighbourhood = st.sidebar.selectbox("Neighbourhood", neighbourhoods)

    # --- Country Code ---
    country_code = st.sidebar.selectbox("Country Code", df["country code"].unique())

    # --- Cancellation Policy ---
    cancellation_policy = st.sidebar.selectbox("Cancellation Policy", df["cancellation_policy"].unique())

    # --- Room Type ---
    room_type = st.sidebar.selectbox("Room Type", df["room type"].unique())

    # --- Last Review Year ---
    last_review_year = st.sidebar.selectbox("Last Review Year", sorted(df["last_review_year"].dropna().unique()))

    # --- Numeric Inputs with safe defaults ---
    lat = st.sidebar.number_input("Latitude",
        float(df["lat"].min()), float(df["lat"].max()), float(df["lat"].median())
    )
    long = st.sidebar.number_input("Longitude",
        float(df["long"].min()), float(df["long"].max()), float(df["long"].median())
    )
    construction_year = st.sidebar.number_input("Construction Year",
        int(df["Construction year"].min()), int(df["Construction year"].max()), int(df["Construction year"].median())
    )
    service_fee = st.sidebar.number_input("Service Fee",
        float(df["service fee"].min()), float(df["service fee"].max()), float(df["service fee"].median())
    )
    minimum_nights = st.sidebar.number_input("Minimum Nights",
        int(df["minimum nights"].min()), int(df["minimum nights"].max()), int(df["minimum nights"].median())
    )
    number_of_reviews = st.sidebar.number_input("Number of Reviews",
        int(df["number of reviews"].min()), int(df["number of reviews"].max()), int(df["number of reviews"].median())
    )
    reviews_per_month = st.sidebar.number_input("Reviews per Month",
        float(df["reviews per month"].min()), float(df["reviews per month"].max()), float(df["reviews per month"].median())
    )
    review_rate_number = st.sidebar.number_input("Review Rate Number",
        int(df["review rate number"].min()), int(df["review rate number"].max()), int(df["review rate number"].median())
    )
    calculated_host_listings_count = st.sidebar.number_input("Calculated Host Listings Count",
        int(df["calculated host listings count"].min()), int(df["calculated host listings count"].max()), int(df["calculated host listings count"].median())
    )
    availability_365 = st.sidebar.number_input("Availability (365 days)",
        int(df["availability 365"].min()), int(df["availability 365"].max()), int(df["availability 365"].median())
    )

    # --- Create input DataFrame ---
    input_dict = {
        "host_identity_verified": host_identity_verified,
        "instant_bookable": instant_bookable,
        "neighbourhood group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "country code": country_code,
        "cancellation_policy": cancellation_policy,
        "room type": room_type,
        "lat": lat,
        "long": long,
        "Construction year": construction_year,
        "service fee": service_fee,
        "minimum nights": minimum_nights,
        "number of reviews": number_of_reviews,
        "reviews per month": reviews_per_month,
        "review rate number": review_rate_number,
        "calculated host listings count": calculated_host_listings_count,
        "availability 365": availability_365,
        "last_review_year": last_review_year,
    }
    return pd.DataFrame([input_dict])

# -------------------------
# Main app
# -------------------------
def main():
    st.title("🏡 Airbnb Price Prediction App")

    df = load_data()
    if df.empty:
        st.stop()

    rf_model, xgb_model = load_models()
    if rf_model is None or xgb_model is None:
        st.stop()

    input_df = user_input(df)

    st.subheader("Your Input")
    st.write(input_df)

    try:
        rf_pred = rf_model.predict(input_df)[0]
        xgb_pred = xgb_model.predict(input_df)[0]

        st.subheader("Predicted Price")
        st.write(f"🌲 Random Forest: **${rf_pred:.2f}**")
        st.write(f"⚡ XGBoost: **${xgb_pred:.2f}**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

if __name__ == "__main__":
    main()
