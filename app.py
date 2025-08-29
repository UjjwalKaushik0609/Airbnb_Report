import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from huggingface_hub import hf_hub_download

# --------------------------
# Dataset + Models
# --------------------------
DATA_URL = "https://raw.githubusercontent.com/UjjwalKaushik0609/Airbnb_Report/refs/heads/main/cleaned_dataset.csv"

# Hugging Face repo
HF_REPO = "UjjwalKaushik/Airbnb_model"
RF_FILE = "best_random_forest.pkl"
XGB_FILE = "best_xgboost.pkl"

# --------------------------
# Load dataset
# --------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        return df
    except Exception as e:
        st.error(f"❌ Could not load dataset: {e}")
        return pd.DataFrame()

# --------------------------
# Load models from Hugging Face Hub
# --------------------------
@st.cache_resource
def load_models():
    try:
        rf_path = hf_hub_download(repo_id=HF_REPO, filename=RF_FILE)
        xgb_path = hf_hub_download(repo_id=HF_REPO, filename=XGB_FILE)

        rf_model = joblib.load(rf_path)
        xgb_model = joblib.load(xgb_path)

        return rf_model, xgb_model
    except Exception as e:
        st.error(f"❌ Could not load models from Hugging Face: {e}")
        return None, None

# --------------------------
# User Input Form
# --------------------------
def user_input(df):
    st.sidebar.header("Enter Listing Details")

    # host_identity_verified & instant_bookable → Yes/No
    host_identity_verified = st.sidebar.selectbox("Host Identity Verified", ["Yes", "No"])
    instant_bookable = st.sidebar.selectbox("Instant Bookable", ["Yes", "No"])

    # neighbourhood_group → neighbourhood
    ng = st.sidebar.selectbox("Neighbourhood Group", sorted(df["neighbourhood group"].dropna().unique()))
    nhoods = df[df["neighbourhood group"] == ng]["neighbourhood"].dropna().unique()
    neighbourhood = st.sidebar.selectbox("Neighbourhood", sorted(nhoods))

    # categorical
    country = st.sidebar.selectbox("Country", sorted(df["country"].dropna().unique()))
    cancellation_policy = st.sidebar.selectbox("Cancellation Policy", sorted(df["cancellation_policy"].dropna().unique()))
    room_type = st.sidebar.selectbox("Room Type", sorted(df["room type"].dropna().unique()))

    # numeric inputs with safe defaults
    lat = st.sidebar.number_input("Latitude", float(df["lat"].min()), float(df["lat"].max()), float(df["lat"].median()))
    long = st.sidebar.number_input("Longitude", float(df["long"].min()), float(df["long"].max()), float(df["long"].median()))
    service_fee = st.sidebar.number_input("Service Fee", float(df["service fee"].min()), float(df["service fee"].max()), float(df["service fee"].median()))
    minimum_nights = st.sidebar.number_input("Minimum Nights", int(df["minimum nights"].min()), int(df["minimum nights"].max()), int(df["minimum nights"].median()))
    number_of_reviews = st.sidebar.number_input("Number of Reviews", int(df["number of reviews"].min()), int(df["number of reviews"].max()), int(df["number of reviews"].median()))
    reviews_per_month = st.sidebar.number_input("Reviews per Month", float(df["reviews per month"].min()), float(df["reviews per month"].max()), float(df["reviews per month"].median()))
    review_rate_number = st.sidebar.number_input("Review Rate Number", int(df["review rate number"].min()), int(df["review rate number"].max()), int(df["review rate number"].median()))
    host_listings_count = st.sidebar.number_input("Calculated Host Listings Count", int(df["calculated host listings count"].min()), int(df["calculated host listings count"].max()), int(df["calculated host listings count"].median()))
    availability_365 = st.sidebar.number_input("Availability 365", int(df["availability 365"].min()), int(df["availability 365"].max()), int(df["availability 365"].median()))
    construction_year = st.sidebar.number_input("Construction Year", int(df["Construction year"].min()), int(df["Construction year"].max()), int(df["Construction year"].median()))

    # last_review_year & month → dropdown
    last_review_year = st.sidebar.selectbox("Last Review Year", sorted(df["last_review_year"].dropna().unique()))
    last_review_month = st.sidebar.selectbox("Last Review Month", sorted(df["last_review_month"].dropna().unique()))

    # convert Yes/No → 1/0
    host_identity_verified = 1 if host_identity_verified == "Yes" else 0
    instant_bookable = 1 if instant_bookable == "Yes" else 0

    # build input row (exclude price!)
    data = {
        "host_identity_verified": host_identity_verified,
        "instant_bookable": instant_bookable,
        "neighbourhood group": ng,
        "neighbourhood": neighbourhood,
        "country": country,
        "cancellation_policy": cancellation_policy,
        "room type": room_type,
        "lat": lat,
        "long": long,
        "service fee": service_fee,
        "minimum nights": minimum_nights,
        "number of reviews": number_of_reviews,
        "reviews per month": reviews_per_month,
        "review rate number": review_rate_number,
        "calculated host listings count": host_listings_count,
        "availability 365": availability_365,
        "Construction year": construction_year,
        "last_review_year": last_review_year,
        "last_review_month": last_review_month
    }

    return pd.DataFrame([data])

# --------------------------
# Main App
# --------------------------
def main():
    st.title("🏡 Airbnb Price Prediction App")

    df = load_data()
    rf_model, xgb_model = load_models()

    if df.empty or rf_model is None or xgb_model is None:
        st.error("⚠️ Could not load dataset/models. Please check paths.")
        return

    input_df = user_input(df)
    st.write("### Your Input:")
    st.dataframe(input_df)

    # ensure input has same features as training (drop 'price')
    feature_cols = [col for col in df.columns if col != "price"]
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "XGBoost"])

    if st.sidebar.button("Predict Price"):
        try:
            if model_choice == "Random Forest":
                pred = rf_model.predict(input_df)[0]
            else:
                pred = xgb_model.predict(input_df)[0]

            st.success(f"💰 Predicted Price: ${pred:,.2f}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

if __name__ == "__main__":
    main()
