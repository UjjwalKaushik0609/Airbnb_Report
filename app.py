import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# -------------------------------
# Safe Run Wrapper
# -------------------------------
def safe_run(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        return None

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/UjjwalKaushik0609/Airbnb_Report/refs/heads/main/cleaned_dataset.csv"
    return safe_run(pd.read_csv, url)

# -------------------------------
# Load Models from Hugging Face Hub
# -------------------------------
@st.cache_resource
def load_models():
    try:
        rf_path = hf_hub_download(repo_id="UjjwalKaushik/Airbnb_model", filename="best_random_forest.pkl")
        xgb_path = hf_hub_download(repo_id="UjjwalKaushik/Airbnb_model", filename="best_xgboost.pkl")

        rf_model = joblib.load(rf_path)
        xgb_model = joblib.load(xgb_path)
        return rf_model, xgb_model
    except Exception as e:
        st.error(f"⚠️ Failed to load models: {e}")
        return None, None

# -------------------------------
# User Input
# -------------------------------
def user_input(df):
    st.sidebar.header("Filter Options")

    # Host Identity Verified
    host_identity_verified = st.sidebar.selectbox(
        "Host Identity Verified", df["host_identity_verified"].unique()
    )

    # Instant Bookable
    instant_bookable = st.sidebar.selectbox(
        "Instant Bookable", df["instant_bookable"].unique()
    )

    # Neighbourhood Group → filters neighbourhood
    neighbourhood_group = st.sidebar.selectbox(
        "Neighbourhood Group", df["neighbourhood group"].unique()
    )
    neighbourhoods = df[df["neighbourhood group"] == neighbourhood_group]["neighbourhood"].unique()
    neighbourhood = st.sidebar.selectbox("Neighbourhood", neighbourhoods)

    # Country
    country = st.sidebar.selectbox("Country", df["country"].unique())

    # Cancellation Policy
    cancellation_policy = st.sidebar.selectbox("Cancellation Policy", df["cancellation_policy"].unique())

    # Room Type
    room_type = st.sidebar.selectbox("Room Type", df["room type"].unique())

    # Construction Year
    min_year, max_year = int(df["Construction year"].min()), int(df["Construction year"].max())
    construction_year = st.sidebar.number_input(
        "Construction Year", min_value=min_year, max_value=max_year,
        value=int(df["Construction year"].median())
    )

    # Service Fee
    min_fee, max_fee = float(df["service fee"].min()), float(df["service fee"].max())
    service_fee = st.sidebar.number_input(
        "Service Fee", min_value=min_fee, max_value=max_fee,
        value=float(df["service fee"].median())
    )

    # Minimum Nights
    min_nights, max_nights = int(df["minimum nights"].min()), int(df["minimum nights"].max())
    minimum_nights = st.sidebar.number_input(
        "Minimum Nights", min_value=min_nights, max_value=max_nights,
        value=int(df["minimum nights"].median())
    )

    # Reviews
    num_reviews = st.sidebar.number_input(
        "Number of Reviews", min_value=int(df["number of reviews"].min()), max_value=int(df["number of reviews"].max()),
        value=int(df["number of reviews"].median())
    )
    reviews_per_month = st.sidebar.number_input(
        "Reviews per Month", min_value=float(df["reviews per month"].min()), max_value=float(df["reviews per month"].max()),
        value=float(df["reviews per month"].median())
    )
    review_rate = st.sidebar.number_input(
        "Review Rate Number", min_value=float(df["review rate number"].min()), max_value=float(df["review rate number"].max()),
        value=float(df["review rate number"].median())
    )

    # Availability
    availability = st.sidebar.number_input(
        "Availability 365", min_value=int(df["availability 365"].min()), max_value=int(df["availability 365"].max()),
        value=int(df["availability 365"].median())
    )

    # Last Review Year & Month (Dropdowns)
    last_review_year = st.sidebar.selectbox("Last Review Year", sorted(df["last_review_year"].dropna().unique()))
    last_review_month = st.sidebar.selectbox("Last Review Month", sorted(df["last_review_month"].dropna().unique()))

    # Model choice
    model_choice = st.sidebar.radio("Select Model", ["Random Forest", "XGBoost"])

    # Input DataFrame
    input_dict = {
        "host_identity_verified": host_identity_verified,
        "instant_bookable": instant_bookable,
        "neighbourhood group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "country": country,
        "cancellation_policy": cancellation_policy,
        "room type": room_type,
        "Construction year": construction_year,
        "service fee": service_fee,
        "minimum nights": minimum_nights,
        "number of reviews": num_reviews,
        "reviews per month": reviews_per_month,
        "review rate number": review_rate,
        "availability 365": availability,
        "last_review_year": last_review_year,
        "last_review_month": last_review_month,
    }

    return pd.DataFrame([input_dict]), model_choice

# -------------------------------
# Main App
# -------------------------------
def main():
    st.title("🏡 Airbnb Price Prediction App")

    df = load_data()
    if df is None:
        return

    rf_model, xgb_model = load_models()
    if rf_model is None or xgb_model is None:
        return

    input_df, model_choice = user_input(df)

    st.subheader("Your Input:")
    st.write(input_df)

    if st.button("Predict Price"):
        try:
            if model_choice == "Random Forest":
                prediction = rf_model.predict(input_df)
            else:
                prediction = xgb_model.predict(input_df)

            st.success(f"💰 Predicted Price: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

if __name__ == "__main__":
    main()
