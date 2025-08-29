import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# -----------------------------
# Load dataset from GitHub
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/UjjwalKaushik0609/Airbnb_Report/refs/heads/main/cleaned_dataset.csv"
    df = pd.read_csv(url)
    return df

# -----------------------------
# Load models from Hugging Face
# -----------------------------
@st.cache_resource
def load_models():
    repo_id = "UjjwalKaushik/Airbnb_model"

    rf_path = hf_hub_download(repo_id=repo_id, filename="best_random_forest.pkl")
    xgb_path = hf_hub_download(repo_id=repo_id, filename="best_xgboost.pkl")

    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)

    return rf_model, xgb_model

# -----------------------------
# Sidebar User Input
# -----------------------------
def user_input(df):
    st.sidebar.header("Input Features")

    # Model selection
    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost"])

    # Host identity verified (True/False)
    host_identity_verified = st.sidebar.selectbox(
        "Host Identity Verified", ["t", "f"]
    )

    # Instant bookable (True/False)
    instant_bookable = st.sidebar.selectbox(
        "Instant Bookable", ["t", "f"]
    )

    # Neighbourhood Group
    neighbourhood_group = st.sidebar.selectbox(
        "Neighbourhood Group", df["neighbourhood group"].unique()
    )

    # Filtered Neighbourhoods
    neighbourhood = st.sidebar.selectbox(
        "Neighbourhood",
        df[df["neighbourhood group"] == neighbourhood_group]["neighbourhood"].unique(),
    )

    # Country
    country = st.sidebar.selectbox("Country", df["country"].unique())

    # Cancellation Policy
    cancellation_policy = st.sidebar.selectbox(
        "Cancellation Policy", df["cancellation_policy"].unique()
    )

    # Room Type
    room_type = st.sidebar.selectbox("Room Type", df["room type"].unique())

    # Construction Year
    construction_year = st.sidebar.number_input(
        "Construction Year", int(df["Construction year"].min()), int(df["Construction year"].max()), 2000
    )

    # Service Fee
    service_fee = st.sidebar.number_input("Service Fee", 0, 500, 50)

    # Minimum Nights
    minimum_nights = st.sidebar.number_input("Minimum Nights", 1, 365, 1)

    # Reviews
    number_of_reviews = st.sidebar.number_input("Number of Reviews", 0, 1000, 10)
    reviews_per_month = st.sidebar.number_input("Reviews per Month", 0.0, 30.0, 1.0)
    review_rate_number = st.sidebar.number_input("Review Rate Number", 0, 10, 5)

    # Host Listings Count
    calculated_host_listings_count = st.sidebar.number_input(
        "Host Listings Count", 0, 50, 1
    )

    # Availability
    availability_365 = st.sidebar.number_input("Availability (days per year)", 0, 365, 180)

    # Last Review Year / Month
    last_review_year = st.sidebar.selectbox(
        "Last Review Year", sorted(df["last_review_year"].dropna().unique())
    )
    last_review_month = st.sidebar.selectbox(
        "Last Review Month", sorted(df["last_review_month"].dropna().unique())
    )

    # Lat/Long
    lat = st.sidebar.number_input("Latitude", float(df["lat"].min()), float(df["lat"].max()), float(df["lat"].mean()))
    long = st.sidebar.number_input("Longitude", float(df["long"].min()), float(df["long"].max()), float(df["long"].mean()))

    # Final Input Data
    input_data = pd.DataFrame(
        {
            "host_identity_verified": [host_identity_verified],
            "instant_bookable": [instant_bookable],
            "neighbourhood group": [neighbourhood_group],
            "neighbourhood": [neighbourhood],
            "country": [country],
            "cancellation_policy": [cancellation_policy],
            "room type": [room_type],
            "Construction year": [construction_year],
            "service fee": [service_fee],
            "minimum nights": [minimum_nights],
            "number of reviews": [number_of_reviews],
            "reviews per month": [reviews_per_month],
            "review rate number": [review_rate_number],
            "calculated host listings count": [calculated_host_listings_count],
            "availability 365": [availability_365],
            "last_review_year": [last_review_year],
            "last_review_month": [last_review_month],
            "lat": [lat],
            "long": [long],
        }
    )

    return input_data, model_choice

# -----------------------------
# Main App
# -----------------------------
def main():
    st.title("Airbnb Price Prediction App 🏡")

    df = load_data()
    rf_model, xgb_model = load_models()

    input_df, model_choice = user_input(df)

    st.subheader("User Input Summary")
    st.write(input_df)

    if st.button("Predict Price"):
        if model_choice == "Random Forest":
            prediction = rf_model.predict(input_df)[0]
        else:
            prediction = xgb_model.predict(input_df)[0]

        st.success(f"💰 Predicted Price: ${prediction:.2f}")

# -----------------------------
if __name__ == "__main__":
    main()
