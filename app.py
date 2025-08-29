import streamlit as st
import pandas as pd
import joblib
import requests
import io

# -------------------------------
# Load dataset from GitHub
# -------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/UjjwalKaushik0609/Airbnb_Report/refs/heads/main/cleaned_dataset.csv"
    df = pd.read_csv(url)
    return df

# -------------------------------
# Load models from Hugging Face
# -------------------------------
@st.cache_resource
def load_models():
    # Replace with your Hugging Face repo details
    rf_url = "https://huggingface.co/<your-username>/<your-hf-repo>/resolve/main/best_random_forest.pkl"
    xgb_url = "https://huggingface.co/<your-username>/<your-hf-repo>/resolve/main/best_xgboost.pkl"

    rf_model = joblib.load(io.BytesIO(requests.get(rf_url).content))
    xgb_model = joblib.load(io.BytesIO(requests.get(xgb_url).content))
    return rf_model, xgb_model

# -------------------------------
# Collect user input
# -------------------------------
def user_input(df):
    st.sidebar.header("Enter Airbnb Details")

    # Host identity verified
    host_identity_verified = st.sidebar.selectbox("Host Identity Verified", ["Yes", "No"])

    # Instant bookable
    instant_bookable = st.sidebar.selectbox("Instant Bookable", ["Yes", "No"])

    # Neighbourhood group
    neighbourhood_group = st.sidebar.selectbox("Neighbourhood Group", df["neighbourhood group"].unique())

    # Filtered neighbourhoods
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
    default_year = int(df["Construction year"].median())
    construction_year = st.sidebar.number_input("Construction Year", min_value=min_year, max_value=max_year, value=default_year)

    # Service Fee
    min_fee, max_fee = int(df["service fee"].min()), int(df["service fee"].max())
    default_fee = int(df["service fee"].median())
    service_fee = st.sidebar.number_input("Service Fee", min_value=min_fee, max_value=max_fee, value=default_fee)

    # Minimum Nights
    min_nights, max_nights = int(df["minimum nights"].min()), int(df["minimum nights"].max())
    default_nights = int(df["minimum nights"].median())
    minimum_nights = st.sidebar.number_input("Minimum Nights", min_value=min_nights, max_value=max_nights, value=default_nights)

    # Number of Reviews
    min_reviews, max_reviews = int(df["number of reviews"].min()), int(df["number of reviews"].max())
    default_reviews = int(df["number of reviews"].median())
    number_of_reviews = st.sidebar.number_input("Number of Reviews", min_value=min_reviews, max_value=max_reviews, value=default_reviews)

    # Reviews per Month
    min_rpm, max_rpm = float(df["reviews per month"].min()), float(df["reviews per month"].max())
    default_rpm = float(df["reviews per month"].median())
    reviews_per_month = st.sidebar.number_input("Reviews per Month", min_value=min_rpm, max_value=max_rpm, value=default_rpm)

    # Review Rate Number
    min_rrn, max_rrn = int(df["review rate number"].min()), int(df["review rate number"].max())
    default_rrn = int(df["review rate number"].median())
    review_rate_number = st.sidebar.number_input("Review Rate Number", min_value=min_rrn, max_value=max_rrn, value=default_rrn)

    # Host Listings Count
    min_list, max_list = int(df["calculated host listings count"].min()), int(df["calculated host listings count"].max())
    default_list = int(df["calculated host listings count"].median())
    calculated_host_listings_count = st.sidebar.number_input("Host Listings Count", min_value=min_list, max_value=max_list, value=default_list)

    # Availability
    min_avail, max_avail = int(df["availability 365"].min()), int(df["availability 365"].max())
    default_avail = int(df["availability 365"].median())
    availability_365 = st.sidebar.number_input("Availability (days per year)", min_value=min_avail, max_value=max_avail, value=default_avail)

    # Latitude
    min_lat, max_lat = float(df["lat"].min()), float(df["lat"].max())
    default_lat = float(df["lat"].median())
    lat = st.sidebar.number_input("Latitude", min_value=min_lat, max_value=max_lat, value=default_lat)

    # Longitude
    min_long, max_long = float(df["long"].min()), float(df["long"].max())
    default_long = float(df["long"].median())
    long = st.sidebar.number_input("Longitude", min_value=min_long, max_value=max_long, value=default_long)

    # Last Review Year (dropdown instead of free input)
    last_review_year = st.sidebar.selectbox("Last Review Year", sorted(df["last_review_year"].dropna().unique()))

    # Last Review Month
    last_review_month = st.sidebar.selectbox("Last Review Month", sorted(df["last_review_month"].dropna().unique()))

    # Model Choice
    model_choice = st.sidebar.radio("Choose Model", ("Random Forest", "XGBoost"))

    # Convert Yes/No → True/False
    host_identity_verified = True if host_identity_verified == "Yes" else False
    instant_bookable = True if instant_bookable == "Yes" else False

    # Create input DataFrame
    data = {
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
        "lat": [lat],
        "long": [long],
        "last_review_year": [last_review_year],
        "last_review_month": [last_review_month],
    }

    return pd.DataFrame(data), model_choice

# -------------------------------
# Main App
# -------------------------------
def main():
    st.title("🏡 Airbnb Price Prediction App")

    df = load_data()
    rf_model, xgb_model = load_models()

    input_df, model_choice = user_input(df)

    st.subheader("Your Input:")
    st.write(input_df)

    if st.button("Predict Price"):
        if model_choice == "Random Forest":
            prediction = rf_model.predict(input_df)[0]
        else:
            prediction = xgb_model.predict(input_df)[0]

        st.success(f"💰 Predicted Price: ${prediction:,.2f}")

if __name__ == "__main__":
    main()
