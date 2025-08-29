import streamlit as st
import pandas as pd
import joblib
import requests
import io

# ----------------------------
# Load Dataset from GitHub
# ----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/UjjwalKaushik0609/Airbnb_Report/refs/heads/main/cleaned_dataset.csv"
    df = pd.read_csv(url)
    return df

# ----------------------------
# Load Models from Hugging Face
# ----------------------------
@st.cache_resource
def load_models():
    rf_url = "https://huggingface.co/UjjwalKaushik/Airbnb_model/resolve/main/best_random_forest.pkl"
    xgb_url = "https://huggingface.co/UjjwalKaushik/Airbnb_model/resolve/main/best_xgboost.pkl"

    rf_model = joblib.load(io.BytesIO(requests.get(rf_url).content))
    xgb_model = joblib.load(io.BytesIO(requests.get(xgb_url).content))

    return rf_model, xgb_model

# ----------------------------
# Sidebar Input Form
# ----------------------------
def user_input(df):
    st.sidebar.header("Input Features")

    # Host identity verified
    host_identity_verified = st.sidebar.selectbox(
        "Host Identity Verified", ["Yes", "No"]
    )

    # Instant bookable
    instant_bookable = st.sidebar.selectbox(
        "Instant Bookable", ["Yes", "No"]
    )

    # Neighbourhood Group
    neighbourhood_group = st.sidebar.selectbox(
        "Neighbourhood Group", df["neighbourhood group"].unique()
    )

    # Filter neighbourhoods based on group
    neighbourhoods = df[df["neighbourhood group"] == neighbourhood_group]["neighbourhood"].unique()
    neighbourhood = st.sidebar.selectbox("Neighbourhood", neighbourhoods)

    # Country
    country = st.sidebar.selectbox("Country", df["country"].unique())

    # Cancellation policy
    cancellation_policy = st.sidebar.selectbox("Cancellation Policy", df["cancellation_policy"].unique())

    # Room type
    room_type = st.sidebar.selectbox("Room Type", df["room type"].unique())

    # Construction Year
    min_year, max_year = int(df["Construction year"].min()), int(df["Construction year"].max())
    default_year = int(df["Construction year"].median())
    construction_year = st.sidebar.number_input(
        "Construction Year", min_value=min_year, max_value=max_year, value=default_year
    )

    # Service fee
    min_fee, max_fee = int(df["service fee"].min()), int(df["service fee"].max())
    default_fee = int(df["service fee"].median())
    service_fee = st.sidebar.number_input(
        "Service Fee", min_value=min_fee, max_value=max_fee, value=default_fee
    )

    # Minimum nights
    min_nights, max_nights = int(df["minimum nights"].min()), int(df["minimum nights"].max())
    default_nights = int(df["minimum nights"].median())
    minimum_nights = st.sidebar.number_input(
        "Minimum Nights", min_value=min_nights, max_value=max_nights, value=default_nights
    )

    # Number of reviews
    min_reviews, max_reviews = int(df["number of reviews"].min()), int(df["number of reviews"].max())
    default_reviews = int(df["number of reviews"].median())
    number_of_reviews = st.sidebar.number_input(
        "Number of Reviews", min_value=min_reviews, max_value=max_reviews, value=default_reviews
    )

    # Reviews per month
    min_rpm, max_rpm = float(df["reviews per month"].min()), float(df["reviews per month"].max())
    default_rpm = float(df["reviews per month"].median())
    reviews_per_month = st.sidebar.number_input(
        "Reviews per Month", min_value=min_rpm, max_value=max_rpm, value=default_rpm
    )

    # Review rate number
    min_rr, max_rr = int(df["review rate number"].min()), int(df["review rate number"].max())
    default_rr = int(df["review rate number"].median())
    review_rate_number = st.sidebar.number_input(
        "Review Rate Number", min_value=min_rr, max_value=max_rr, value=default_rr
    )

    # Calculated host listings count
    min_hl, max_hl = int(df["calculated host listings count"].min()), int(df["calculated host listings count"].max())
    default_hl = int(df["calculated host listings count"].median())
    host_listings_count = st.sidebar.number_input(
        "Host Listings Count", min_value=min_hl, max_value=max_hl, value=default_hl
    )

    # Availability 365
    min_avail, max_avail = int(df["availability 365"].min()), int(df["availability 365"].max())
    default_avail = int(df["availability 365"].median())
    availability_365 = st.sidebar.number_input(
        "Availability (days)", min_value=min_avail, max_value=max_avail, value=default_avail
    )

    # Last Review Year (dropdown)
    years = sorted(df["last_review_year"].dropna().unique())
    last_review_year = st.sidebar.selectbox("Last Review Year", years)

    # Last Review Month
    months = sorted(df["last_review_month"].dropna().unique())
    last_review_month = st.sidebar.selectbox("Last Review Month", months)

    # Model choice
    model_choice = st.sidebar.radio("Choose Model", ["Random Forest", "XGBoost"])

    # Build input dataframe
    input_data = {
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
        "number of reviews": number_of_reviews,
        "reviews per month": reviews_per_month,
        "review rate number": review_rate_number,
        "calculated host listings count": host_listings_count,
        "availability 365": availability_365,
        "last_review_year": last_review_year,
        "last_review_month": last_review_month
    }

    return pd.DataFrame([input_data]), model_choice

# ----------------------------
# Main App
# ----------------------------
def main():
    st.title("Airbnb Price Prediction App 🏡")

    df = load_data()
    rf_model, xgb_model = load_models()

    input_df, model_choice = user_input(df)

    # Encode categorical values if necessary (same as training)
    input_encoded = pd.get_dummies(input_df)
    df_encoded = pd.get_dummies(df.drop(columns=["price"]))
    input_encoded = input_encoded.reindex(columns=df_encoded.columns, fill_value=0)

    # Select model
    if model_choice == "Random Forest":
        model = rf_model
    else:
        model = xgb_model

    # Predict price
    prediction = model.predict(input_encoded)[0]

    st.subheader("Predicted Price")
    st.success(f"${prediction:,.2f}")

if __name__ == "__main__":
    main()
