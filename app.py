import streamlit as st
import pandas as pd
import joblib
import os

# -------------------
# Load Data from GitHub
# -------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/UjjwalKaushik0609/Airbnb_Report/refs/heads/main/cleaned_dataset.csv"
    df = pd.read_csv(url)

    # Keep only relevant columns
    df = df[[
        "neighbourhood group", "neighbourhood", "room type", 
        "minimum nights", "number of reviews", "reviews per month",
        "calculated host listings count", "availability 365",
        "host_identity_verified", "instant_bookable", "Construction year", "price"
    ]]
    return df

df = load_data()

# -------------------
# Load Models from HF
# -------------------
@st.cache_resource
def load_models():
    rf_path = os.path.join(os.path.dirname(__file__), "best_random_forest.pkl")
    xgb_path = os.path.join(os.path.dirname(__file__), "best_xgboost.pkl")

    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)
    return rf_model, xgb_model

rf_model, xgb_model = load_models()

# -------------------
# User Input
# -------------------
def user_input(df):
    st.sidebar.header("Input Features")

    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost"])

    # Correct names from dataset
    neighbourhood_group = st.sidebar.selectbox("Neighbourhood Group", df["neighbourhood group"].unique())
    neighbourhoods = df[df["neighbourhood group"] == neighbourhood_group]["neighbourhood"].unique()
    neighbourhood = st.sidebar.selectbox("Neighbourhood", neighbourhoods)

    room_type = st.sidebar.selectbox("Room Type", df["room type"].unique())

    # Additional numerical features
    minimum_nights = st.sidebar.number_input("Minimum Nights", 1, 365, 3)
    number_of_reviews = st.sidebar.slider("Number of Reviews", 0, 500, 50)
    reviews_per_month = st.sidebar.number_input("Reviews per Month", 0.0, 30.0, 1.0)
    calculated_host_listings_count = st.sidebar.number_input("Host Listings Count", 1, 100, 1)
    availability_365 = st.sidebar.slider("Availability (days/year)", 0, 365, 180)

    # Extra features
    host_identity_verified = st.sidebar.selectbox("Host Identity Verified", ["t", "f"])
    instant_bookable = st.sidebar.selectbox("Instant Bookable", ["t", "f"])
    construction_year = st.sidebar.selectbox("Construction Year", sorted(df["Construction year"].dropna().unique()))

    # Collect into dataframe
    input_data = pd.DataFrame({
        "neighbourhood group": [neighbourhood_group],
        "neighbourhood": [neighbourhood],
        "room type": [room_type],
        "minimum nights": [minimum_nights],
        "number of reviews": [number_of_reviews],
        "reviews per month": [reviews_per_month],
        "calculated host listings count": [calculated_host_listings_count],
        "availability 365": [availability_365],
        "host_identity_verified": [host_identity_verified],
        "instant_bookable": [instant_bookable],
        "Construction year": [construction_year]
    })

    return input_data, model_choice

# -------------------
# Main
# -------------------
def main():
    st.title("🏡 Airbnb Price Prediction App")
    st.write("Predict listing **Price** based on input features.")

    input_df, model_choice = user_input(df)

    st.subheader("User Input:")
    st.write(input_df)

    # Prediction
    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_df)[0]
    else:
        prediction = xgb_model.predict(input_df)[0]

    st.subheader("💰 Predicted Price:")
    st.success(f"${prediction:,.2f}")

if __name__ == "__main__":
    main()
