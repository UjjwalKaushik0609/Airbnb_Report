import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# -------------------------------
# Load Dataset (for options only)
# -------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/UjjwalKaushik/Airbnb_model/main/airbnb_clean.csv"
    return pd.read_csv(url)

df = load_data()

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model(filename):
    model_path = hf_hub_download(
        repo_id="UjjwalKaushik/Airbnb_model",
        filename=filename
    )
    return joblib.load(model_path)

# -------------------------------
# Sidebar Inputs
# -------------------------------
def user_input():
    st.sidebar.header("🔧 Input Features")

    # Model selection
    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost"])
    model_file = "best_random_forest.pkl" if model_choice == "Random Forest" else "best_xgboost.pkl"
    model = load_model(model_file)

    # Neighbourhood Group
    neighbourhood_group = st.sidebar.selectbox("Neighbourhood Group", df["neighbourhood_group"].unique())

    # Neighbourhood auto-filter
    filtered_neighbourhoods = df[df["neighbourhood_group"] == neighbourhood_group]["neighbourhood"].unique()
    neighbourhood = st.sidebar.selectbox("Neighbourhood", filtered_neighbourhoods)

    # Room type
    room_type = st.sidebar.selectbox("Room Type", df["room_type"].unique())

    # Host Identity Verified
    host_identity_verified = st.sidebar.selectbox("Host Identity Verified", [True, False])

    # Instant Bookable
    instant_bookable = st.sidebar.selectbox("Instant Bookable", [True, False])

    # Year dropdown
    year = st.sidebar.selectbox("Year", sorted(df["year"].unique()))

    # Numerical inputs
    minimum_nights = st.sidebar.slider("Minimum Nights", 1, 365, 7)
    number_of_reviews = st.sidebar.slider("Number of Reviews", 0, 500, 10)
    reviews_per_month = st.sidebar.slider("Reviews per Month", 0.0, 10.0, 1.0)
    calculated_host_listings_count = st.sidebar.slider("Host Listings Count", 1, 50, 1)
    availability_365 = st.sidebar.slider("Availability (days)", 0, 365, 180)

    # Combine into DataFrame
    data = {
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "room_type": room_type,
        "host_identity_verified": host_identity_verified,
        "instant_bookable": instant_bookable,
        "year": year,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365
    }

    return pd.DataFrame([data]), model_choice, model

# -------------------------------
# Main App
# -------------------------------
def main():
    st.title("🏠 Airbnb Price Prediction")
    st.write("Predict the **price of an Airbnb listing** based on its features.")

    input_df, model_choice, model = user_input()

    st.subheader("🔍 Your Input Features")
    st.write(input_df)

    try:
        prediction = model.predict(input_df)[0]
        st.subheader(f"💰 Predicted Price ({model_choice}):")
        st.success(f"${prediction:,.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
