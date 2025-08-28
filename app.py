import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt

# ------------------------------
# Load model from Hugging Face Hub
# ------------------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="UjjwalKaushik/Airbnb_model",   # Your Hugging Face repo
        filename="airbnb_model.pkl"             # File in repo
    )
    return joblib.load(model_path)

model = load_model()

# ------------------------------
# Load dataset for choices (optional, small sample for dropdowns)
# ------------------------------
@st.cache_data
def load_dataset():
    df = pd.read_csv("cleaned_dataset.csv")  # must be in repo
    return df

df = load_dataset()

# Extract options for dropdowns
neighbourhood_groups = df["neighbourhood group"].unique().tolist()
neighbourhoods_by_group = {
    group: df[df["neighbourhood group"] == group]["neighbourhood"].unique().tolist()
    for group in neighbourhood_groups
}
countries = df["country"].unique().tolist()
cancellation_policies = df["cancellation_policy"].unique().tolist()
room_types = df["room type"].unique().tolist()

# ------------------------------
# Sidebar Input Form
# ------------------------------
st.sidebar.header("Enter Airbnb Listing Details")

def user_input():
    # Dropdowns
    neighbourhood_group = st.sidebar.selectbox("Neighbourhood Group", neighbourhood_groups)
    neighbourhood = st.sidebar.selectbox("Neighbourhood", neighbourhoods_by_group[neighbourhood_group])
    country = st.sidebar.selectbox("Country", countries)
    country_code = st.sidebar.number_input("Country Code", min_value=1, step=1)
    cancellation_policy = st.sidebar.selectbox("Cancellation Policy", cancellation_policies)
    room_type = st.sidebar.selectbox("Room Type", room_types)

    # True/False as dropdown
    host_identity_verified = st.sidebar.selectbox("Host Identity Verified", [True, False])
    instant_bookable = st.sidebar.selectbox("Instant Bookable", [True, False])

    # Numeric Inputs
    lat = st.sidebar.slider("Latitude", float(df["lat"].min()), float(df["lat"].max()), float(df["lat"].mean()))
    long = st.sidebar.slider("Longitude", float(df["long"].min()), float(df["long"].max()), float(df["long"].mean()))
    construction_year = st.sidebar.number_input("Construction Year", min_value=1900, max_value=2025, step=1, value=2000)
    service_fee = st.sidebar.number_input("Service Fee", min_value=0, max_value=500, step=1, value=50)
    minimum_nights = st.sidebar.number_input("Minimum Nights", min_value=1, max_value=365, step=1, value=3)
    number_of_reviews = st.sidebar.number_input("Number of Reviews", min_value=0, max_value=1000, step=1, value=10)
    reviews_per_month = st.sidebar.slider("Reviews per Month", 0.0, 30.0, float(df["reviews per month"].mean()))
    review_rate_number = st.sidebar.slider("Review Rate Number", 0, 5, 4)
    calculated_host_listings_count = st.sidebar.number_input("Host Listings Count", min_value=0, max_value=100, step=1, value=1)
    availability_365 = st.sidebar.slider("Availability (days per year)", 0, 365, 180)

    # Date Inputs
    last_review_year = st.sidebar.selectbox("Last Review Year", sorted(df["last_review_year"].dropna().unique().astype(int)))
    last_review_month = st.sidebar.selectbox("Last Review Month", list(range(1, 13)))

    # Build dictionary
    data = {
        "neighbourhood group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "country": country,
        "country code": country_code,
        "cancellation_policy": cancellation_policy,
        "room type": room_type,
        "host_identity_verified": host_identity_verified,
        "instant_bookable": instant_bookable,
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
        "last_review_month": last_review_month
    }

    return pd.DataFrame([data])

input_data = user_input()

# ------------------------------
# Prediction
# ------------------------------
st.subheader("Prediction")
try:
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Total Cost: **${prediction:.2f}**")
except Exception as e:
    st.error(f"Error making prediction: {e}")

# ------------------------------
# Visualization
# ------------------------------
st.subheader("Visualizations")

# Example: Price distribution
fig, ax = plt.subplots()
df["total_cost"].hist(bins=50, ax=ax, color="#00C9A7")
ax.set_title("Distribution of Total Cost")
ax.set_xlabel("Total Cost")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Example: Availability vs Price
fig2, ax2 = plt.subplots()
ax2.scatter(df["availability 365"], df["total_cost"], alpha=0.3)
ax2.set_xlabel("Availability (days)")
ax2.set_ylabel("Total Cost")
ax2.set_title("Availability vs Price")
st.pyplot(fig2)
