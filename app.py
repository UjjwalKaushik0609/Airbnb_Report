import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ---------------------------
# Load trained model
# ---------------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="UjjwalKaushik/Airbnb_model",
        filename="airbnb_model.pkl"
    )
    return joblib.load(model_path)

model = load_model()

# ---------------------------
# Load dataset (for options)
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_dataset.csv")

df = load_data()

# ---------------------------
# Sidebar Input Function
# ---------------------------
def user_input():
    data = {}

    # 🔹 Host Identity Verified (True/False)
    data["host_identity_verified"] = st.sidebar.selectbox(
        "Host Identity Verified", [True, False]
    )

    # 🔹 Instant Bookable (True/False)
    data["instant_bookable"] = st.sidebar.selectbox(
        "Instant Bookable", [True, False]
    )

    # 🔹 Neighbourhood Group
    neighbourhood_group = st.sidebar.selectbox(
        "Neighbourhood Group", df["neighbourhood group"].unique()
    )

    # 🔹 Neighbourhood (filtered by group)
    filtered_neighbourhoods = df[df["neighbourhood group"] == neighbourhood_group]["neighbourhood"].unique()
    data["neighbourhood group"] = neighbourhood_group
    data["neighbourhood"] = st.sidebar.selectbox("Neighbourhood", filtered_neighbourhoods)

    # 🔹 Country
    data["country"] = st.sidebar.selectbox("Country", df["country"].unique())

    # 🔹 Country Code
    data["country code"] = st.sidebar.selectbox("Country Code", df["country code"].unique())

    # 🔹 Cancellation Policy
    data["cancellation_policy"] = st.sidebar.selectbox("Cancellation Policy", df["cancellation_policy"].unique())

    # 🔹 Room Type
    data["room type"] = st.sidebar.selectbox("Room Type", df["room type"].unique())

    # 🔹 Construction Year (dropdown)
    years = list(range(int(df["Construction year"].min()), int(df["Construction year"].max()) + 1))
    data["Construction year"] = st.sidebar.selectbox("Construction Year", years)

    # 🔹 Other numerical features with sliders
    numerical = [
        'lat', 'long', 'service fee', 'minimum nights',
        'number of reviews', 'reviews per month', 'review rate number',
        'calculated host listings count', 'availability 365',
        'last_review_year', 'last_review_month'
    ]

    for col in numerical:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        default_val = float(df[col].mean())
        data[col] = st.sidebar.slider(f"{col}", min_val, max_val, default_val)

    return pd.DataFrame([data])

# ---------------------------
# Streamlit App
# ---------------------------
st.title("🏡 Airbnb Price Prediction")
st.write("Fill in the details in the sidebar to predict the Airbnb listing price.")

# Get input
input_data = user_input()

# Prediction
try:
    prediction = model.predict(input_data)[0]
    st.subheader("💰 Predicted Price:")
    st.success(f"${prediction:,.2f}")
except Exception as e:
    st.error(f"Error making prediction: {e}")
