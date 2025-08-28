import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io

# ----------------------------
# Load Model from Hugging Face
# ----------------------------
@st.cache_resource
def load_model():
    url = "https://huggingface.co/UjjwalKaushik/Airbnb_model/resolve/main/best_random_forest.pkl"
    r = requests.get(url)
    r.raise_for_status()
    with io.BytesIO(r.content) as f:
        model = joblib.load(f)
    return model

model = load_model()

# ----------------------------
# Define Feature Lists
# ----------------------------
categorical = [
    'neighbourhood group', 'neighbourhood', 'country', 'country code',
    'cancellation_policy', 'room type'
]

numerical = [
    'host_identity_verified', 'lat', 'long', 'instant_bookable',
    'Construction year', 'service fee', 'minimum nights',
    'number of reviews', 'reviews per month', 'review rate number',
    'calculated host listings count', 'availability 365',
    'last_review_year', 'last_review_month',
    'price'   # 👈 Added so pipeline matches
]

# ----------------------------
# Sidebar User Input
# ----------------------------
def user_input():
    data = {}

    # categorical
    for col in categorical:
        data[col] = st.sidebar.text_input(f"{col}")

    # numerical
    for col in numerical:
        min_val, max_val = 0, 10000
        default_val = 1
        # special ranges
        if col in ["lat"]:
            min_val, max_val, default_val = -90.0, 90.0, 40.0
        elif col in ["long"]:
            min_val, max_val, default_val = -180.0, 180.0, -73.0
        elif col in ["price", "service fee"]:
            min_val, max_val, default_val = 0, 2000, 100
        elif col in ["availability 365"]:
            min_val, max_val, default_val = 0, 365, 180
        elif col in ["last_review_month"]:
            min_val, max_val, default_val = 1, 12, 6

        data[col] = st.sidebar.slider(f"{col}", min_val, max_val, default_val)

    return pd.DataFrame([data])

# ----------------------------
# App Layout
# ----------------------------
st.title("🏡 Airbnb Price Prediction App")

st.write("Adjust the features in the sidebar and get a predicted **total cost**")

input_data = user_input()

st.subheader("Your Input:")
st.write(input_data)

# ----------------------------
# Prediction
# ----------------------------
try:
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Total Cost: **{prediction:.2f}**")
except Exception as e:
    st.error(f"Error making prediction: {e}")
