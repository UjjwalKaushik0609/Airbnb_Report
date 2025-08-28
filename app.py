import streamlit as st
import pandas as pd
import joblib
import requests
import os

# -----------------------
# Load Dataset
# -----------------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_dataset.csv")

df = load_data()

# Define categorical and numerical columns
categorical = ['neighbourhood group', 'neighbourhood', 'country', 'country code',
               'cancellation_policy', 'room type']

numerical = ['host_identity_verified', 'lat', 'long', 'instant_bookable',
             'Construction year', 'service fee', 'minimum nights',
             'number of reviews', 'reviews per month', 'review rate number',
             'calculated host listings count', 'availability 365',
             'last_review_year', 'last_review_month']

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model():
    model_path = "best_random_forest.pkl"

    # Download from Hugging Face if not available
    if not os.path.exists(model_path):
        url = "https://huggingface.co/UjjwalKaushik/Airbnb_model/resolve/main/best_random_forest.pkl"
        r = requests.get(url)
        r.raise_for_status()
        with open(model_path, "wb") as f:
            f.write(r.content)

    return joblib.load(model_path)

model = load_model()

# -----------------------
# User Input Function
# -----------------------
def user_input():
    data = {}
    for col in categorical:
        options = df[col].dropna().unique().tolist()
        data[col] = st.sidebar.selectbox(f"{col}", options)

    for col in numerical:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        default_val = float(df[col].median())

        if min_val == max_val:
            # Handle constant columns safely
            data[col] = st.sidebar.number_input(f"{col}", value=min_val)
        else:
            step = (max_val - min_val) / 100  # reasonable slider step
            data[col] = st.sidebar.slider(f"{col}", min_val, max_val, default_val, step=step)

    return pd.DataFrame([data])

# -----------------------
# Streamlit App Layout
# -----------------------
st.title("🏡 Airbnb Price Prediction App")

st.write("This app predicts the **Total Cost** of an Airbnb listing "
         "based on user-selected input features.")

input_data = user_input()

st.subheader("🔹 User Input")
st.write(input_data)

# -----------------------
# Make Prediction
# -----------------------
try:
    prediction = model.predict(input_data)[0]
    st.subheader("💰 Predicted Total Cost")
    st.success(f"${prediction:,.2f}")
except Exception as e:
    st.error(f"Error making prediction: {e}")
