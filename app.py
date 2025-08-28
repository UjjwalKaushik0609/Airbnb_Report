import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Load dataset
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")
    return df

# =========================
# Load models
# =========================
@st.cache_resource
def load_model(model_name):
    if model_name == "Random Forest":
        return joblib.load("best_random_forest.pkl")
    elif model_name == "XGBoost":
        return joblib.load("best_xgboost.pkl")

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Airbnb Price Predictor", layout="wide")

st.title("🏠 Airbnb Price Prediction Dashboard")

# Load dataset
df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Input Features")

# Model choice
model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "XGBoost"])
model = load_model(model_choice)

# Feature inputs (use same columns used in training)
room_type = st.sidebar.selectbox("Room Type", df["room type"].unique())
neighbourhood_group = st.sidebar.selectbox("Neighbourhood Group", df["neighbourhood group"].unique())
neighbourhood = st.sidebar.selectbox("Neighbourhood", df["neighbourhood"].unique())
minimum_nights = st.sidebar.number_input("Minimum Nights", min_value=1, max_value=100, value=3)
number_of_reviews = st.sidebar.number_input("Number of Reviews", min_value=0, value=10)
reviews_per_month = st.sidebar.number_input("Reviews per Month", min_value=0.0, value=1.0, step=0.1)
availability_365 = st.sidebar.number_input("Availability (days)", min_value=0, max_value=365, value=180)

# =========================
# Prediction
# =========================
if st.sidebar.button("Predict Price"):
    try:
        # Prepare input
        input_dict = {
            "room type": [room_type],
            "neighbourhood group": [neighbourhood_group],
            "neighbourhood": [neighbourhood],
            "minimum nights": [minimum_nights],
            "number of reviews": [number_of_reviews],
            "reviews per month": [reviews_per_month],
            "availability 365": [availability_365],
        }

        input_df = pd.DataFrame(input_dict)

        # Ensure categories align with training
        for col in ["room type", "neighbourhood group", "neighbourhood"]:
            input_df[col] = input_df[col].astype(str)

        # Prediction
        pred = model.predict(input_df)[0]
        st.success(f"💰 Predicted Price: **${pred:.2f}** using {model_choice}")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# =========================
# Data Preview & Visualization
# =========================
st.subheader("📊 Dataset Preview")
st.write(df.head())

# Simple visualization
st.subheader("Room Type Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x="room type", ax=ax)
st.pyplot(fig)

st.subheader("Price Distribution by Neighbourhood Group")
fig, ax = plt.subplots()
sns.boxplot(data=df, x="neighbourhood group", y="price", ax=ax)
st.pyplot(fig)
