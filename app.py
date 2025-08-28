import streamlit as st
import pandas as pd
import joblib
import requests
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Airbnb Price Prediction", layout="wide")

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_dataset.csv")

df = load_data()

# ---------------------------
# Load Model from Hugging Face
# ---------------------------
@st.cache_resource
def load_model(model_name):
    urls = {
        "Random Forest": "https://huggingface.co/UjjwalKaushik/Airbnb_model/resolve/main/best_random_forest.pkl",
        "XGBoost": "https://huggingface.co/UjjwalKaushik/Airbnb_model/resolve/main/best_xgboost.pkl",
    }

    url = urls[model_name]
    filename = url.split("/")[-1]

    if not os.path.exists(filename):
        with st.spinner(f"Downloading {model_name} model..."):
            r = requests.get(url)
            r.raise_for_status()
            with open(filename, "wb") as f:
                f.write(r.content)

    return joblib.load(filename)

# ---------------------------
# Sidebar - Select Model
# ---------------------------
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio("Choose a model", ["Random Forest", "XGBoost"])
model = load_model(model_choice)

# ---------------------------
# Sidebar - User Input
# ---------------------------
st.sidebar.header("Prediction Input")

room_type = st.sidebar.selectbox("Room Type", df["room type"].unique())
neighbourhood_group = st.sidebar.selectbox("Neighbourhood Group", df["neighbourhood group"].unique())
minimum_nights = st.sidebar.number_input("Minimum Nights", min_value=1, max_value=1000, value=3)
number_of_reviews = st.sidebar.number_input("Number of Reviews", min_value=0, max_value=500, value=10)
reviews_per_month = st.sidebar.number_input("Reviews per Month", min_value=0.0, max_value=30.0, value=1.0)
availability_365 = st.sidebar.number_input("Availability (days/year)", min_value=0, max_value=365, value=180)

# ---------------------------
# Prediction
# ---------------------------
input_data = pd.DataFrame({
    "room type": [room_type],
    "neighbourhood group": [neighbourhood_group],
    "minimum nights": [minimum_nights],
    "number of reviews": [number_of_reviews],
    "reviews per month": [reviews_per_month],
    "availability 365": [availability_365]
})

if st.sidebar.button("Predict Price"):
    pred = model.predict(input_data)[0]
    st.success(f"💰 Estimated Price: **${pred:.2f}** using {model_choice}")

# ---------------------------
# Main Dashboard
# ---------------------------
st.title("🏠 Airbnb Data Explorer & Price Prediction")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------------
# Visualizations
# ---------------------------
st.subheader("Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.write("### Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["price"], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

with col2:
    st.write("### Room Type Count")
    fig, ax = plt.subplots()
    df["room type"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

st.write("### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
