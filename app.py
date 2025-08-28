import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests
import io

# Dark theme setup
st.set_page_config(page_title="Airbnb Price Prediction", layout="wide")

st.title("🏠 Airbnb Price Prediction App")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_dataset.csv")

df = load_data()

# Load models from Hugging Face
@st.cache_resource
def load_model_from_hf(repo, filename):
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    response = requests.get(url)
    response.raise_for_status()
    return joblib.load(io.BytesIO(response.content))

rf_model = load_model_from_hf("UjjwalKaushik/Airbnb_model", "best_random_forest.pkl")
xgb_model = load_model_from_hf("UjjwalKaushik/Airbnb_model", "best_xgboost.pkl")

st.sidebar.header("Enter Airbnb Details")
# Example input fields (you should expand based on your dataset features)
guests = st.sidebar.number_input("Number of Guests", min_value=1, max_value=10, value=2)
bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=0, max_value=10, value=1)
bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=0, max_value=5, value=1)

input_data = pd.DataFrame({
    "guests": [guests],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms]
})

st.subheader("Model Predictions")
if st.sidebar.button("Predict Price"):
    rf_pred = rf_model.predict(input_data)[0]
    xgb_pred = xgb_model.predict(input_data)[0]
    
    st.write(f"🌲 Random Forest Prediction: **{rf_pred:.2f}**")
    st.write(f"🚀 XGBoost Prediction: **{xgb_pred:.2f}**")

# Show dataset overview
st.subheader("Dataset Overview")
st.dataframe(df.head())

# Plots
st.subheader("Exploratory Data Analysis")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df["total_cost"], bins=30, kde=True, ax=axes[0])
axes[0].set_title("Distribution of Total Cost")
sns.scatterplot(x="bedrooms", y="total_cost", data=df, ax=axes[1])
axes[1].set_title("Bedrooms vs Cost")
st.pyplot(fig)
