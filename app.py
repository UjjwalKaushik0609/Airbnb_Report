import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_dataset.csv")

df = load_data()

# ---------------------------
# Load Model (from Hugging Face)
# ---------------------------
@st.cache_resource
def load_model():
    url = "https://huggingface.co/UjjwalKaushik/Airbnb_model/resolve/main/best_random_forest.pkl"
    r = requests.get(url)
    r.raise_for_status()
    return joblib.load(io.BytesIO(r.content))

model = load_model()

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Airbnb Price Prediction", layout="wide")
st.title("🏡 Airbnb Price Prediction App")
st.write("This app predicts the **Total Cost** of an Airbnb listing based on its features.")

# ---------------------------
# Feature Lists
# ---------------------------
categorical = ['neighbourhood group', 'neighbourhood', 'country', 'country code',
               'cancellation_policy', 'room type']

numerical = ['host_identity_verified', 'lat', 'long', 'instant_bookable',
             'Construction year', 'service fee', 'minimum nights',
             'number of reviews', 'reviews per month', 'review rate number',
             'calculated host listings count', 'availability 365',
             'last_review_year', 'last_review_month']

all_features = categorical + numerical

# ---------------------------
# User Input
# ---------------------------
st.sidebar.header("Enter Listing Details")

def user_input():
    data = {}
    for col in categorical:
        options = df[col].dropna().unique().tolist()
        data[col] = st.sidebar.selectbox(f"{col}", options)

    for col in numerical:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        default_val = float(df[col].median())
        data[col] = st.sidebar.slider(f"{col}", min_val, max_val, default_val)
    
    return pd.DataFrame([data])

input_data = user_input()

st.subheader("🔹 Your Input Data")
st.write(input_data)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Total Cost"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Total Cost: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------------------
# Feature Importance
# ---------------------------
st.subheader("🔍 Feature Importance (from Random Forest)")

try:
    importances = model.named_steps["randomforestregressor"].feature_importances_
    feature_names = model.named_steps["preprocessor"].get_feature_names_out(all_features)

    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x="Importance", y="Feature", data=fi_df, ax=ax)
    ax.set_title("Top 15 Important Features")
    st.pyplot(fig)

except Exception as e:
    st.warning(f"Could not display feature importance: {e}")
