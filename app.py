import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown

st.set_page_config(page_title="Airbnb Price Prediction", page_icon="🏠", layout="wide")

st.title("🏠 Airbnb Price Prediction App")

# --- Download and Load Models ---
@st.cache_resource
def load_models():
    # --- Classifier Features (.pkl) ---
    url = "https://drive.google.com/uc?id=1IwHCTMGUtmjRNI-QHd-zU9Lr7r59cWOc"
    gdown.download(url, "clf_features.pkl", quiet=False)
    clf_features = joblib.load("clf_features.pkl")

    # --- Classifier Model (.pkl.gz) ---
    url = "https://drive.google.com/uc?id=1Bh7Ig0m8ZFg4gi2zdVGJ7JtZIkkxHHRp"
    gdown.download(url, "clf_model.pkl.gz", quiet=False)
    clf_model = joblib.load("clf_model.pkl.gz")

    # --- Regression Features (.pkl) ---
    url = "https://drive.google.com/uc?id=1ajaBfSUiorYptqjjC7kKKrVFzYFHOeU-"
    gdown.download(url, "reg_features.pkl", quiet=False)
    reg_features = joblib.load("reg_features.pkl")

    # --- Regression Model (.pkl.gz) ---
    url = "https://drive.google.com/uc?id=1ENewUWOs_tiz4hZt8WUnRlcU6-naoTgu"
    gdown.download(url, "reg_model.pkl.gz", quiet=False)
    reg_model = joblib.load("reg_model.pkl.gz")

    return clf_features, clf_model, reg_features, reg_model

clf_features, clf_model, reg_features, reg_model = load_models()

st.success("✅ Models loaded successfully from Google Drive!")

# --- Example Input Form ---
st.header("Input Features")
guest_count = st.number_input("Number of Guests", min_value=1, max_value=10, value=2)
room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])

if st.button("Predict"):
    st.write("🔮 This is where predictions would be made using the models.")
