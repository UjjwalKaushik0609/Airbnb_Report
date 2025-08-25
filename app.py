import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import tempfile
import os

# ========================
# Google Drive File IDs
# ========================
REG_MODEL_ID = "1ENewUWOs_tiz4hZt8WUnRlcU6-naoTgu"
CLF_MODEL_ID = "1Bh7Ig0m8ZFg4gi2zdVGJ7JtZIkkxHHRp"
REG_FEAT_ID  = "1ajaBfSUiorYptqjjC7kKKrVFzYFHOeU-"
CLF_FEAT_ID  = "1IwHCTMGUtmjRNI-QHd-zU9Lr7r59cWOc"

def download_from_gdrive(file_id, is_gzip=False):
    """Download a file from Google Drive and load with joblib"""
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    response.raise_for_status()
    suffix = ".pkl.gz" if is_gzip else ".pkl"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    obj = joblib.load(tmp_path)
    os.remove(tmp_path)
    return obj

# ========================
# Load Models & Features
# ========================
@st.cache_resource
def load_models():
    reg_model = download_from_gdrive(REG_MODEL_ID, is_gzip=True)
    clf_model = download_from_gdrive(CLF_MODEL_ID, is_gzip=True)
    reg_features = download_from_gdrive(REG_FEAT_ID)
    clf_features = download_from_gdrive(CLF_FEAT_ID)
    return reg_model, clf_model, reg_features, clf_features

reg_model, clf_model, reg_features, clf_features = load_models()

# ========================
# Streamlit UI
# ========================
st.set_page_config(
    page_title="Airbnb Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Airbnb Price & Classification App")
st.write("Choose between **Price Prediction (Regression)** and **Category Prediction (Classification)**")

option = st.radio("Select Mode:", ["💲 Regression (Price Prediction)", "🔍 Classification (Category)"])

# ========================
# Input Section
# ========================
def user_input(features):
    st.subheader("Enter Airbnb Listing Details")
    data = {}
    for col in features:
        if "room" in col.lower() or "type" in col.lower():
            data[col] = st.selectbox(f"{col}", ["Option1", "Option2"])
        elif "yes" in col.lower() or "no" in col.lower() or "flag" in col.lower():
            data[col] = st.radio(f"{col}", ["Yes", "No"])
        else:
            data[col] = st.number_input(f"{col}", value=0.0)
    return pd.DataFrame([data])

# ========================
# Run Prediction
# ========================
if option.startswith("💲"):
    inputs = user_input(reg_features)
    if st.button("Predict Price"):
        pred = reg_model.predict(inputs)[0]
        st.success(f"Estimated Price: **${pred:,.2f}** per night")

else:
    inputs = user_input(clf_features)
    if st.button("Predict Category"):
        pred = clf_model.predict(inputs)[0]
        st.success(f"Predicted Category: **{pred}**")
