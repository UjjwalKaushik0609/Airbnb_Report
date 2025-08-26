import streamlit as st
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Airbnb Price Prediction",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------
# Load Models & Features from Hugging Face
# -----------------
REPO_ID = "UjjwalKaushik/Airbnb_model"

@st.cache_resource
def load_models():
    # Download files
    clf_pipeline_path = hf_hub_download(repo_id=REPO_ID, filename="clf_pipeline.pkl")
    reg_pipeline_path = hf_hub_download(repo_id=REPO_ID, filename="reg_pipeline.pkl")
    clf_features_path = hf_hub_download(repo_id=REPO_ID, filename="clf_features.pkl")
    reg_features_path = hf_hub_download(repo_id=REPO_ID, filename="reg_features.pkl")

    # Load
    clf_pipeline = joblib.load(clf_pipeline_path)
    reg_pipeline = joblib.load(reg_pipeline_path)
    clf_features = joblib.load(clf_features_path)
    reg_features = joblib.load(reg_features_path)

    return clf_pipeline, reg_pipeline, clf_features, reg_features

clf_pipeline, reg_pipeline, clf_features, reg_features = load_models()

# -----------------
# Streamlit UI
# -----------------
st.title("🏠 Airbnb Prediction App")
st.markdown("Dark theme enabled. Predict **category (expensive/affordable)** and **price** from features.")

task = st.radio("Choose Task", ["💰 Price Prediction", "🏷️ Category Prediction"])

# Collect user input dynamically
input_data = {}
st.header("Enter Airbnb details")

# Use reg_features for price input fields
features_to_use = reg_features if task == "💰 Price Prediction" else clf_features

for col in features_to_use:
    input_data[col] = st.number_input(f"{col}", value=0.0)

user_df = pd.DataFrame([input_data])

# Category prediction
if task == "🏷️ Category Prediction":
    if st.button("🔮 Predict Category"):
        pred_class = clf_pipeline.predict(user_df[clf_features])[0]
        st.success(f"Prediction: {'Expensive 💰' if pred_class == 1 else 'Affordable 🏡'}")

# Price prediction
if task == "💰 Price Prediction":
    if st.button("💵 Predict Price"):
        pred_price = reg_pipeline.predict(user_df[reg_features])[0]
        st.success(f"Predicted Price: ${pred_price:,.2f}")

