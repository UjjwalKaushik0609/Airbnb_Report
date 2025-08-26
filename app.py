import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

# Load models from Hugging Face
@st.cache_resource
def load_models():
    # Download and cache regression pipeline
    reg_model_path = hf_hub_download(
        repo_id="your-username/your-repo",
        filename="reg_pipeline.pkl"
    )
    reg_pipeline = joblib.load(reg_model_path)

    # Download and cache classification pipeline
    clf_model_path = hf_hub_download(
        repo_id="your-username/your-repo",
        filename="clf_pipeline.pkl"
    )
    clf_pipeline = joblib.load(clf_model_path)

    return reg_pipeline, clf_pipeline


st.title("🏠 Airbnb Price Prediction App")

reg_pipeline, clf_pipeline = load_models()

# Example input UI
st.header("Enter Airbnb Listing Features")

room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
neighbourhood = st.text_input("Neighbourhood")
minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=3)
availability = st.slider("Availability 365", 0, 365, 180)

if st.button("Predict"):
    # Example feature dict (adjust to match your feature engineering)
    features = {
        "room_type": room_type,
        "neighbourhood": neighbourhood,
        "minimum_nights": minimum_nights,
        "availability_365": availability,
    }

    # Convert to dataframe
    input_df = pd.DataFrame([features])

    # Predict
    price_pred = reg_pipeline.predict(input_df)[0]
    category_pred = clf_pipeline.predict(input_df)[0]

    st.success(f"💰 Predicted Price: ${price_pred:.2f}")
    st.info(f"📊 Category: {category_pred}")
