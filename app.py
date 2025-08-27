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
# Load Models & Features
# -----------------
REPO_ID = "UjjwalKaushik/Airbnb_model"

@st.cache_resource
def load_models():
    clf_pipeline_path = hf_hub_download(repo_id=REPO_ID, filename="clf_pipeline.pkl")
    reg_pipeline_path = hf_hub_download(repo_id=REPO_ID, filename="reg_pipeline.pkl")
    clf_features_path = hf_hub_download(repo_id=REPO_ID, filename="clf_features.pkl")
    reg_features_path = hf_hub_download(repo_id=REPO_ID, filename="reg_features.pkl")

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
st.markdown("Predict **category (expensive/affordable)** and **price** from features.")

task = st.radio("Choose Task", ["💰 Price Prediction", "🏷️ Category Prediction"])

# -----------------
# Collect User Input
# -----------------
st.header("Enter Airbnb Details")

features_to_use = reg_features if task == "💰 Price Prediction" else clf_features
input_data = {}

# --- Always show categorical dropdowns ---
input_data["host_is_verified"] = st.selectbox("Host Verified?", ["verified", "not verified"])
input_data["neighbourhood_group"] = st.selectbox("Neighbourhood Group", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
input_data["neighbourhood"] = st.selectbox("Neighbourhood", ["Harlem", "Midtown", "Williamsburg", "Astoria", "Chelsea"])

# Max listings in dataset (adjust if needed)
MAX_LISTINGS = 100  

# --- Other features ---
for col in features_to_use:

    if col in ["host_is_verified", "neighbourhood_group", "neighbourhood"]:
        continue  # already handled above

    elif col == "host_id":
        input_data[col] = st.number_input("Host ID", min_value=1, max_value=1000000, value=100, step=1)

    elif col == "latitude":
        input_data[col] = st.slider("Latitude", -90.0, 90.0, 40.0)

    elif col == "longitude":
        input_data[col] = st.slider("Longitude", -180.0, 180.0, -74.0)

    elif col == "construction_year":
        input_data[col] = st.number_input("Construction Year", min_value=1900, max_value=2025, value=2010, step=1)

    elif col == "service_fee":
        input_data[col] = st.slider("Service Fee ($)", 0, 500, 50)

    elif col == "minimum_nights":
        input_data[col] = st.slider("Minimum Nights", 1, 365, 7)

    elif col == "calculated_host_listings_count":
        input_data[col] = st.slider("Host Listings Count", 1, MAX_LISTINGS, 1)

    elif col == "host_listings_ratio":
        input_data[col] = 0  # placeholder

    else:
        input_data[col] = st.slider(col, 0, 100, 0)

# --- Calculate host_listings_ratio automatically ---
if "calculated_host_listings_count" in input_data:
    input_data["host_listings_ratio"] = input_data["calculated_host_listings_count"] / MAX_LISTINGS

# -----------------
# Create DataFrame
# -----------------
user_df = pd.DataFrame([input_data])

# Ensure all required features exist (avoid KeyError)
for col in reg_features:
    if col not in user_df.columns:
        user_df[col] = 0
for col in clf_features:
    if col not in user_df.columns:
        user_df[col] = 0

# Debug: show user_df to verify inputs
st.write("🔍 Input DataFrame:", user_df)

# -----------------
# Predictions
# -----------------
if task == "🏷️ Category Prediction":
    if st.button("🔮 Predict Category"):
        pred_class = clf_pipeline.predict(user_df[clf_features])[0]
        st.success(f"Prediction: {'Expensive 💰' if pred_class == 1 else 'Affordable 🏡'}")

if task == "💰 Price Prediction":
    if st.button("💵 Predict Price"):
        pred_price = reg_pipeline.predict(user_df[reg_features])[0]
        st.success(f"Predicted Price: ${pred_price:,.2f}")
