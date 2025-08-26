import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

# ---------------------------
# Load Models from Hugging Face
# ---------------------------
@st.cache_resource
def load_models():
    repo_id = "UjjwalKaushik/Airbnb_model"

    # Regression model
    reg_model_path = hf_hub_download(repo_id=repo_id, filename="reg_model.pkl.gz")
    reg_features_path = hf_hub_download(repo_id=repo_id, filename="reg_features.pkl")

    # Classification model
    clf_model_path = hf_hub_download(repo_id=repo_id, filename="clf_model.pkl.gz")
    clf_features_path = hf_hub_download(repo_id=repo_id, filename="clf_features.pkl")

    reg_pipeline = joblib.load(reg_model_path)
    clf_pipeline = joblib.load(clf_model_path)

    reg_features = joblib.load(reg_features_path)
    clf_features = joblib.load(clf_features_path)

    return reg_pipeline, clf_pipeline, reg_features, clf_features


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🏡 Airbnb Price & Category Prediction")
st.write("Upload Airbnb details and predict price (regression) or category (classification).")

reg_pipeline, clf_pipeline, reg_features, clf_features = load_models()

# User Input
st.sidebar.header("Enter Airbnb Listing Details")
room_type = st.sidebar.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
neighbourhood = st.sidebar.text_input("Neighbourhood", "Brooklyn")
minimum_nights = st.sidebar.number_input("Minimum Nights", min_value=1, value=3)
reviews = st.sidebar.number_input("Number of Reviews", min_value=0, value=10)

# Convert to feature dict
input_data = {
    "room_type": room_type,
    "neighbourhood": neighbourhood,
    "minimum_nights": minimum_nights,
    "number_of_reviews": reviews,
}

# ---------------------------
# Prediction Buttons
# ---------------------------
if st.button("🔮 Predict Price"):
    try:
        price = reg_pipeline.predict([input_data])[0]
        st.success(f"💰 Estimated Price: ${price:.2f}")
    except Exception as e:
        st.error(f"Error in Regression Prediction: {e}")

if st.button("🏷️ Predict Category"):
    try:
        category = clf_pipeline.predict([input_data])[0]
        st.success(f"📌 Predicted Category: {category}")
    except Exception as e:
        st.error(f"Error in Classification Prediction: {e}")
