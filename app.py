import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

# Hugging Face repo
REPO_ID = "UjjwalKaushik/Airbnb_model"

@st.cache_resource
def load_models():
    try:
        # Download files from Hugging Face
        clf_model_path = hf_hub_download(repo_id=REPO_ID, filename="clf_model.pkl")
        clf_pipeline_path = hf_hub_download(repo_id=REPO_ID, filename="clf_pipeline.pkl")
        clf_features_path = hf_hub_download(repo_id=REPO_ID, filename="clf_features.pkl")
        reg_features_path = hf_hub_download(repo_id=REPO_ID, filename="reg_features.pkl")

        # Load with joblib
        clf_model = joblib.load(clf_model_path)
        clf_pipeline = joblib.load(clf_pipeline_path)
        clf_features = joblib.load(clf_features_path)
        reg_features = joblib.load(reg_features_path)

        return clf_model, clf_pipeline, clf_features, reg_features
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, None, None, None


# Load models
clf_model, clf_pipeline, clf_features, reg_features = load_models()

st.title("🏡 Airbnb Price & Category Predictor")

task = st.radio("Choose Task", ["💰 Price Prediction", "🏷 Category Prediction"])

if task == "💰 Price Prediction" and clf_pipeline and reg_features is not None:
    st.subheader("Enter Airbnb details for Price Prediction")
    inputs = {}
    for feat in reg_features:
        inputs[feat] = st.text_input(f"{feat}")

    if st.button("Predict Price"):
        try:
            X = [list(inputs.values())]
            price = clf_pipeline.predict(X)[0]
            st.success(f"💰 Predicted Price: ${price:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif task == "🏷 Category Prediction" and clf_model and clf_features is not None:
    st.subheader("Enter Airbnb details for Category Prediction")
    inputs = {}
    for feat in clf_features:
        inputs[feat] = st.text_input(f"{feat}")

    if st.button("Predict Category"):
        try:
            X = [list(inputs.values())]
            category = clf_model.predict(X)[0]
            st.success(f"🏷 Predicted Category: {category}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

