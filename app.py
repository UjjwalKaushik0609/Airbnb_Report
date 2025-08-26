import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

# ==============================
# Hugging Face Repo
# ==============================
HF_REPO = "UjjwalKaushik/Airbnb_model"

# ==============================
# Load models (cached in Streamlit)
# ==============================
@st.cache_resource
def load_models():
    # Download and load each file
    reg_model = joblib.load(
        hf_hub_download(HF_REPO, "reg_model.pkl.gz")
    )
    clf_model = joblib.load(
        hf_hub_download(HF_REPO, "clf_model.pkl.gz")
    )
    reg_features = joblib.load(
        hf_hub_download(HF_REPO, "reg_features.pkl")
    )
    clf_features = joblib.load(
        hf_hub_download(HF_REPO, "clf_features.pkl")
    )

    return reg_model, clf_model, reg_features, clf_features


# ==============================
# Sidebar: Manual refresh option
# ==============================
st.sidebar.header("⚙️ Settings")

if st.sidebar.button("♻️ Clear Cache & Redownload"):
    st.cache_resource.clear()
    st.warning("Cache cleared! Reloading models...")
    st.rerun()


# ==============================
# Main App
# ==============================
st.title("🏠 Airbnb Report App")

reg_model, clf_model, reg_features, clf_features = load_models()

st.success("✅ Models are ready to use!")
