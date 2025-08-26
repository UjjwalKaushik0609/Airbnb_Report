import streamlit as st
import joblib
import numpy as np
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
    progress = st.progress(0, text="📥 Downloading model files...")
    files = [
        "reg_model.pkl.gz",
        "clf_model.pkl.gz",
        "reg_features.pkl",
        "clf_features.pkl",
    ]

    paths = []
    total = len(files)
    for i, fname in enumerate(files, start=1):
        with st.spinner(f"Downloading {fname}..."):
            path = hf_hub_download(repo_id=HF_REPO, filename=fname)
            paths.append(path)
        progress.progress(i / total, text=f"Downloaded {i}/{total} files")

    st.success("✅ All files downloaded and cached!")

    reg_model = joblib.load(paths[0])
    clf_model = joblib.load(paths[1])
    reg_features = joblib.load(paths[2])
    clf_features = joblib.load(paths[3])

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


# ==============================
# User Input Form
# ==============================
st.header("📊 Make Predictions")

st.write("Enter feature values to test the models:")

# Create inputs dynamically based on features
user_inputs = {}
cols = st.columns(2)  # split inputs into 2 columns
for i, feature in enumerate(reg_features):
    with cols[i % 2]:
        user_inputs[feature] = st.number_input(
            f"{feature}", value=0.0, step=1.0, format="%.2f"
        )

# Convert to numpy array
input_array = np.array([list(user_inputs.values())])

# ==============================
# Predictions
# ==============================
if st.button("🔮 Predict"):
    # Regression prediction
    reg_pred = reg_model.predict(input_array)[0]

    # Classification prediction
    clf_pred = clf_model.predict(input_array)[0]

    # Classification probabilities
    try:
        clf_probs = clf_model.predict_proba(input_array)[0]
    except Exception:
        clf_probs = None

    st.subheader("📌 Results")
    st.write(f"**Regression output:** {reg_pred:.2f}")
    st.write(f"**Classification output:** {clf_pred}")

    if clf_probs is not None:
        st.write("**Classification probabilities:**")
        prob_table = {
            str(label): f"{prob:.2%}"
            for label, prob in zip(clf_model.classes_, clf_probs)
        }
        st.json(prob_table)
