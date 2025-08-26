import streamlit as st
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# ==============================
# Hugging Face Repo
# ==============================
HF_REPO = "UjjwalKaushik/Airbnb_model"

# ==============================
# Load pipelines
# ==============================
@st.cache_resource
def load_models():
    progress = st.progress(0, text="📥 Downloading pipeline files...")
    files = ["reg_pipeline.pkl.gz", "clf_pipeline.pkl.gz"]

    paths = []
    total = len(files)
    for i, fname in enumerate(files, start=1):
        with st.spinner(f"Downloading {fname}..."):
            path = hf_hub_download(repo_id=HF_REPO, filename=fname)
            paths.append(path)
        progress.progress(i / total, text=f"Downloaded {i}/{total} files")

    st.success("✅ Pipelines loaded!")
    reg_pipeline = joblib.load(paths[0])
    clf_pipeline = joblib.load(paths[1])
    return reg_pipeline, clf_pipeline


# ==============================
# Sidebar: Manual refresh option
# ==============================
st.sidebar.header("⚙️ Settings")
if st.sidebar.button("♻️ Clear Cache & Redownload"):
    st.cache_resource.clear()
    st.warning("Cache cleared! Reloading...")
    st.rerun()


# ==============================
# Main App
# ==============================
st.title("🏠 Airbnb Report App")

reg_pipeline, clf_pipeline = load_models()

# ==============================
# User Input Form
# ==============================
st.header("📊 Make Predictions")
st.write("Enter feature values:")

user_inputs = {}

col1, col2 = st.columns(2)

with col1:
    user_inputs["host_identity_verified"] = st.radio("Host identity verified?", ["Yes", "No"])
    user_inputs["neighbourhood_group"] = st.selectbox(
        "Neighbourhood Group", ["Brooklyn", "Manhattan", "Queens", "Bronx", "Staten Island"]
    )
    user_inputs["neighbourhood"] = st.text_input("Neighbourhood", "Williamsburg")
    user_inputs["country"] = st.selectbox("Country", ["US"])
    user_inputs["instant_bookable"] = st.radio("Instant Bookable?", ["True", "False"])
    user_inputs["cancellation_policy"] = st.selectbox("Cancellation Policy", ["flexible", "moderate", "strict"])
    user_inputs["room_type"] = st.selectbox(
        "Room Type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
    )
    user_inputs["house_rules"] = st.text_area("House Rules (optional)")
    user_inputs["license"] = st.text_input("License (optional)")

with col2:
    user_inputs["latitude"] = st.slider("Latitude", -90.0, 90.0, 40.7)
    user_inputs["longitude"] = st.slider("Longitude", -180.0, 180.0, -73.9)
    user_inputs["construction_year"] = st.number_input("Construction Year", 1800, 2025, 2000)
    user_inputs["price"] = st.slider("Price ($)", 0, 1000, 100)
    user_inputs["service_fee"] = st.slider("Service Fee ($)", 0, 500, 50)
    user_inputs["minimum_nights"] = st.slider("Minimum Nights", 1, 365, 2)
    user_inputs["number_of_reviews"] = st.number_input("Number of Reviews", 0, 10000, 10)
    user_inputs["reviews_per_month"] = st.number_input("Reviews per Month", 0.0, 100.0, 1.0)
    user_inputs["review_rate_number"] = st.slider("Review Rating", 0.0, 5.0, 4.5)
    user_inputs["availability_365"] = st.slider("Availability (days)", 0, 365, 180)
    user_inputs["calculated_host_listings_count"] = st.number_input("Host Listings Count", 0, 1000, 1)
    user_inputs["host_listings_ratio"] = st.number_input("Host Listings Ratio", 0.0, 100.0, 1.0)


# ==============================
# Predictions
# ==============================
if st.button("🔮 Predict"):
    input_df = pd.DataFrame([user_inputs])   # raw values go here

    # Use pipelines (handle encoding + scaling internally)
    reg_pred = reg_pipeline.predict(input_df)[0]
    clf_pred = clf_pipeline.predict(input_df)[0]

    clf_probs = getattr(clf_pipeline, "predict_proba", lambda x: None)(input_df)

    st.subheader("📌 Results")
    st.write(f"**Predicted Price (Regression):** ${reg_pred:.2f}")
    st.write(f"**Classification Output:** {clf_pred}")

    if clf_probs is not None:
        st.write("**Classification Probabilities:**")
        st.json({str(label): f"{prob:.2%}" for label, prob in zip(clf_pipeline.classes_, clf_probs[0])})
