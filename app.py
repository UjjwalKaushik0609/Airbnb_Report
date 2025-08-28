import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# -------------------------------
# Load Model Function
# -------------------------------
@st.cache_resource
def load_model(model_choice):
    filename = "best_random_forest.pkl" if model_choice == "Random Forest" else "best_xgboost.pkl"
    model_path = hf_hub_download(
        repo_id="UjjwalKaushik/Airbnb_model",
        filename=filename
    )
    return joblib.load(model_path)

# -------------------------------
# Load Dataset for dropdown values
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")  # keep this in repo
    return df

df = load_data()

# -------------------------------
# Sidebar - Model Selection
# -------------------------------
st.sidebar.header("🔧 Settings")
model_choice = st.sidebar.radio("Choose Model", ["Random Forest", "XGBoost"])
model = load_model(model_choice)

# -------------------------------
# Sidebar - User Inputs
# -------------------------------
st.sidebar.header("📌 Input Features")

def user_input():
    data = {}

    # Neighbourhood Group & Neighbourhood
    neighbourhood_group = st.sidebar.selectbox("Neighbourhood Group", df["neighbourhood_group"].unique())
    neighbourhoods = df[df["neighbourhood_group"] == neighbourhood_group]["neighbourhood"].unique()
    neighbourhood = st.sidebar.selectbox("Neighbourhood", sorted(neighbourhoods))

    # Numeric sliders
    data['calculated_host_listings_count'] = st.sidebar.slider("Calculated Host Listings Count", 0, 50, 1)
    data['availability_365'] = st.sidebar.slider("Availability (days per year)", 0, 365, 100)

    # Year dropdown
    years = sorted(df["last_review_year"].dropna().unique())
    data['last_review_year'] = st.sidebar.selectbox("Last Review Year", years)
    data['last_review_month'] = st.sidebar.slider("Last Review Month", 1, 12, 6)

    # Booleans
    data['host_identity_verified'] = st.sidebar.selectbox("Host Identity Verified", [True, False])
    data['instant_bookable'] = st.sidebar.selectbox("Instant Bookable", [True, False])

    # Store categorical
    data['neighbourhood_group'] = neighbourhood_group
    data['neighbourhood'] = neighbourhood

    return pd.DataFrame([data])

input_data = user_input()

# -------------------------------
# Prediction
# -------------------------------
st.subheader("🎯 Predicted Price")
try:
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Price: **${prediction:.2f}**")
except Exception as e:
    st.error(f"Error making prediction: {e}")

# -------------------------------
# Visualization
# -------------------------------
st.subheader("📊 Data Overview")
st.write("Sample of dataset:")
st.dataframe(df.head(10))
