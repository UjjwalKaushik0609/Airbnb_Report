import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ---------------------------
# Load cleaned dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")   # keep in repo or also move to HF
    return df

# ---------------------------
# Load model from Hugging Face
# ---------------------------
@st.cache_resource
def load_model(choice):
    repo_id = "your-username/your-repo"  # 🔹 replace with your HF repo name

    if choice == "Random Forest":
        model_path = hf_hub_download(repo_id=repo_id, filename="best_random_forest.pkl")
    else:
        model_path = hf_hub_download(repo_id=repo_id, filename="best_xgboost.pkl")

    return joblib.load(model_path)

# ---------------------------
# Sidebar Inputs
# ---------------------------
def user_input(df):
    st.sidebar.header("Input Features")

    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost"])

    neighbourhood_group = st.sidebar.selectbox("Neighbourhood Group", df["neighbourhood_group"].unique())
    neighbourhoods = df[df["neighbourhood_group"] == neighbourhood_group]["neighbourhood"].unique()
    neighbourhood = st.sidebar.selectbox("Neighbourhood", neighbourhoods)

    room_type = st.sidebar.selectbox("Room Type", df["room_type"].unique())
    minimum_nights = st.sidebar.number_input("Minimum Nights", 1, 365, 3)
    number_of_reviews = st.sidebar.slider("Number of Reviews", 0, 500, 50)
    reviews_per_month = st.sidebar.number_input("Reviews per Month", 0.0, 30.0, 1.0)
    calculated_host_listings_count = st.sidebar.number_input("Host Listings Count", 1, 100, 1)
    availability_365 = st.sidebar.slider("Availability (days/year)", 0, 365, 180)

    input_data = pd.DataFrame({
        "neighbourhood_group": [neighbourhood_group],
        "neighbourhood": [neighbourhood],
        "room_type": [room_type],
        "minimum_nights": [minimum_nights],
        "number_of_reviews": [number_of_reviews],
        "reviews_per_month": [reviews_per_month],
        "calculated_host_listings_count": [calculated_host_listings_count],
        "availability_365": [availability_365]
    })

    return input_data, model_choice

# ---------------------------
# Main App
# ---------------------------
def main():
    st.title("🏠 Airbnb Price Prediction App")

    df = load_data()
    input_df, model_choice = user_input(df)
    model = load_model(model_choice)

    st.subheader("Your Input")
    st.write(input_df)

    prediction = model.predict(input_df)
    st.subheader("Predicted Price 💵")
    st.success(f"${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()
