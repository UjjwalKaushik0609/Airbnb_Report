import streamlit as st
import pandas as pd
import joblib
import requests
import io
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Airbnb Price Prediction", layout="wide", page_icon="🏠", initial_sidebar_state="expanded")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/UjjwalKaushik0609/Airbnb_Report/refs/heads/main/cleaned_dataset.csv"
    return pd.read_csv(url)

df = load_data()

# -------------------------------
# Load Models from Hugging Face
# -------------------------------
@st.cache_resource
def load_model_from_hf(repo_id, filename):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    response = requests.get(url)
    response.raise_for_status()
    return joblib.load(io.BytesIO(response.content))

rf_model = load_model_from_hf("UjjwalKaushik/Airbnb_model", "best_random_forest.pkl")
xgb_model = load_model_from_hf("UjjwalKaushik/Airbnb_model", "best_xgboost.pkl")

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🏠 Airbnb Price Prediction Dashboard (Dark Theme)")
st.markdown("Predict prices with **Random Forest** and **XGBoost**, and explore dataset insights.")

st.sidebar.header("User Input Features")

# Example inputs (you can expand based on your dataset)
room_type = st.sidebar.selectbox("Room Type", df["room_type"].unique())
neighbourhood = st.sidebar.selectbox("Neighbourhood", df["neighbourhood"].unique())
minimum_nights = st.sidebar.slider("Minimum Nights", 1, 30, 1)
number_of_reviews = st.sidebar.slider("Number of Reviews", 0, 500, 10)

# Prepare input DataFrame
input_data = pd.DataFrame({
    "room_type": [room_type],
    "neighbourhood": [neighbourhood],
    "minimum_nights": [minimum_nights],
    "number_of_reviews": [number_of_reviews]
})

# Predict
if st.button("🔮 Predict Price"):
    rf_pred = rf_model.predict(input_data)[0]
    xgb_pred = xgb_model.predict(input_data)[0]

    st.success(f"**Random Forest Prediction:** ${rf_pred:.2f}")
    st.info(f"**XGBoost Prediction:** ${xgb_pred:.2f}")

# -------------------------------
# Data Visualizations
# -------------------------------
st.subheader("📊 Dataset Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("### Average Price by Room Type")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="room_type", y="price", ax=ax, palette="Blues_d")
    st.pyplot(fig)

with col2:
    st.write("### Average Price by Neighbourhood")
    fig, ax = plt.subplots(figsize=(6,4))
    top_neigh = df.groupby("neighbourhood")["price"].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=top_neigh.values, y=top_neigh.index, ax=ax, palette="Greens_d")
    st.pyplot(fig)
