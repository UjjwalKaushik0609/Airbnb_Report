import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Page Config (Dark Airbnb Theme)
# -------------------------------
st.set_page_config(page_title="Airbnb Analytics Dashboard", layout="wide")

# Airbnb style CSS (dark canvas background)
st.markdown("""
    <style>
        body, .stApp {background-color: #121212; color: #f5f5f5;}
        .stMetric {background-color: #1e1e1e; padding: 15px; border-radius: 10px;}
        .stButton>button {
            background-color: #FF5A5F; color: white; border-radius: 8px; padding: 0.5em 1em;
        }
        .stButton>button:hover {background-color: #e04852;}
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Models + Features
# -------------------------------
reg = joblib.load("reg_model.pkl")
clf = joblib.load("clf_model.pkl")
reg_features = joblib.load("reg_features.pkl")
clf_features = joblib.load("clf_features.pkl")

# -------------------------------
# App Title
# -------------------------------
st.title("🏡 Airbnb Data Explorer & Prediction App")

st.markdown("Explore Airbnb trends and predict **Price 💲** & **Demand 🔥** using trained ML models.")

# -------------------------------
# Sidebar User Input
# -------------------------------
st.sidebar.header("🔧 Input Features")

lat = st.sidebar.slider("Latitude", 40.5, 40.95, 40.75)
lon = st.sidebar.slider("Longitude", -74.25, -73.70, -73.95)
min_nights = st.sidebar.number_input("Minimum Nights", 1, 100, 3)
service_fee = st.sidebar.number_input("Service Fee ($)", 0, 200, 20)

room_type = st.sidebar.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
neigh_group = st.sidebar.selectbox("Neighbourhood Group", ["Brooklyn", "Manhattan", "Queens", "Bronx", "Staten Island"])

# Prepare input dataframe
input_dict = {
    "lat": lat,
    "long": lon,
    "minimum nights": min_nights,
    "service fee": service_fee,
    "room type_Private room": 1 if room_type == "Private room" else 0,
    "room type_Shared room": 1 if room_type == "Shared room" else 0,
    "neighbourhood group_Bronx": 1 if neigh_group == "Bronx" else 0,
    "neighbourhood group_Brooklyn": 1 if neigh_group == "Brooklyn" else 0,
    "neighbourhood group_Manhattan": 1 if neigh_group == "Manhattan" else 0,
    "neighbourhood group_Queens": 1 if neigh_group == "Queens" else 0,
    "neighbourhood group_Staten Island": 1 if neigh_group == "Staten Island" else 0,
}
input_df = pd.DataFrame([input_dict])

# Align with model features
input_reg = input_df.reindex(columns=reg_features, fill_value=0)
input_clf = input_df.reindex(columns=clf_features, fill_value=0)

# -------------------------------
# Predictions
# -------------------------------
st.subheader("🔮 Predictions")

col1, col2 = st.columns(2)

with col1:
    try:
        price_pred = np.expm1(reg.predict(input_reg))[0]
        st.metric("Predicted Price", f"${price_pred:,.2f}")
    except Exception as e:
        st.error(f"Price prediction error: {e}")

with col2:
    try:
        demand_pred = clf.predict(input_clf)[0]
        demand_label = "🔥 High Demand" if demand_pred == 1 else "❄️ Low Demand"
        st.metric("Predicted Demand", demand_label)
    except Exception as e:
        st.error(f"Demand prediction error: {e}")

# -------------------------------
# Example Data Visualizations
# -------------------------------
st.subheader("📊 Insights & Charts")

# Example: Room type distribution
sample_df = pd.DataFrame({
    "Room Type": ["Entire home/apt", "Private room", "Shared room"],
    "Counts": [3400, 2200, 400]
})
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(data=sample_df, x="Room Type", y="Counts", palette="magma", ax=ax)
ax.set_title("Distribution of Room Types", color="white")
ax.set_facecolor("#121212")
st.pyplot(fig)
st.caption("Entire homes dominate listings, followed by private rooms.")

# Example: Price distribution
prices = np.random.gamma(shape=2, scale=100, size=1000)  # mock data
fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(prices, bins=40, kde=True, color="skyblue", ax=ax)
ax.set_title("Distribution of Prices", color="white")
ax.set_facecolor("#121212")
st.pyplot(fig)
st.caption("Most listings are under $500, but a few outliers raise the average.")

# Example: Correlation heatmap
mock_df = pd.DataFrame({
    "lat": np.random.uniform(40.5, 40.95, 300),
    "long": np.random.uniform(-74.25, -73.70, 300),
    "price": np.random.gamma(2, 100, 300),
    "min_nights": np.random.randint(1, 30, 300)
})
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(mock_df.corr(), annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlations", color="white")
st.pyplot(fig)
st.caption("Heatmap shows relationships between numeric features like price, nights, and location.")
