import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# Load dataset & model
# ----------------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_dataset.csv")

@st.cache_resource
def load_model():
    return joblib.load("airbnb_model.pkl")

df = load_data()
model = load_model()

# ----------------------
# User Input Function
# ----------------------
def user_input():
    data = {}

    # --- Neighbourhood Group & Neighbourhood ---
    neighbourhood_group = st.sidebar.selectbox("Neighbourhood Group", df["neighbourhood group"].unique())
    neighbourhood_options = df[df["neighbourhood group"] == neighbourhood_group]["neighbourhood"].unique()
    neighbourhood = st.sidebar.selectbox("Neighbourhood", neighbourhood_options)

    data["neighbourhood group"] = neighbourhood_group
    data["neighbourhood"] = neighbourhood

    # --- Other categorical features ---
    data["country"] = st.sidebar.selectbox("Country", df["country"].unique())
    data["country code"] = st.sidebar.selectbox("Country Code", df["country code"].unique())
    data["cancellation_policy"] = st.sidebar.selectbox("Cancellation Policy", df["cancellation_policy"].unique())
    data["room type"] = st.sidebar.selectbox("Room Type", df["room type"].unique())

    # --- Boolean categorical (True/False) ---
    data["host_identity_verified"] = st.sidebar.selectbox("Host Identity Verified", [True, False])
    data["instant_bookable"] = st.sidebar.selectbox("Instant Bookable", [True, False])

    # --- Numerical features ---
    for col in ["lat", "long", "Construction year", "service fee", "minimum nights",
                "number of reviews", "reviews per month", "review rate number",
                "calculated host listings count", "availability 365"]:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        default_val = float(df[col].mean())
        data[col] = st.sidebar.slider(f"{col}", min_val, max_val, default_val)

    # --- Special cases ---
    # Year dropdown
    year_min = int(df["last_review_year"].min())
    year_max = int(df["last_review_year"].max())
    data["last_review_year"] = st.sidebar.selectbox("Last Review Year", list(range(year_min, year_max + 1)))

    # Month dropdown (1–12)
    data["last_review_month"] = st.sidebar.selectbox("Last Review Month", list(range(1, 13)))

    return pd.DataFrame([data])


# ----------------------
# Main App
# ----------------------
st.title("🏡 Airbnb Price Prediction")
st.write("Predict **Total Cost** of Airbnb listings based on input features.")

# Collect user input
input_data = user_input()

# Make prediction
try:
    prediction = model.predict(input_data)[0]
    st.subheader("💰 Predicted Total Cost")
    st.success(f"${prediction:,.2f}")
except Exception as e:
    st.error(f"Error making prediction: {e}")

# ----------------------
# Data Visualization
# ----------------------
st.subheader("📊 Data Insights")

# Price Distribution
fig, ax = plt.subplots()
sns.histplot(df["total_cost"], bins=50, kde=True, ax=ax)
ax.set_title("Distribution of Total Cost")
st.pyplot(fig)

# Avg price per neighbourhood group
fig, ax = plt.subplots()
df.groupby("neighbourhood group")["total_cost"].mean().plot(kind="bar", ax=ax)
ax.set_ylabel("Average Total Cost")
ax.set_title("Average Total Cost by Neighbourhood Group")
st.pyplot(fig)

# Scatter: price vs. reviews
fig, ax = plt.subplots()
sns.scatterplot(x="number of reviews", y="total_cost", data=df, alpha=0.5, ax=ax)
ax.set_title("Reviews vs. Total Cost")
st.pyplot(fig)
