import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(page_title="Airbnb Data Explorer", layout="wide")

# -----------------------
# Load Data
# -----------------------
@st.cache_data
def load_data():
    reviews = pd.read_csv("reviews.zip", compression="zip")
    calendar = pd.read_csv("calendar.zip", compression="zip")
    listings = pd.read_csv("listings.csv")
    return reviews, calendar, listings

reviewsDF, calendarDF, listingsDF = load_data()

st.title("🏡 Airbnb Data Explorer")
st.success("Datasets Loaded Successfully ✅")

# -----------------------
# Dataset Overview
# -----------------------
st.subheader("Dataset Shapes")
st.write(f"**Reviews:** {reviewsDF.shape}")
st.write(f"**Calendar:** {calendarDF.shape}")
st.write(f"**Listings:** {listingsDF.shape}")

# -----------------------
# Data Preview
# -----------------------
st.subheader("📊 Data Samples")
tab1, tab2, tab3 = st.tabs(["Reviews", "Calendar", "Listings"])
with tab1:
    st.dataframe(reviewsDF.head())
with tab2:
    st.dataframe(calendarDF.head())
with tab3:
    st.dataframe(listingsDF.head())

# -----------------------
# EDA Section
# -----------------------
st.subheader("📈 Exploratory Data Analysis")

# Price distribution
if "price" in listingsDF.columns:
    st.markdown("### Price Distribution")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(listingsDF['price'].dropna().apply(lambda x: float(str(x).replace("$","").replace(",",""))),
                 bins=50, kde=True, ax=ax)
    ax.set_xlabel("Price ($)")
    st.pyplot(fig)

# Room type counts
if "room_type" in listingsDF.columns:
    st.markdown("### Room Types")
    st.bar_chart(listingsDF["room_type"].value_counts())

# Reviews per month trend
if "date" in calendarDF.columns:
    st.markdown("### Availability Over Time")
    calendarDF["date"] = pd.to_datetime(calendarDF["date"])
    trend = calendarDF.groupby(calendarDF["date"].dt.to_period("M")).size()
    st.line_chart(trend)

# -----------------------
# ML Model Loading
# -----------------------
st.subheader("🤖 Model Comparison")

def safe_load(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except:
        return None

models = {
    "Ridge Regression": safe_load("ridge_regression_model.pkl"),
    "Lasso Regression": safe_load("lasso_regression_model.pkl"),
    "Linear Regression": safe_load("linear_regression_model.pkl"),
    "Random Forest": safe_load("random_forest_model.pkl"),
    "XGBoost": safe_load("xgboost_model.pkl")
}

loaded_models = {name: m for name, m in models.items() if m is not None}

if loaded_models:
    st.success(f"Loaded Models: {', '.join(loaded_models.keys())}")
else:
    st.error("❌ No models loaded. Upload `.pkl` files.")

# -----------------------
# Prediction Form
# -----------------------
st.subheader("💡 Price Prediction")

if loaded_models:
    # let user choose which model
    model_choice = st.selectbox("Choose a model", list(loaded_models.keys()))
    model = loaded_models[model_choice]

    bedrooms = st.slider("Bedrooms", 0, 10, 2)
    bathrooms = st.slider("Bathrooms", 0, 5, 1)
    accommodates = st.slider("Accommodates", 1, 16, 4)

    input_data = pd.DataFrame([[bedrooms, bathrooms, accommodates]],
                              columns=["bedrooms", "bathrooms", "accommodates"])

    if st.button("Predict Price"):
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Price ({model_choice}): ${prediction:,.2f}")
else:
    st.warning("Prediction model not available. Upload your trained `.pkl` files.")

