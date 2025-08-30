import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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
# ML Model Comparison
# -----------------------
st.subheader("🤖 Model Comparison")

try:
    ridge = joblib.load("ridge_regression_model.pkl")
    rf = joblib.load("random_forest_model.pkl")
    xgb = joblib.load("xgb_model.pkl")

    st.write("Models loaded successfully ✅")
    st.write("- Ridge Regression")
    st.write("- Random Forest")
    st.write("- XGBoost")

except Exception as e:
    st.error("Could not load models. Make sure .pkl files are uploaded.")

# -----------------------
# Prediction Form
# -----------------------
st.subheader("💡 Price Prediction")

try:
    model = joblib.load("ridge_regression_model.pkl")  # default model
    st.write("Using Ridge Regression Model for prediction")

    # Example input features (expand later as per dataset)
    bedrooms = st.slider("Bedrooms", 0, 10, 2)
    bathrooms = st.slider("Bathrooms", 0, 5, 1)
    accommodates = st.slider("Accommodates", 1, 16, 4)

    input_data = pd.DataFrame([[bedrooms, bathrooms, accommodates]],
                              columns=["bedrooms", "bathrooms", "accommodates"])

    if st.button("Predict Price"):
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Price: ${prediction:,.2f}")

except Exception as e:
    st.warning("Prediction model not available. Upload your trained .pkl files.")
