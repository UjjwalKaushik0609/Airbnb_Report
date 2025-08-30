import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Airbnb Price Prediction", layout="wide")

st.title("🏡 Airbnb Price Prediction App")
st.write("Switch between ML models and enter property details to predict the nightly price.")

# -----------------------
# Model Selection
# -----------------------
model_choice = st.selectbox(
    "Choose Model", 
    ["Ridge Regression", "Random Forest", "XGBoost"]
)

# Load selected model + features
try:
    if model_choice == "Ridge Regression":
        model = joblib.load("ridge_regression_model.pkl")
        feature_names = joblib.load("ridge_regression_model_features.pkl")

    elif model_choice == "Random Forest":
        model = joblib.load("random_forest_model.pkl")
        feature_names = joblib.load("random_forest_model_features.pkl")

    elif model_choice == "XGBoost":
        model = joblib.load("xgboost_model.pkl")
        feature_names = joblib.load("xgboost_model_features.pkl")

    st.success(f"{model_choice} loaded successfully ✅")

except Exception:
    st.error("❌ Model or feature file not found. Please upload .pkl and *_features.pkl files.")
    st.stop()

# -----------------------
# Dynamic Input Form
# -----------------------
st.subheader("🔧 Enter Property Details")

input_dict = {}

for feat in feature_names:
    # numeric features
    if feat in ["bedrooms", "bathrooms", "accommodates", "minimum_nights", 
                "availability_365", "number_of_reviews", "review_scores_rating"]:
        input_dict[feat] = st.number_input(feat, min_value=0, max_value=100, value=1)

    # categorical features (OneHotEncoded)
    elif "room_type" in feat.lower():
        options = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
        choice = st.selectbox("Room Type", options)
        # one-hot encoding trick
        for opt in options:
            col = f"room_type_{opt}"
            input_dict[col] = 1 if col == feat and f"room_type_{choice}" == col else 0

    elif "neighbourhood" in feat.lower():
        # put some dummy options, better to load from listings.csv
        options = ["Downtown", "Uptown", "Suburbs", "Other"]
        choice = st.selectbox("Neighbourhood", options)
        for opt in options:
            col = f"neighbourhood_{opt}"
            input_dict[col] = 1 if col == feat and f"neighbourhood_{choice}" == col else 0

    # everything else default to 0
    else:
        input_dict[feat] = 0

# Convert dict to dataframe
input_data = pd.DataFrame([input_dict], columns=feature_names)

# -----------------------
# Prediction
# -----------------------
if st.button("💰 Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Price: **${prediction:,.2f}**")

