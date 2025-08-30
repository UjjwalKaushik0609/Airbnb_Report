import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostRegressor

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    reviews = pd.read_csv("data/reviews.zip", compression="zip")
    calendar = pd.read_csv("data/calendar.zip", compression="zip")
    listings = pd.read_csv("data/listings.csv")
    return reviews, calendar, listings

reviewsDF, calendarDF, listingsDF = load_data()

# ------------------------------
# Load Models
# ------------------------------
linreg_model = joblib.load("models/linear_regression_model.pkl")
ridge_model = joblib.load("models/ridge_regression_model.pkl")
lasso_model = joblib.load("models/lasso_regression_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
xgb_model = joblib.load("models/xgboost_model.pkl")

catb_model = CatBoostRegressor()
catb_model.load_model("models/catboost_model.cbm")  # adjust path

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Airbnb Price Prediction App")

bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=1)
gym = st.checkbox("Gym available")
room_type = st.selectbox("Room Type", ["Entire_home_apt", "Private_room", "Shared_room"])

# Example for more features
hot_tub_sauna_pool = st.checkbox("Hot tub / Sauna / Pool")
air_conditioning = st.checkbox("Air Conditioning")

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Price"):
    input_features = pd.DataFrame({
        "bedrooms": [bedrooms],
        "gym": [1 if gym else 0],
        "room_type_Entire_home_apt": [1 if room_type=="Entire_home_apt" else 0],
        "room_type_Private_room": [1 if room_type=="Private_room" else 0],
        "room_type_Shared_room": [1 if room_type=="Shared_room" else 0],
        "hot_tub_sauna_or_pool": [1 if hot_tub_sauna_pool else 0],
        "air_conditioning": [1 if air_conditioning else 0],
    })
    
    # Choose model to predict
    model_choice = st.selectbox("Select Model", ["Linear Regression", "Ridge", "Lasso", "Random Forest", "XGBoost", "CatBoost"])
    
    if model_choice == "Linear Regression":
        prediction = linreg_model.predict(input_features)
    elif model_choice == "Ridge":
        prediction = ridge_model.predict(input_features)
    elif model_choice == "Lasso":
        prediction = lasso_model.predict(input_features)
    elif model_choice == "Random Forest":
        prediction = rf_model.predict(input_features)
    elif model_choice == "XGBoost":
        prediction = xgb_model.predict(input_features)
    else:
        prediction = catb_model.predict(input_features)
    
    st.success(f"Predicted Price: ${prediction[0]:.2f}")

