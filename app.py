import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load dataset and models
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")   # use cleaned dataset
    # normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

@st.cache_resource
def load_model(choice):
    if choice == "Random Forest":
        return joblib.load("best_random_forest.pkl")
    else:
        return joblib.load("best_xgboost.pkl")

df = load_data()

# ----------------------------
# User Input
# ----------------------------
def user_input():
    st.sidebar.header("Input Features")

    # Choose Model
    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost"])

    # Year dropdown (if exists)
    if "year" in df.columns:
        year = st.sidebar.selectbox("Year", sorted(df["year"].unique()))
        df_filtered = df[df["year"] == year]
    else:
        year = None
        df_filtered = df

    # Neighbourhood group
    neighbourhood_group = st.sidebar.selectbox(
        "Neighbourhood Group", df_filtered["neighbourhood_group"].unique()
    )

    # Neighbourhood auto-filtered
    neighbourhood = st.sidebar.selectbox(
        "Neighbourhood",
        df_filtered[df_filtered["neighbourhood_group"] == neighbourhood_group]["neighbourhood"].unique()
    )

    # Room type
    room_type = st.sidebar.selectbox("Room Type", df_filtered["room_type"].unique())

    # Host identity verified
    host_identity_verified = st.sidebar.selectbox("Host Identity Verified", ["t", "f"])

    # Instant bookable
    instant_bookable = st.sidebar.selectbox("Instant Bookable", ["t", "f"])

    # Numerical sliders
    minimum_nights = st.sidebar.slider(
        "Minimum Nights", 1, int(df["minimum_nights"].max()), int(df["minimum_nights"].mean())
    )

    number_of_reviews = st.sidebar.slider(
        "Number of Reviews", 0, int(df["number_of_reviews"].max()), int(df["number_of_reviews"].mean())
    )

    reviews_per_month = st.sidebar.slider(
        "Reviews per Month", 0.0, float(df["reviews_per_month"].max()), float(df["reviews_per_month"].mean())
    )

    availability_365 = st.sidebar.slider(
        "Availability (days)", 0, 365, int(df["availability_365"].mean())
    )

    # Collect inputs
    data = {
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "room_type": room_type,
        "host_identity_verified": host_identity_verified,
        "instant_bookable": instant_bookable,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "availability_365": availability_365,
    }

    return pd.DataFrame([data]), model_choice

# ----------------------------
# Main App
# ----------------------------
def main():
    st.title("Airbnb Price Prediction App 🏠")

    input_df, model_choice = user_input()
    model = load_model(model_choice)

    st.subheader("Your Input Features")
    st.write(input_df)

    # Predict price
    prediction = model.predict(input_df)[0]
    st.subheader("🎯 Predicted Price")
    st.success(f"${prediction:,.2f} (using {model_choice})")

if __name__ == "__main__":
    main()
