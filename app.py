import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load Models and Features
# ---------------------------
@st.cache_resource
def load_models():
    reg_model = joblib.load("reg_model.pkl")
    clf_model = joblib.load("clf_model.pkl")
    reg_features = joblib.load("reg_features.pkl")
    clf_features = joblib.load("clf_features.pkl")
    return reg_model, clf_model, reg_features, clf_features


st.set_page_config(page_title="Airbnb Report App", page_icon="🏠", layout="wide")
st.title("🏠 Airbnb Report App")

reg_model, clf_model, reg_features, clf_features = load_models()
st.success("✅ Models are ready to use!")


# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("📌 Enter Airbnb Details")

room_type = st.sidebar.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
accommodates = st.sidebar.number_input("Accommodates", 1, 16, 2)
bathrooms = st.sidebar.number_input("Bathrooms", 0.5, 5.0, 1.0, step=0.5)
bedrooms = st.sidebar.number_input("Bedrooms", 0, 10, 1)

# ---------------------------
# Predict Button
# ---------------------------
if st.sidebar.button("🔮 Predict"):
    # Build dataframe from inputs
    input_data = pd.DataFrame([{
        "room_type": room_type,
        "accommodates": accommodates,
        "bathrooms": bathrooms,
        "bedrooms": bedrooms
    }])

    # ---------------------------
    # Regression Prediction
    # ---------------------------
    X_reg = input_data.reindex(columns=reg_features, fill_value=0)
    price_pred = reg_model.predict(X_reg)[0]

    # ---------------------------
    # Classification Prediction
    # ---------------------------
    X_clf = input_data.reindex(columns=clf_features, fill_value=0)
    demand_pred = clf_model.predict(X_clf)[0]
    demand_label = "🔥 High Demand" if demand_pred == 1 else "❄️ Low Demand"

    # ---------------------------
    # Show Results
    # ---------------------------
    st.subheader("📊 Prediction Results")
    st.metric("💰 Predicted Price", f"${price_pred:.2f}")
    st.metric("📈 Demand Prediction", demand_label)

    # ---------------------------
    # Visualization
    # ---------------------------
    st.subheader("📉 Feature Overview")
    st.bar_chart(input_data.T)

