import streamlit as st
import joblib
import pandas as pd

# -------------------------
# Load Models + Features
# -------------------------
reg_pipeline = joblib.load("reg_pipeline.pkl")
clf_pipeline = joblib.load("clf_pipeline.pkl")
reg_features = joblib.load("reg_features.pkl")
clf_features = joblib.load("clf_features.pkl")

st.title("🏡 Airbnb Price & Demand Prediction App")

st.markdown("Enter Airbnb listing details below to predict **price** and **demand**:")

# -------------------------
# Auto-generate Input Form
# -------------------------
def generate_inputs(feature_list):
    inputs = {}
    for col in feature_list:
        if col == "demand":  # target, skip
            continue
        # Guess input type
        if "id" in col.lower() or "number" in col.lower() or "count" in col.lower():
            inputs[col] = st.number_input(col, min_value=0, value=1)
        elif "ratio" in col.lower() or "price" in col.lower() or "score" in col.lower():
            inputs[col] = st.number_input(col, value=0.0)
        elif col in ["latitude", "longitude"]:
            inputs[col] = st.number_input(col, value=0.0, format="%.6f")
        elif col.isnumeric():
            inputs[col] = st.number_input(col, value=0.0)
        else:
            inputs[col] = st.text_input(col, "")
    return pd.DataFrame([inputs])

# Build input frame using union of regression + classification features
all_features = sorted(set(reg_features) | set(clf_features))
input_df = generate_inputs(all_features)

st.subheader("🔎 Input Preview")
st.write(input_df)

# -------------------------
# Predictions
# -------------------------
if st.button("Predict"):
    # Regression
    X_reg = input_df.reindex(columns=reg_features, fill_value=0)
    price_pred = reg_pipeline.predict(X_reg)[0]

    # Classification
    X_clf = input_df.reindex(columns=clf_features, fill_value=0)
    demand_pred = clf_pipeline.predict(X_clf)[0]
    demand_prob = clf_pipeline.predict_proba(X_clf)[0][1]

    st.success(f"💰 Predicted Price: **${price_pred:,.2f}**")
    st.info(f"📈 Predicted Demand: **{'High' if demand_pred == 1 else 'Low'}** (Probability: {demand_prob:.2%})")
