import streamlit as st
import pandas as pd
import numpy as np
import joblib, gzip, os, requests, matplotlib.pyplot as plt, seaborn as sns

def gdrive_download(file_id, dest_path):
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        r = requests.get(url)
        with open(dest_path, "wb") as f:
            f.write(r.content)
    return dest_path

REG_MODEL_ID = "1ENewUWOs_tiz4hZt8WUnRlcU6-naoTgu"
CLF_MODEL_ID = "1Bh7Ig0m8ZFg4gi2zdVGJ7JtZIkkxHHRp"
REG_FEAT_ID  = "1ajaBfSUiorYptqjjC7kKKrVFzYFHOeU-"
CLF_FEAT_ID  = "1IwHCTMGUtmjRNI-QHd-zU9Lr7r59cWOc"

reg_model_path = gdrive_download(REG_MODEL_ID, "reg_model.pkl.gz")
clf_model_path = gdrive_download(CLF_MODEL_ID, "clf_model.pkl.gz")
reg_features_path = gdrive_download(REG_FEAT_ID, "reg_features.pkl")
clf_features_path = gdrive_download(CLF_FEAT_ID, "clf_features.pkl")

with gzip.open(reg_model_path, "rb") as f:
    reg = joblib.load(f)
with gzip.open(clf_model_path, "rb") as f:
    clf = joblib.load(f)
X_reg_cols = joblib.load(reg_features_path)
X_clf_cols = joblib.load(clf_features_path)

st.set_page_config(page_title="Airbnb Report", layout="wide")

st.title("🏡 Airbnb Price & Demand Prediction")
st.markdown("Interactive dashboard with **dark Airbnb-style theme**.")

uploaded_file = st.file_uploader("Upload Airbnb dataset (CSV)", type=["csv", "gz"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    st.success(f"Data loaded! Shape: {df.shape}")

    st.subheader("📊 Price Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df['price'], bins=50, kde=True, ax=ax, color="orange")
    ax.set_title("Distribution of Prices")
    st.pyplot(fig)

    st.subheader("🏘️ Room Types")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="room type", ax=ax, palette="viridis")
    st.pyplot(fig)

    st.subheader("🔮 Predictions")
    st.markdown("### Price Prediction")
    input_vals = st.text_area("Enter numeric features (comma separated)", "")
    if input_vals:
        try:
            vals = list(map(float, input_vals.split(",")))
            input_df = pd.DataFrame([vals], columns=X_reg_cols[:len(vals)])
            pred_price = np.expm1(reg.predict(input_df)[0])
            st.success(f"💰 Predicted Price: ${pred_price:,.2f}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("### Demand Prediction")
    input_vals2 = st.text_area("Enter demand features (comma separated)", "")
    if input_vals2:
        try:
            vals = list(map(float, input_vals2.split(",")))
            input_df = pd.DataFrame([vals], columns=X_clf_cols[:len(vals)])
            demand_pred = clf.predict(input_df)[0]
            label = "🔥 High Demand" if demand_pred == 1 else "📉 Low Demand"
            st.success(f"Prediction: {label}")
        except Exception as e:
            st.error(f"Error: {e}")
