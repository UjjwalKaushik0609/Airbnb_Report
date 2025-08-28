import streamlit as st
import pandas as pd
import joblib
import requests
import io
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Load Dataset (for dropdowns + visualization)
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_dataset.csv")

df = load_data()

# ----------------------------
# Load Model from Hugging Face
# ----------------------------
@st.cache_resource
def load_model():
    url = "https://huggingface.co/UjjwalKaushik/Airbnb_model/resolve/main/best_random_forest.pkl"
    r = requests.get(url)
    r.raise_for_status()
    return joblib.load(io.BytesIO(r.content))

model = load_model()

# ----------------------------
# App Layout
# ----------------------------
st.title("🏠 Airbnb Price Prediction App")
st.write("Predict **Total Price** of a listing based on input features.")

# ----------------------------
# User Input Section
# ----------------------------
st.sidebar.header("Enter Listing Details")

def user_input():
    data = {}
    
    # Dropdowns for categorical features
    data['neighbourhood group'] = st.sidebar.selectbox("Neighbourhood Group", df['neighbourhood group'].unique())
    data['neighbourhood'] = st.sidebar.selectbox("Neighbourhood", df['neighbourhood'].unique())
    data['country'] = st.sidebar.selectbox("Country", df['country'].unique())
    data['country code'] = st.sidebar.selectbox("Country Code", df['country code'].unique())
    data['cancellation_policy'] = st.sidebar.selectbox("Cancellation Policy", df['cancellation_policy'].unique())
    data['room type'] = st.sidebar.selectbox("Room Type", df['room type'].unique())

    # Sliders for numerical features
    for col in ['lat','long','Construction year','service fee','minimum nights',
                'number of reviews','reviews per month','review rate number',
                'calculated host listings count','availability 365',
                'last_review_year','last_review_month']:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        default_val = float(df[col].median())
        data[col] = st.sidebar.slider(col, min_val, max_val, default_val)

    # Binary categorical encoded as 0/1
    data['host_identity_verified'] = st.sidebar.selectbox("Host Identity Verified", ["t", "f"])
    data['instant_bookable'] = st.sidebar.selectbox("Instant Bookable", ["t", "f"])

    # Convert to numeric
    data['host_identity_verified'] = 1 if data['host_identity_verified'] == "t" else 0
    data['instant_bookable'] = 1 if data['instant_bookable'] == "t" else 0
    
    return pd.DataFrame([data])

input_data = user_input()

# ----------------------------
# Prediction
# ----------------------------
try:
    prediction = model.predict(input_data)[0]
    st.subheader("💰 Predicted Total Price")
    st.success(f"${prediction:,.2f}")
except Exception as e:
    st.error(f"Error making prediction: {e}")

# ----------------------------
# Visualization
# ----------------------------
st.subheader("📊 Price Distribution by Room Type")
selected_room = input_data['room type'].values[0]

fig, ax = plt.subplots()
sns.set_style("darkgrid")   # better for dark mode
sns.histplot(df[df['room type'] == selected_room]['total_cost'], bins=30, kde=True, ax=ax, color="#00C9A7")
ax.set_title(f"Distribution of Total Price for {selected_room}", color="white")
ax.set_xlabel("Total Price", color="white")
ax.set_ylabel("Frequency", color="white")
ax.tick_params(colors="white")
st.pyplot(fig)
