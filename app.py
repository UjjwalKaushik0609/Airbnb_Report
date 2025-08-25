import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, classification_report

# Airbnb brand colors
AIRBNB_RED = "#FF5A5F"
AIRBNB_TEAL = "#00A699"
AIRBNB_ORANGE = "#FC642D"

sns.set_style("darkgrid")

st.set_page_config(page_title="Airbnb Analysis & Prediction", layout="wide")

# ----------------------
# Load Data
# ----------------------
@st.cache_data
def load_data():
    return pd.read_csv("Airbnb_Cleaned_Ready.csv.gz", low_memory=False)

df = load_data()

# ----------------------
# Train Models (cached)
# ----------------------
@st.cache_resource
def train_price_model(df):
    df_ml = df.copy()
    cat_cols = [c for c in ['room type','neighbourhood group','instant_bookable',
                            'cancellation_policy','host_identity_verified'] if c in df_ml.columns]
    df_ml = pd.get_dummies(df_ml, columns=cat_cols, drop_first=True)
    drop_cols = [c for c in ['id','NAME','host name','country','country code','neighbourhood','last review'] if c in df_ml.columns]
    df_ml = df_ml.drop(columns=drop_cols, errors='ignore')
    leakage_cols = [c for c in df_ml.columns if 'price' in c or 'review' in c or 'avail' in c]
    X = df_ml.drop(columns=leakage_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    y = np.log1p(df['price'])
    reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg.fit(X, y)
    return reg, X

@st.cache_resource
def train_demand_model(df):
    df_ml = df.copy()
    cat_cols = [c for c in ['room type','neighbourhood group','instant_bookable',
                            'cancellation_policy','host_identity_verified'] if c in df_ml.columns]
    df_ml = pd.get_dummies(df_ml, columns=cat_cols, drop_first=True)
    drop_cols = [c for c in ['id','NAME','host name','country','country code','neighbourhood','last review'] if c in df_ml.columns]
    df_ml = df_ml.drop(columns=drop_cols, errors='ignore')
    leakage_cols = [c for c in df_ml.columns if 'price' in c or 'review' in c or 'avail' in c or 'demand' in c]
    X = df_ml.drop(columns=leakage_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    y = df['demand']
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
    clf.fit(X, y)
    return clf, X

reg, X_price = train_price_model(df)
clf, X_demand = train_demand_model(df)

# ----------------------
# Sidebar Navigation
# ----------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to:", ["Exploratory Data Analysis", "Price Prediction", "Demand Classification", "Try it Yourself"])

# ----------------------
# EDA
# ----------------------
if page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    
    if 'price' in df.columns:
        st.subheader("Distribution of Listing Prices")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.histplot(df['price'], bins=50, kde=True, color=AIRBNB_RED, ax=ax)
        st.pyplot(fig)
        st.caption("Most listings are low-priced, but a few very expensive outliers exist.")
    
    if 'room type' in df.columns:
        st.subheader("Room Type Distribution")
        fig, ax = plt.subplots(figsize=(8,5))
        sns.countplot(data=df, x="room type", hue=None, palette=[AIRBNB_RED, AIRBNB_TEAL, AIRBNB_ORANGE], legend=False, ax=ax)
        st.pyplot(fig)
        st.caption("Entire homes dominate, followed by private rooms.")
    
    if 'neighbourhood group' in df.columns:
        st.subheader("Neighbourhood Group Distribution")
        order = df['neighbourhood group'].value_counts().index
        fig, ax = plt.subplots(figsize=(8,5))
        sns.countplot(data=df, y="neighbourhood group", hue=None, order=order, palette=[AIRBNB_TEAL, AIRBNB_RED, AIRBNB_ORANGE], legend=False, ax=ax)
        st.pyplot(fig)
        st.caption("Some neighbourhoods have a higher concentration of listings.")
    
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12,8))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)
    st.caption("Shows how numeric features are related (e.g., service fee and price).")

# ----------------------
# Price Prediction
# ----------------------
elif page == "Price Prediction":
    st.title("💰 Price Prediction Model")
    y = np.log1p(df['price'])
    X_train, X_test, y_train, y_test = train_test_split(X_price, y, test_size=0.2, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    st.metric("MAE", round(mean_absolute_error(np.expm1(y_test), np.expm1(y_pred)),2))
    st.metric("R²", round(r2_score(np.expm1(y_test), np.expm1(y_pred)),3))
    st.caption("The model predicts listing prices within a small error range.")
    
    st.subheader("Top Features Driving Price")
    importances = pd.Series(reg.feature_importances_, index=X_price.columns).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10,5))
    importances.plot(kind="bar", color=AIRBNB_RED, ax=ax)
    st.pyplot(fig)
    st.caption("These features are most influential in predicting listing prices.")

# ----------------------
# Demand Classification
# ----------------------
elif page == "Demand Classification":
    st.title("📈 Demand Classification Model")
    scores = cross_val_score(clf, X_demand, df['demand'], cv=5, scoring="accuracy", n_jobs=-1)
    st.metric("Cross-validated Accuracy", round(scores.mean(),3))
    st.caption("Model accuracy shows how well it distinguishes between high-demand and low-demand listings.")
    
    st.subheader("Top Features Driving Demand")
    importances = pd.Series(clf.feature_importances_, index=X_demand.columns).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10,5))
    importances.plot(kind="bar", color=AIRBNB_TEAL, ax=ax)
    st.pyplot(fig)
    st.caption("These features are most influential in predicting demand.")

# ----------------------
# Interactive
# ----------------------
elif page == "Try it Yourself":
    st.title("🧪 Try Your Own Listing")
    st.write("Adjust the options below to predict price & demand for a new listing.")
    
    room_type = st.selectbox("Room Type", df['room type'].unique() if 'room type' in df.columns else ["Entire home/apt"])
    neigh_group = st.selectbox("Neighbourhood Group", df['neighbourhood group'].unique() if 'neighbourhood group' in df.columns else ["Other"])
    min_nights = st.slider("Minimum Nights", 1, 30, 2)
    service_fee = st.slider("Service Fee ($)", 0, 200, 20)
    
    input_dict = {
        "minimum nights": min_nights,
        "service fee": service_fee,
        "room type": room_type,
        "neighbourhood group": neigh_group
    }
    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df).reindex(columns=X_price.columns, fill_value=0)
    
    price_pred = np.expm1(reg.predict(input_df)[0])
    demand_pred = clf.predict(input_df.reindex(columns=X_demand.columns, fill_value=0))[0]
    
    st.subheader("Predictions")
    st.success(f"💰 Estimated Price: ${price_pred:,.0f}")
    st.success(f"📈 Demand Prediction: {'High' if demand_pred==1 else 'Low'}")
