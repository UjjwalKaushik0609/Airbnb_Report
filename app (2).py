import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, classification_report

# ---------------------------------
# App Config
# ---------------------------------
st.set_page_config(page_title="Airbnb Analytics", layout="wide")
sns.set_style("darkgrid")
plt.style.use("dark_background")

# ---------------------------------
# Load Data
# ---------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Airbnb_Cleaned_Ready.csv.gz", low_memory=False)

df = load_data()
st.title("🏠 Airbnb Portfolio Dashboard")

st.markdown("Explore Airbnb data, visualize trends, and test ML models for **price prediction** and **demand classification**.")

# ---------------------------------
# Sidebar Filters
# ---------------------------------
st.sidebar.header("🔎 Filters")
if "neighbourhood group" in df.columns:
    ng_filter = st.sidebar.multiselect("Neighbourhood Group", df["neighbourhood group"].unique())
    if ng_filter:
        df = df[df["neighbourhood group"].isin(ng_filter)]

if "room type" in df.columns:
    rt_filter = st.sidebar.multiselect("Room Type", df["room type"].unique())
    if rt_filter:
        df = df[df["room type"].isin(rt_filter)]

st.sidebar.write("Dataset size:", df.shape)

# ---------------------------------
# EDA Tabs
# ---------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Distribution", "🏘 Room Types", "🌍 Map & Correlation", "📈 Reviews Over Time"])

with tab1:
    if "price" in df.columns:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.histplot(df["price"].dropna(), bins=50, kde=True, ax=ax, color="cyan")
        ax.set_title("Distribution of Listing Prices")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10,4))
        sns.boxplot(x=df["price"], ax=ax, color="purple")
        ax.set_title("Price Outliers")
        st.pyplot(fig)

with tab2:
    if "room type" in df.columns:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.countplot(data=df, x="room type", ax=ax, palette="viridis")
        ax.set_title("Room Type Distribution")
        st.pyplot(fig)

with tab3:
    if "neighbourhood group" in df.columns:
        fig, ax = plt.subplots(figsize=(10,6))
        order = df["neighbourhood group"].value_counts().index
        sns.countplot(data=df, y="neighbourhood group", order=order, ax=ax, palette="magma")
        ax.set_title("Listings by Neighbourhood Group")
        st.pyplot(fig)

    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

with tab4:
    if "last review" in df.columns:
        reviews_over_time = df.dropna(subset=["last review"]).groupby(df["last review"].astype("datetime64[M]")).size()
        fig, ax = plt.subplots(figsize=(12,5))
        reviews_over_time.plot(kind="line", ax=ax, color="lime")
        ax.set_title("Number of Reviews Over Time")
        st.pyplot(fig)

# ---------------------------------
# Machine Learning Section
# ---------------------------------
st.header("🤖 Machine Learning Models")

ml_choice = st.radio("Select a model to run:", ["💲 Price Prediction", "📈 Demand Classification"])

# --- Price Prediction ---
if ml_choice == "💲 Price Prediction" and "price" in df.columns:
    df_ml = df.copy()
    cat_cols = [c for c in ["room type", "neighbourhood group"] if c in df_ml.columns]
    df_ml = pd.get_dummies(df_ml, columns=cat_cols, drop_first=True)
    df_ml.columns = [c.lower().strip() for c in df_ml.columns]

    leakage_cols = [c for c in df_ml.columns if "price" in c or "review" in c or "avail" in c or "demand" in c]
    X = df_ml.drop(columns=leakage_cols, errors="ignore").select_dtypes(include=[np.number]).fillna(0)
    y = np.log1p(df_ml["price"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    y_test_orig, y_pred_orig = np.expm1(y_test), np.expm1(y_pred)

    st.subheader("📊 Price Prediction Results")
    st.write("MAE:", round(mean_absolute_error(y_test_orig, y_pred_orig), 2))
    st.write("R²:", round(r2_score(y_test_orig, y_pred_orig), 3))

    fig, ax = plt.subplots(figsize=(10,4))
    pd.Series(reg.feature_importances_, index=X.columns).sort_values(ascending=False).head(10).plot(kind="bar", ax=ax, color="orange")
    ax.set_title("Top Features Driving Price")
    st.pyplot(fig)

# --- Demand Classification ---
if ml_choice == "📈 Demand Classification" and "demand" in df.columns:
    safe_cols = ["lat", "long", "neighbourhood group", "room type", "minimum nights"]
    df_clf = df[safe_cols + ["demand"]].dropna()
    df_clf = pd.get_dummies(df_clf, columns=["neighbourhood group", "room type"], drop_first=True)

    X = df_clf.drop(columns=["demand"])
    y = df_clf["demand"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.subheader("📊 Demand Classification Results")
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    st.write("Cross-validated Accuracy:", round(scores.mean(), 3))
    st.text(classification_report(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(10,4))
    pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10).plot(kind="bar", ax=ax, color="cyan")
    ax.set_title("Top Features Driving Demand")
    st.pyplot(fig)

    # Save models for interactive prediction
    reg_model = None
    try:
        reg_model = reg
    except:
        pass

    clf_model = clf

# ---------------------------------
# Interactive Prediction
# ---------------------------------
st.header("🎯 Try It Yourself: Predict Price & Demand")

with st.form("predict_form"):
    lat = st.number_input("Latitude", value=float(df["lat"].median()))
    lon = st.number_input("Longitude", value=float(df["long"].median()))
    nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=3)
    room_type = st.selectbox("Room Type", df["room type"].unique())
    neigh_group = st.selectbox("Neighbourhood Group", df["neighbourhood group"].unique())
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame({
        "lat": [lat],
        "long": [lon],
        "minimum nights": [nights],
        "room type": [room_type],
        "neighbourhood group": [neigh_group]
    })

    input_df = pd.get_dummies(input_df, columns=["room type","neighbourhood group"], drop_first=True)
    for col in X.columns:
        if col not in input_df:
            input_df[col] = 0
    input_df = input_df[X.columns]

    # Demand prediction
    demand_pred = clf.predict(input_df)[0]
    demand_label = "High Demand" if demand_pred == 1 else "Low Demand"

    # Price prediction (reuse reg model if available)
    price_pred = None
    if 'reg' in locals():
        price_pred_log = reg.predict(input_df)[0]
        price_pred = np.expm1(price_pred_log)

    st.subheader("📌 Predictions")
    if price_pred:
        st.write(f"💲 Estimated Price: **${price_pred:,.0f}**")
    st.write(f"🔥 Demand Level: **{demand_label}**")

st.success("✅ App Ready! Use the sidebar to filter data and explore insights.")
