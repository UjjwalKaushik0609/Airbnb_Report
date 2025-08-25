
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, classification_report

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Airbnb Analysis & Prediction", layout="wide")
sns.set_style("darkgrid")

st.title("🏠 Airbnb Data Analysis & Prediction Dashboard")
st.markdown("This app explores Airbnb listings data, builds machine learning models for **price prediction** and **demand classification**, and lets you try interactive predictions.")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Airbnb_Cleaned_Ready.csv.gz", low_memory=False)
    if "last review" in df.columns:
        df["last review"] = pd.to_datetime(df["last review"], errors="coerce")
    return df

df = load_data()

# -------------------------------
# Train Models (cached)
# -------------------------------
@st.cache_resource
def train_price_model(df):
    df_ml = df.copy()
    # One-hot encode categorical
    cat_cols = [c for c in ['room type','neighbourhood group','instant_bookable',
                            'cancellation_policy','host_identity_verified'] if c in df_ml.columns]
    df_ml = pd.get_dummies(df_ml, columns=cat_cols, drop_first=True)
    # Drop leakage
    leakage_cols = [col for col in df_ml.columns if 'price' in col.lower() or 'review' in col.lower() or 'avail' in col.lower()]
    X = df_ml.drop(columns=leakage_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    y = np.log1p(df['price'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train)
    return reg, X

@st.cache_resource
def train_demand_model(df):
    df_ml = df.copy()
    # One-hot encode categorical
    cat_cols = [c for c in ['room type','neighbourhood group','instant_bookable',
                            'cancellation_policy','host_identity_verified'] if c in df_ml.columns]
    df_ml = pd.get_dummies(df_ml, columns=cat_cols, drop_first=True)
    # Drop leakage
    leakage_cols = [col for col in df_ml.columns if 'price' in col.lower() or 'review' in col.lower() or 'avail' in col.lower() or 'demand' in col.lower()]
    X = df_ml.drop(columns=leakage_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    y = df['demand']
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
    clf.fit(X, y)
    return clf, X

reg, X_price = train_price_model(df)
clf, X_demand = train_demand_model(df)

# -------------------------------
# Navigation
# -------------------------------
section = st.sidebar.radio("Navigate", ["📊 Exploratory Data Analysis", "💰 Price Prediction", "📈 Demand Classification", "🧪 Try It Yourself"])

# -------------------------------
# EDA
# -------------------------------
if section == "📊 Exploratory Data Analysis":
    st.header("Exploratory Data Analysis (EDA)")

    if 'price' in df.columns:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(10,5))
        sns.histplot(df['price'], bins=50, kde=True, ax=ax, color="skyblue")
        st.pyplot(fig)
        st.markdown("📌 **Explanation:** Most listings are concentrated at lower price ranges, while a few very expensive outliers exist. This is typical in Airbnb data.")

        st.subheader("Price Outliers")
        fig, ax = plt.subplots(figsize=(10,3))
        sns.boxplot(x=df['price'], ax=ax, color="salmon")
        st.pyplot(fig)
        st.markdown("📌 **Explanation:** The boxplot highlights extreme price outliers compared to the majority of affordable listings.")

    if 'room type' in df.columns:
        st.subheader("Room Type Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(data=df, x='room type', ax=ax, palette="viridis")
        st.pyplot(fig)
        st.markdown("📌 **Explanation:** Entire homes/apartments usually dominate the market, while shared rooms are less common.")

    if 'neighbourhood group' in df.columns:
        st.subheader("Neighbourhood Group Distribution")
        order = df['neighbourhood group'].value_counts().index
        fig, ax = plt.subplots(figsize=(8,5))
        sns.countplot(data=df, y='neighbourhood group', order=order, ax=ax, palette="magma")
        st.pyplot(fig)
        st.markdown("📌 **Explanation:** Some neighbourhood groups (like Manhattan or Brooklyn in NYC data) have a much higher concentration of listings.")

    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)
    st.markdown("📌 **Explanation:** This heatmap shows which numeric features move together. Strong correlations may indicate redundancy or important relationships.")

    if 'last review' in df.columns:
        st.subheader("Reviews Over Time")
        reviews_over_time = df.dropna(subset=['last review']).groupby(df['last review'].dt.to_period('M')).size()
        reviews_over_time.index = reviews_over_time.index.to_timestamp()
        fig, ax = plt.subplots(figsize=(12,5))
        reviews_over_time.plot(kind="line", ax=ax, color="lime")
        st.pyplot(fig)
        st.markdown("📌 **Explanation:** The number of reviews per month shows how Airbnb usage and demand has changed over time.")

# -------------------------------
# Price Prediction (Regression)
# -------------------------------
elif section == "💰 Price Prediction":
    st.header("Price Prediction with Random Forest")

    df_ml = df.copy()
    cat_cols = [c for c in ['room type','neighbourhood group','instant_bookable','cancellation_policy','host_identity_verified'] if c in df_ml.columns]
    df_ml = pd.get_dummies(df_ml, columns=cat_cols, drop_first=True)
    leakage_cols = [col for col in df_ml.columns if 'price' in col.lower() or 'review' in col.lower() or 'avail' in col.lower()]
    X_reg = df_ml.drop(columns=leakage_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    y_reg = np.log1p(df['price'])

    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_test_orig, y_pred_orig = np.expm1(y_test), np.expm1(y_pred)

    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**R² Score:** {r2:.3f}")
    st.markdown("📌 **Interpretation:** On average, the model's predicted prices differ from actual prices by the MAE amount. R² shows how much variance is explained.")

    st.subheader("Top Features Driving Price")
    importances = pd.Series(reg.feature_importances_, index=X_reg.columns).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10,4))
    importances.plot(kind='bar', ax=ax, color="skyblue")
    st.pyplot(fig)
    st.markdown("📌 **Explanation:** These features are the strongest predictors of price according to the Random Forest model.")

# -------------------------------
# Demand Classification (Classifier)
# -------------------------------
elif section == "📈 Demand Classification":
    st.header("Demand Classification with Random Forest")

    df_ml = df.copy()
    cat_cols = [c for c in ['room type','neighbourhood group','instant_bookable','cancellation_policy','host_identity_verified'] if c in df_ml.columns]
    df_ml = pd.get_dummies(df_ml, columns=cat_cols, drop_first=True)
    leakage_cols = [col for col in df_ml.columns if 'price' in col.lower() or 'review' in col.lower() or 'avail' in col.lower() or 'demand' in col.lower()]
    X_clf = df_ml.drop(columns=leakage_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    y_clf = df['demand']

    scores = cross_val_score(clf, X_clf, y_clf, cv=5, scoring='accuracy', n_jobs=-1)
    st.write(f"**Cross-validated Accuracy:** {scores.mean():.3f}")
    st.markdown("📌 **Interpretation:** Accuracy indicates the proportion of listings where the model correctly predicts whether demand is high or low.")

    st.subheader("Top Features Driving Demand")
    importances = pd.Series(clf.feature_importances_, index=X_clf.columns).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10,4))
    importances.plot(kind='bar', ax=ax, color="orange")
    st.pyplot(fig)
    st.markdown("📌 **Explanation:** These features most strongly influence whether a listing is classified as high or low demand.")

# -------------------------------
# Interactive Prediction
# -------------------------------
elif section == "🧪 Try It Yourself":
    st.header("Interactive Airbnb Price & Demand Prediction")

    st.markdown("Fill in the details of a hypothetical listing to get predicted **price** and **demand level**.")

    # Sidebar input form
    room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
    neighbourhood_group = st.selectbox("Neighbourhood Group", df['neighbourhood group'].unique())
    min_nights = st.slider("Minimum Nights", 1, 30, 3)
    host_listings = st.slider("Host Listings Count", 1, 10, 1)

    input_dict = {
        "minimum nights": min_nights,
        "calculated host listings count": host_listings,
        "room type": room_type,
        "neighbourhood group": neighbourhood_group
    }
    input_df = pd.DataFrame([input_dict])

    # Encode
    cat_cols = [c for c in ['room type','neighbourhood group'] if c in input_df.columns]
    input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    input_df = input_df.reindex(columns=X_price.columns, fill_value=0)

    # Price prediction
    price_pred_log = reg.predict(input_df)[0]
    price_pred = np.expm1(price_pred_log)

    # Demand prediction
    input_df_clf = input_df.reindex(columns=X_demand.columns, fill_value=0)
    demand_pred = clf.predict(input_df_clf)[0]

    st.success(f"💰 Predicted Price: **${price_pred:.2f}**")
    st.success(f"📈 Predicted Demand: **{'High' if demand_pred==1 else 'Low'}**")
    st.markdown("📌 **Explanation:** The price is estimated based on historical listings with similar features, while demand is classified relative to the dataset median.")
