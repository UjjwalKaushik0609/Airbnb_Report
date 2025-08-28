# app.py
import os
import io
import requests
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk

st.set_page_config(page_title="Airbnb Price Predictor", page_icon="🏡", layout="wide")

# ==========
# Utilities
# ==========
HF_MODEL_URL = "https://huggingface.co/UjjwalKaushik/Airbnb_model/resolve/main/best_random_forest.pkl?download=1"
LOCAL_MODEL_PATH = "best_random_forest.pkl"
LOCAL_DATASET_PATH = "cleaned_dataset.csv"
TARGET = "price"  # we’re predicting nightly price

# Columns used for training (derived from your project)
RAW_FEATURES = [
    "neighbourhood group", "neighbourhood", "room type",
    "host_identity_verified", "instant_bookable", "cancellation_policy",
    "lat", "long", "Construction year",
    "minimum nights", "number of reviews", "reviews per month",
    "availability 365"
]
# Optional: columns that exist but should not be used for prediction
DROP_ALWAYS = ["id", "NAME", "host id", TARGET]

def ensure_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    """Ensure all required raw feature columns exist; add missing as NaNs."""
    for c in required:
        if c not in df.columns:
            df[c] = np.nan
    # Only keep required + any extras the pipeline can ignore
    return df[required].copy()

# ====================
# Cached load helpers
# ====================
@st.cache_resource
def load_model() -> object:
    if not os.path.exists(LOCAL_MODEL_PATH):
        r = requests.get(HF_MODEL_URL, timeout=60)
        r.raise_for_status()
        with open(LOCAL_MODEL_PATH, "wb") as f:
            f.write(r.content)
    return joblib.load(LOCAL_MODEL_PATH)

@st.cache_data
def load_dataset() -> pd.DataFrame:
    # Try CSV first, then Excel as fallback
    if os.path.exists(LOCAL_DATASET_PATH):
        return pd.read_csv(LOCAL_DATASET_PATH, low_memory=False)
    # Fallbacks if filename differs
    for alt in ["Airbnb_dataset.csv", "cleaned_dataset.xls", "cleaned_dataset.xlsx"]:
        if os.path.exists(alt):
            if alt.endswith(".csv"):
                return pd.read_csv(alt, low_memory=False)
            else:
                return pd.read_excel(alt)
    raise FileNotFoundError(
        "Could not find cleaned dataset. Ensure 'cleaned_dataset.csv' is in the repo."
    )

model = load_model()
df = load_dataset()

# Use dataset to confirm columns and build allowed options for selects
df_cols = df.columns.tolist()

# ==========================
# Sidebar: Single Prediction
# ==========================
st.sidebar.title("🧮 Single Listing Prediction")

# Defensive default values
def _safe_default(col, fallback):
    return df[col].iloc[0] if col in df_cols and df[col].notna().any() else fallback

neighbourhood_group = st.sidebar.selectbox(
    "Neighbourhood Group",
    sorted(df["neighbourhood group"].dropna().unique().tolist()) if "neighbourhood group" in df_cols else []
)
neighbourhood = st.sidebar.selectbox(
    "Neighbourhood",
    sorted(df["neighbourhood"].dropna().unique().tolist()) if "neighbourhood" in df_cols else []
)
room_type = st.sidebar.selectbox(
    "Room Type",
    sorted(df["room type"].dropna().unique().tolist()) if "room type" in df_cols else []
)
host_identity_verified = st.sidebar.selectbox(
    "Host Identity Verified",
    sorted(df["host_identity_verified"].dropna().unique().tolist()) if "host_identity_verified" in df_cols else [0, 1]
)
instant_bookable = st.sidebar.selectbox(
    "Instant Bookable",
    sorted(df["instant_bookable"].dropna().unique().tolist()) if "instant_bookable" in df_cols else [0, 1]
)
cancellation_policy = st.sidebar.selectbox(
    "Cancellation Policy",
    sorted(df["cancellation_policy"].dropna().unique().tolist()) if "cancellation_policy" in df_cols else []
)

lat = st.sidebar.number_input("Latitude", value=float(df["lat"].median() if "lat" in df_cols else 40.7))
long = st.sidebar.number_input("Longitude", value=float(df["long"].median() if "long" in df_cols else -73.9))
construction_year = st.sidebar.number_input("Construction Year", min_value=1800, max_value=2100,
                                            value=int(df["Construction year"].median() if "Construction year" in df_cols else 2000))
minimum_nights = st.sidebar.number_input("Minimum Nights", min_value=1, value=int(df["minimum nights"].median() if "minimum nights" in df_cols else 1))
number_of_reviews = st.sidebar.number_input("Number of Reviews", min_value=0, value=int(df["number of reviews"].median() if "number of reviews" in df_cols else 0))
reviews_per_month = st.sidebar.number_input("Reviews per Month", min_value=0.0, value=float(df["reviews per month"].median() if "reviews per month" in df_cols else 0.0))
availability_365 = st.sidebar.number_input("Availability (days/year)", min_value=0, max_value=365,
                                           value=int(df["availability 365"].median() if "availability 365" in df_cols else 180))

single_input = pd.DataFrame([{
    "neighbourhood group": neighbourhood_group,
    "neighbourhood": neighbourhood,
    "room type": room_type,
    "host_identity_verified": host_identity_verified,
    "instant_bookable": instant_bookable,
    "cancellation_policy": cancellation_policy,
    "lat": lat,
    "long": long,
    "Construction year": construction_year,
    "minimum nights": minimum_nights,
    "number of reviews": number_of_reviews,
    "reviews per month": reviews_per_month,
    "availability 365": availability_365
}])

if st.sidebar.button("🔮 Predict Price"):
    try:
        X_single = ensure_columns(single_input, RAW_FEATURES)
        pred = float(model.predict(X_single)[0])
        st.sidebar.success(f"Estimated nightly price: **${pred:,.2f}**")
    except Exception as e:
        st.sidebar.error(f"Prediction failed: {e}")

# ==========
# Main Tabs
# ==========
st.title("🏡 Airbnb Price Prediction & Insights")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Predict (Bulk)", "🗺️ Map", "🧠 Explain Model", "🔍 Explore Data"]
)

# ----------------------
# Tab 1: Bulk Prediction
# ----------------------
with tab1:
    st.subheader("Bulk Predict from CSV")
    st.write("Upload a CSV with columns matching the training features. "
             "You can download a template below.")

    # Build & offer a template CSV
    template = pd.DataFrame(columns=RAW_FEATURES)
    st.download_button(
        "Download input template CSV",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="airbnb_input_template.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv", "xls", "xlsx"])
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)

            st.write("Preview:", df_up.head())

            # Align columns
            X_bulk = ensure_columns(df_up, RAW_FEATURES)

            # Predict
            preds = model.predict(X_bulk)
            out = df_up.copy()
            out["predicted_price"] = preds

            st.success("Predictions complete.")
            st.write(out.head())

            # Download results
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions CSV",
                data=csv_bytes,
                file_name="airbnb_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ---------------
# Tab 2: Map View
# ---------------
with tab2:
    st.subheader("Map of Listings (Predicted Price Color)")
    # Take a sample for performance if dataset is huge
    map_df = df.copy()
    if len(map_df) > 5000:
        map_df = map_df.sample(5000, random_state=42)

    # Ensure features exist; compute predictions for map
    try:
        X_map = ensure_columns(map_df, RAW_FEATURES)
        map_df["predicted_price"] = model.predict(X_map)
    except Exception as e:
        st.error(f"Cannot compute predictions for map: {e}")
        map_df["predicted_price"] = np.nan

    # Drop rows without coordinates
    map_df = map_df.dropna(subset=["lat", "long"])
    if map_df.empty:
        st.info("No coordinates available to plot.")
    else:
        # Build a pydeck map
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[long, lat]',
            get_radius=50,
            get_fill_color='[min(255, predicted_price*0.5), 100, 150]',
            pickable=True
        )
        view_state = pdk.ViewState(
            latitude=float(map_df["lat"].mean()),
            longitude=float(map_df["long"].mean()),
            zoom=10,
            pitch=0
        )
        r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Price: {predicted_price}"})
        st.pydeck_chart(r)

# -----------------------
# Tab 3: Explainability
# -----------------------
with tab3:
    st.subheader("Feature Importances (Tree-based)")
    st.write("Shows raw feature importances from the underlying tree model after preprocessing.")

    try:
        # Try to extract preprocessor + model
        # Works if your saved object is a Pipeline(preprocessor, model)
        preproc = None
        core_model = None

        # Case 1: Pipeline with named steps
        if hasattr(model, "named_steps"):
            preproc = model.named_steps.get("preprocessor", None)
            core_model = model.named_steps.get("model", None)
        # Case 2: ColumnTransformer directly followed by estimator
        if core_model is None and hasattr(model, "steps"):
            # last step likely the estimator
            core_model = model.steps[-1][1]
            # try find preprocessor by name
            for name, step in model.steps:
                if "preprocess" in name or "preprocessor" in name:
                    preproc = step
                    break

        # Get feature names post-preprocessing if possible
        feature_names = RAW_FEATURES
        if preproc is not None and hasattr(preproc, "get_feature_names_out"):
            feature_names = preproc.get_feature_names_out(RAW_FEATURES).tolist()

        # Get importances
        if hasattr(core_model, "feature_importances_"):
            importances = core_model.feature_importances_
            imp_df = pd.DataFrame({
                "feature": feature_names[:len(importances)],
                "importance": importances
            }).sort_values("importance", ascending=False).head(25)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(data=imp_df, x="importance", y="feature", ax=ax)
            ax.set_title("Top Feature Importances")
            st.pyplot(fig)
            st.dataframe(imp_df, use_container_width=True)
        else:
            st.info("The loaded estimator does not expose `feature_importances_`.")
    except Exception as e:
        st.error(f"Could not compute feature importances: {e}")

# -------------------
# Tab 4: Explore Data
# -------------------
with tab4:
    st.subheader("Basic EDA")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Price Distribution**")
        if "price" in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df["price"], bins=60, kde=True, ax=ax)
            ax.set_xlim(0, np.nanpercentile(df["price"], 99))
            st.pyplot(fig)
        else:
            st.info("`price` column not found in dataset.")

    with colB:
        st.markdown("**Average Price by Neighbourhood Group**")
        if "neighbourhood group" in df.columns and "price" in df.columns:
            grp = df.groupby("neighbourhood group")["price"].mean().sort_values(ascending=False)
            fig, ax = plt.subplots()
            grp.plot(kind="bar", ax=ax)
            ax.set_ylabel("Avg Price")
            st.pyplot(fig)
        else:
            st.info("Required columns not found.")

    st.markdown("**Room Type Distribution**")
    if "room type" in df.columns:
        fig, ax = plt.subplots()
        df["room type"].value_counts().plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.info("`room type` column not found.")
