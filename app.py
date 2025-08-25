import os
import gdown
import joblib
import requests
import streamlit as st

# ==============================
# Google Drive File IDs
# ==============================
REG_MODEL_ID = "1ENewUWOs_tiz4hZt8WUnRlcU6-naoTgu"
CLF_MODEL_ID = "1Bh7Ig0m8ZFg4gi2zdVGJ7JtZIkkxHHRp"
REG_FEAT_ID  = "1ajaBfSUiorYptqjjC7kKKrVFzYFHOeU-"
CLF_FEAT_ID  = "1IwHCTMGUtmjRNI-QHd-zU9Lr7r59cWOc"


# ==============================
# Utility: check if update needed
# ==============================
def is_update_needed(file_id, local_file):
    """Check if Google Drive file size differs from local cached file."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        gdrive_size = int(response.headers.get("Content-Length", 0))
        local_size = os.path.getsize(local_file) if os.path.exists(local_file) else -1
        return gdrive_size != local_size
    except Exception:
        return True  # if check fails, force re-download


# ==============================
# Utility: download from Drive
# ==============================
def download_from_gdrive(file_id, output=None, force=False):
    """Download file from Google Drive if not cached or outdated."""
    if output is None:
        output = f"{file_id}.pkl"

    if not force and os.path.exists(output) and not is_update_needed(file_id, output):
        return output  # use cached file

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)
    return output


# ==============================
# Load models (cached in Streamlit)
# ==============================
@st.cache_resource
def load_models():
    file_ids = [
        (REG_MODEL_ID, "reg_model.pkl"),
        (CLF_MODEL_ID, "clf_model.pkl"),
        (REG_FEAT_ID, "reg_features.pkl"),
        (CLF_FEAT_ID, "clf_features.pkl"),
    ]

    progress = st.progress(0, text="📥 Downloading model files...")
    total = len(file_ids)

    paths = []
    for i, (fid, name) in enumerate(file_ids, start=1):
        with st.spinner(f"Downloading {name}..."):
            path = download_from_gdrive(fid, name)
            paths.append(path)
        progress.progress(i / total, text=f"Downloaded {i}/{total} files")

    st.success("✅ All files downloaded and loaded!")

    # Load objects with joblib
    reg_model = joblib.load(paths[0])
    clf_model = joblib.load(paths[1])
    reg_features = joblib.load(paths[2])
    clf_features = joblib.load(paths[3])

    return reg_model, clf_model, reg_features, clf_features


# ==============================
# Sidebar: Manual refresh option
# ==============================
st.sidebar.header("⚙️ Settings")

if st.sidebar.button("♻️ Clear Cache & Redownload"):
    st.cache_resource.clear()
    st.warning("Cache cleared! Reloading models...")
    st.rerun()


# ==============================
# Example usage in your app
# ==============================
st.title("🏠 Airbnb Report App")

reg_model, clf_model, reg_features, clf_features = load_models()

st.success("Models are ready to use!")
