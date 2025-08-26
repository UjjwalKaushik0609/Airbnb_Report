@st.cache_resource
def load_models():
    file_ids = [
        (REG_MODEL_ID, "reg_model.pkl.gz"),
        (CLF_MODEL_ID, "clf_model.pkl.gz"),
        (REG_FEAT_ID, "reg_feature.pkl"),
        (CLF_FEAT_ID, "clf_feature.pkl"),
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

    reg_model = joblib.load(paths[0])
    clf_model = joblib.load(paths[1])
    reg_features = joblib.load(paths[2])
    clf_features = joblib.load(paths[3])

    return reg_model, clf_model, reg_features, clf_features

