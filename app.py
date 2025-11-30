import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

st.set_page_config(page_title="AI-Based Firewall Dashboard", layout="wide")

# ---------------------------
# LOAD ALL MODELS & COMPONENTS
# ---------------------------

@st.cache_resource
def load_models():
    try:
        xgb_model = joblib.load("models/xgboost_baseline.pkl")
        iso_model = joblib.load("models/isolation_forest_zero_day.pkl")
        scaler = joblib.load("models/standard_scaler.pkl")
        encoder = joblib.load("models/label_encoder.pkl")  # could be dict
        autoencoder = tf.keras.models.load_model("models/autoencoder_zero_day.h5", compile=False)
        feature_names = json.load(open("models/feature_names.json"))
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None
    return xgb_model, iso_model, scaler, encoder, autoencoder, feature_names


xgb_model, iso_model, scaler, encoder, autoencoder, feature_names = load_models()

st.title("🔐 AI-Based Firewall System — Visualization & Zero-Day Detection Dashboard")
st.write("Upload network flow(s) to analyze using Supervised ML + Zero-Day Detection + SHAP Explainability.")


# ---------------------------
# PREPROCESSING FUNCTION
# ---------------------------

def preprocess(df):
    df = df.copy()
    required_cats = ["proto", "service", "state"]
    for col in required_cats:
        if col not in df.columns:
            st.error(f"Column '{col}' missing from input!")
            return None

    # Support encoder being either a dict or a single LabelEncoder
    if isinstance(encoder, dict):
        for col in required_cats:
            enc = encoder.get(col)
            if enc is None:
                st.error(f"No encoder found for column '{col}'")
                return None
            # safe transform: unseen categories -> -1
            df[col] = df[col].astype(str).map(lambda v: enc.transform([v])[0] if v in enc.classes_ else -1)
    else:
        # single encoder fallback (less likely correct for multiple columns)
        for col in required_cats:
            df[col] = df[col].astype(str).map(lambda v: encoder.transform([v])[0] if v in encoder.classes_ else -1)

    for col in ["id", "label", "attack_cat"]:
        if col in df.columns:
            df = df.drop(col, axis=1)

    missing_cols = set(feature_names) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    df = df[feature_names]
    df_scaled = scaler.transform(df)
    return df, df_scaled


# ---------------------------
# PREDICTION FUNCTIONS
# ---------------------------

def predict_supervised(df_scaled):
    if hasattr(xgb_model, "predict_proba"):
        probs = xgb_model.predict_proba(df_scaled)[:, 1]
        preds = (probs > 0.5).astype(int)
    else:
        preds = xgb_model.predict(df_scaled)
        # fallback for probs: cast preds to float (or set NaNs)
        probs = preds.astype(float)
    return preds, probs


def predict_zero_day(df_scaled):
    # Autoencoder reconstruction error
    recon = autoencoder.predict(df_scaled, verbose=0)
    recon_err = np.mean(np.square(df_scaled - recon), axis=1)

    # Isolation Forest anomaly score
    iso_score = iso_model.decision_function(df_scaled) * -1  # invert so higher=more anomaly

    # Normalize both for hybrid
    re = (recon_err - recon_err.min()) / (recon_err.max() - recon_err.min() + 1e-8)
    is_ = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-8)

    hybrid = (re + is_) / 2

    return recon_err, iso_score, hybrid


# ---------------------------
# SHAP EXPLAINABILITY (XGBOOST ONLY)
# ---------------------------

@st.cache_resource
def load_shap_explainer():
    return shap.TreeExplainer(xgb_model)

explainer = load_shap_explainer()


def shap_plot(df_scaled):
    shap_values = explainer.shap_values(df_scaled)
    # handle list output (multiclass)
    if isinstance(shap_values, list):
        # attempt to show contribution toward class 1 if present
        shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_vals = shap_values
    # create DataFrame for features so feature names display correctly
    feats = pd.DataFrame(df_scaled, columns=feature_names)
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.summary_plot(shap_vals, features=feats, feature_names=feature_names, show=False)
    st.pyplot(fig)


# ---------------------------
# FILE UPLOAD SECTION
# ---------------------------

uploaded = st.file_uploader("Upload CSV (single or multiple flows)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("📄 Uploaded Data Preview")
    st.write(df.head())

    df_processed, df_scaled = preprocess(df)

    if df_processed is None:
        st.stop()

    # ---------------------------
    # SUPERVISED PREDICTION
    # ---------------------------
    st.subheader("🤖 Supervised ML Prediction (XGBoost)")
    preds, probs = predict_supervised(df_scaled)

    df["ML_Prediction"] = preds
    df["ML_Attack_Probability"] = probs

    st.dataframe(df[["ML_Prediction", "ML_Attack_Probability"]].head(10))

    # ---------------------------
    # ZERO-DAY DETECTION
    # ---------------------------
    st.subheader("🕵️ Zero-Day Detection (Autoencoder + Isolation Forest)")

    recon_err, iso_score, hybrid = predict_zero_day(df_scaled)

    df["AE_Recon_Error"] = recon_err
    df["IF_Anomaly_Score"] = iso_score
    df["Hybrid_Anomaly"] = hybrid

    st.dataframe(df[["AE_Recon_Error", "IF_Anomaly_Score", "Hybrid_Anomaly"]].head(10))

    # ---------------------------
    # VISUALIZATIONS
    # ---------------------------

    st.subheader("📊 Zero-Day Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Autoencoder Reconstruction Error")
        fig, ax = plt.subplots()
        sns.histplot(recon_err, bins=40, kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("### Isolation Forest Anomaly Score")
        fig, ax = plt.subplots()
        sns.histplot(iso_score, bins=40, kde=True, ax=ax)
        st.pyplot(fig)

    st.write("### Hybrid Zero-Day Anomaly Score")
    fig, ax = plt.subplots()
    sns.histplot(hybrid, bins=40, kde=True, ax=ax)
    st.pyplot(fig)

    # ---------------------------
    # SHAP for first row
    # ---------------------------
    st.subheader("🔍 SHAP Explanation for Supervised Model")

    shap_target_index = 0

    st.write(f"Showing SHAP explanation for Row #{shap_target_index}")

    shap_plot(df_scaled[shap_target_index:shap_target_index+1])

