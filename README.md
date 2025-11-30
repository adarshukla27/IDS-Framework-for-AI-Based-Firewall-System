# Intrusion Detection Framework for AI-Based Firewall — Anomaly Detection using UNSW NB15 Dataset

Lightweight Streamlit app and notebooks for anomaly detection on network traffic (UNSW NB15). The project demonstrates stacked approaches (Autoencoder, XGBoost, Isolation Forest) and tools for local explainability and visualization.

## Contents
- app.py — Streamlit app for inference, visualization and model explainability.
- Anomaly_Detection_Using_UNSW_NB15.ipynb - EDA and model experiments.
- Anomaly_Detection_Using_UNSW_NB15(Auto_Encoders_IF_training).ipynb - Autoencoder + Isolation Forest training experiments.
- models/
  - autoencoder_zero_day.h5 - pretrained autoencoder weights.
  - feature_names.json - model features ordering used at inference.
  - isolation_forest_zero_day.pkl - pretrained Isolation Forest model.
  - label_encoder.pkl - pretrained label encoder for categorical features.
  - logistic_regression_baseline.pkl - pretrained Logistic Regression model.
  - random_forest_baseline.pkl - pretrained Random Forest model.
  - standard_scaler.pkl - pretrained standard scaler for feature normalization.
  - support_vector_machine_baseline.pkl - pretrained SVM model.
  - xgboost_baseline.pkl - pretrained XGBoost model.
- requirements.txt - Python dependencies.

## Quick start (Windows)
1. Create and activate virtual environment:
   - python -m venv .venv
   - .venv\Scripts\activate
2. Install dependencies:
   - pip install -r requirements.txt
3. Run the app (Streamlit):
   - streamlit run app.py
4. Open the notebooks:
   - jupyter notebook

## Model / Data notes
- Data used in the notebooks: UNSW NB15 dataset (download through Kaggle).
- feature_names.json must match the features expected by saved models; update whenever preprocessing or feature selection changes.
- SHAP values are computed only for XGBoost model from baseline models because XGBoost is best performing model in our experiments to explain feature contributions to predictions.
- A hybrid approach is used for zero-day anomaly detection: Autoencoder for anomaly scoring, followed by Isolation Forest for final classification.
- Autoencoder reconstruction error is visualized using a log-scale histogram to highlight small/zero values safely.
- Further explainability is provided through SHAP values for the Zero-day Anomaly Detection models(Autoencoder + Isolation Forest).

## Development & testing
- Use the notebooks to retrain models; save artifacts into models/.
- Use pytest for unit tests (if you add tests).
- Pin versions in requirements.txt before deploying.


## Contributing
- Open issues or PRs for bugs, improvements, or updated models.
- Keep model artifacts under models/ and update feature_names.json when changing features.

