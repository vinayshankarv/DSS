import os
import numpy as np
import pandas as pd
import joblib
from ml_pipeline.preprocessing import preprocess_single_record


def predict_price(record, model_name="rf"):
    """
    Predict price for a single product record.
    Handles preprocessing and feature alignment without scaling.
    """

    # -----------------------------
    # 1️⃣ Validate input
    # -----------------------------
    if not isinstance(record, dict):
        raise TypeError("❌ Input must be a dictionary containing product fields like title, platform, rating, etc.")

    # -----------------------------
    # 2️⃣ Load model only (no scaler)
    # -----------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "model", f"{model_name}_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    model = joblib.load(model_path)

    # -----------------------------
    # 3️⃣ Preprocess record
    # -----------------------------
    X = preprocess_single_record(record)
    if isinstance(X, pd.Series):
        X = X.to_frame().T

    # -----------------------------
    # 4️⃣ Align features with training model
    # -----------------------------
    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is not None:
        missing = [c for c in expected_features if c not in X.columns]
        for col in missing:
            X[col] = 0
        X = X[expected_features]

    # -----------------------------
    # 5️⃣ Predict using model directly (no scaling)
    # -----------------------------
    try:
        pred_log = model.predict(X)[0]
    except Exception as e:
        raise RuntimeError(f"❌ Model prediction failed: {e}")

    # -----------------------------
    # 6️⃣ Reverse log-transform to rupees
    # -----------------------------
    pred_price = np.expm1(pred_log)

    # -----------------------------
    # 7️⃣ Return clean float for UI/log
    # -----------------------------
    return round(float(pred_price), 2)
