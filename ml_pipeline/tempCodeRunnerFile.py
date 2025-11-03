import os
import numpy as np
import pandas as pd
import joblib
from ml_pipeline.preprocessing import preprocess_single_record

def predict_price(record, model_name="rf"):
    """
    Predict price for a single product record.
    Ensures preprocessing matches training exactly.
    """
    # 1. Paths
    model_path = f"model/{model_name}_model.pkl"
    scaler_path = "model/scaler.pkl"

    # 2. Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)

    # 3. Preprocess input
    X = preprocess_single_record(record)  # must return DataFrame
    if isinstance(X, pd.Series):
        X = X.to_frame().T

    # 4. Predict in log-space and invert
    pred_log = model.predict(X)[0]
    pred = np.expm1(pred_log)

    # 5. Round result
    return round(float(pred), 2)
