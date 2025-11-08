# ml_pipeline/preprocessing.py
import os
import re
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_and_feature_engineer(path="outputs/scraped_results.csv"):
    # === Ensure directories ===
    os.makedirs("model", exist_ok=True)

    # === Load data ===
    df = pd.read_csv(path)

    # --- Title extraction ---
    if "title" in df.columns:
        df["title"] = df["title"].astype(str)
    elif "name" in df.columns:
        df["title"] = df["name"].astype(str)
    else:
        raise ValueError("Input CSV must contain a 'title' or 'name' column.")

    # --- Price cleaning ---
    df["price"] = pd.to_numeric(
        df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True),
        errors="coerce"
    )
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]  # sanity filter

    # --- Outlier trimming ---
    q_low, q_high = df["price"].quantile([0.01, 0.99])
    df = df[df["price"].between(q_low, q_high)]

    # --- Brand extraction ---
    df["brand"] = (
        df["title"]
        .str.extract(r"^([A-Za-z0-9]+)", expand=False)
        .fillna("UNKNOWN")
        .str.upper()
    )

    # --- Rating ---
    df["rating"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(0)

    # --- Timestamp and recency ---
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp.now()
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").fillna(pd.Timestamp.now())
    df["days_since"] = (datetime.now() - df["timestamp"]).dt.days

    # --- Text-derived features ---
    df["is_ultra"] = df["title"].str.contains("ultra", case=False).astype(int)
    df["is_fe"] = df["title"].str.contains(r"\bFE\b", case=False).astype(int)
    df["title_len"] = df["title"].str.len()

    # --- Brand-level price stats ---
    brand_stats = (
        df.groupby("brand")["price"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "brand_mean", "std": "brand_std"})
    )

    # Save brand stats for prediction-time lookup (one-line)
    brand_stats.reset_index().to_csv("model/brand_stats.csv", index=False)

    # merge stats back
    df = df.merge(brand_stats, left_on="brand", right_index=True, how="left")
    df["brand_mean"].fillna(df["price"].mean(), inplace=True)
    df["brand_std"].fillna(0, inplace=True)

    # --- Platform encoding ---
    df["platform"] = df.get("platform", "Unknown").fillna("Unknown")

    enc_path = "model/platform_encoder.pkl"
    if os.path.exists(enc_path):
        enc = joblib.load(enc_path)
        platform_ohe = enc.transform(df[["platform"]])
    else:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        platform_ohe = enc.fit_transform(df[["platform"]])
        joblib.dump(enc, enc_path)

    platform_cols = [f"platform_{c}" for c in enc.categories_[0]]
    df_platform = pd.DataFrame(platform_ohe, columns=platform_cols, index=df.index)
    df = pd.concat([df, df_platform], axis=1)

    # === Feature assembly ===
    base_features = ["rating", "days_since", "is_ultra", "is_fe", "title_len", "brand_mean", "brand_std"]
    features = base_features + platform_cols

    X = df[features].fillna(0)
    y = np.log1p(df["price"])  # log-transform target for stability

    # --- Scaling numeric features ---
    scaler_path = "model/scaler.pkl"
    numeric_cols = ["rating", "days_since", "title_len", "brand_mean", "brand_std"]

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    joblib.dump({"scaler": scaler, "columns": numeric_cols}, scaler_path)

    # === Train-test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
#               Single Record Preprocessor (for Predictions)
# ----------------------------------------------------------------------
def preprocess_single_record(record: dict):
    """
    Preprocess a single product record (for prediction).
    record: dict with keys like title, platform, rating, timestamp, etc.
    Returns: preprocessed 1-row DataFrame ready for model.predict()
    """
    import joblib
    import numpy as np
    import pandas as pd
    from datetime import datetime

    # --- Load encoders & scalers (fail fast if missing) ---
    enc_path = "model/platform_encoder.pkl"
    scaler_path = "model/scaler.pkl"
    brand_stats_path = "model/brand_stats.csv"

    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Missing encoder: {enc_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    enc = joblib.load(enc_path)
    scaler_obj = joblib.load(scaler_path)
    scaler = scaler_obj["scaler"]
    numeric_cols = scaler_obj["columns"]

    # --- Convert to DataFrame ---
    df = pd.DataFrame([record])

    # --- Title & brand ---
    df["title"] = df.get("title", df.get("name", "")).astype(str)
    df["brand"] = (
        df["title"]
        .str.extract(r"^([A-Za-z0-9]+)", expand=False)
        .fillna("UNKNOWN")
        .str.upper()
    )

    # --- Derived features ---
    df["rating"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(0)
    df["timestamp"] = pd.to_datetime(df.get("timestamp", datetime.now()), errors="coerce")
    df["days_since"] = (datetime.now() - df["timestamp"]).dt.days
    df["is_ultra"] = df["title"].str.contains("ultra", case=False).astype(int)
    df["is_fe"] = df["title"].str.contains(r"\bFE\b", case=False).astype(int)
    df["title_len"] = df["title"].str.len()

    # --- Platform encoding ---
    df["platform"] = df.get("platform", "Unknown").fillna("Unknown")
    platform_ohe = enc.transform(df[["platform"]])
    platform_cols = [f"platform_{c}" for c in enc.categories_[0]]
    df_platform = pd.DataFrame(platform_ohe, columns=platform_cols, index=df.index)
    df = pd.concat([df, df_platform], axis=1)

    # Ensure platform columns exist (defensive)
    for col in platform_cols:
        if col not in df.columns:
            df[col] = 0

    # --- Brand-level stats: load saved CSV and merge ---
    if os.path.exists(brand_stats_path):
        brand_stats = pd.read_csv(brand_stats_path)
        # brand_stats should have columns: brand, brand_mean, brand_std
        if "brand" not in brand_stats.columns:
            # if saved with index, reset
            brand_stats = brand_stats.reset_index().rename(columns={brand_stats.columns[0]: "brand"})
        df = df.merge(brand_stats, on="brand", how="left")
        # fallback to global mean if brand missing
        global_mean = brand_stats["brand_mean"].mean() if "brand_mean" in brand_stats else 0
        df["brand_mean"].fillna(global_mean, inplace=True)
        df["brand_std"].fillna(0, inplace=True)
    else:
        # last-resort fallback
        df["brand_mean"] = 0
        df["brand_std"] = 0

    # --- Feature selection ---
    base_features = ["rating", "days_since", "is_ultra", "is_fe", "title_len", "brand_mean", "brand_std"]
    features = base_features + platform_cols
    X = df[features].fillna(0)

    # --- Scale numeric features (use saved scaler) ---
    X[numeric_cols] = scaler.transform(X[numeric_cols])

    return X
