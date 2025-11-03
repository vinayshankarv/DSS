import os
import glob
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def extract_features(title):
    """Extract brand, series, storage (GB), RAM (GB), and 5G flag."""
    title = str(title).lower()

    # Extract brand (first word or known brand keywords)
    brands = ["samsung", "apple", "oneplus", "xiaomi", "redmi", "realme",
              "vivo", "oppo", "motorola", "iqoo", "google"]
    brand = next((b for b in brands if b in title), "other")

    storage = re.findall(r"(\d+)\s*gb", title)
    ram = re.findall(r"(\d+)\s*gb\s*ram", title)

    if "pro" in title:
        series = "pro"
    elif "plus" in title:
        series = "plus"
    elif "ultra" in title:
        series = "ultra"
    elif "max" in title:
        series = "max"
    elif "mini" in title:
        series = "mini"
    else:
        series = "standard"

    is_5g = 1 if "5g" in title else 0

    return {
        "brand": brand,
        "storage_gb": int(storage[0]) if storage else 0,
        "ram_gb": int(ram[0]) if ram else 0,
        "series": series,
        "is_5g": is_5g
    }


def train_and_save():
    """Train ML models on the latest scraped data and save results."""
    csv_files = glob.glob("outputs/scraped_results_*.csv")
    if not csv_files:
        raise FileNotFoundError("❌ No scraped_results_*.csv file found in outputs/ folder.")

    latest_file = max(csv_files, key=os.path.getctime)
    print(f"📂 Using latest data file: {latest_file}")

    df = pd.read_csv(latest_file)

    # ----------------------------
    # 1️⃣ Clean and validate price
    # ----------------------------
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )

    before_rows = len(df)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    after_rows = len(df)

    if df.empty:
        raise ValueError("❌ No valid price data after cleaning — check scraper output.")

    print(f"🧹 Cleaned price data: {before_rows - after_rows} rows dropped, {after_rows} valid entries remain.")
    print(f"💰 Price range: ₹{df['price'].min():,.0f} — ₹{df['price'].max():,.0f} | mean: ₹{df['price'].mean():,.0f}")

    # ----------------------------
    # 2️⃣ Feature extraction
    # ----------------------------
    feats = df["title"].apply(extract_features).apply(pd.Series)
    df = pd.concat([df, feats], axis=1)

    # ----------------------------
    # 3️⃣ Encode categorical features
    # ----------------------------
    df["brand_encoded"] = df["brand"].astype("category").cat.codes
    df["series_encoded"] = df["series"].astype("category").cat.codes
    df["platform_encoded"] = df["platform"].astype("category").cat.codes

    # ----------------------------
    # 4️⃣ Clean rating
    # ----------------------------
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating"] = df["rating"].fillna(df["rating"].median() if not df["rating"].isnull().all() else 0)

    # ----------------------------
    # 5️⃣ Final feature matrix
    # ----------------------------
    X = df[[
        "rating",
        "storage_gb",
        "ram_gb",
        "brand_encoded",
        "series_encoded",
        "platform_encoded",
        "is_5g"
    ]]
    y = np.log1p(df["price"])

    if y.isnull().any():
        raise ValueError("❌ NaNs found in target variable y after cleaning.")

    # ----------------------------
    # 6️⃣ Split and scale
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("model", exist_ok=True)
    joblib.dump(scaler, "model/scaler.pkl")

    # ----------------------------
    # 7️⃣ Train models
    # ----------------------------
    models = {
        "rf": RandomForestRegressor(n_estimators=400, max_depth=12, random_state=42),
        "lr": LinearRegression()
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results.append((name, rmse, r2))
        joblib.dump(model, f"model/{name}_model.pkl")
        print(f"✅ {name.upper()} model saved. RMSE={rmse:.4f}, R2={r2:.4f}")

    # ----------------------------
    # 8️⃣ Save results
    # ----------------------------
    pd.DataFrame(results, columns=["Model", "RMSE", "R2"]).to_csv(
        "model/model_performance.csv", index=False
    )
    print("📈 Model performance saved to model/model_performance.csv")
    print("🎯 Training completed successfully.")


if __name__ == "__main__":
    train_and_save()
