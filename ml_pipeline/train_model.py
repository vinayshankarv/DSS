# ml_pipeline/train_model.py
# --- THIS IS THE CORRECTED FILE ---

import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. IMPORT the correct preprocessing function
from ml_pipeline.preprocessing import clean_and_feature_engineer

def train_and_save():
    """
    Train ML models using the unified preprocessing pipeline.
    """
    csv_files = glob.glob("outputs/scraped_results_*.csv")
    if not csv_files:
        raise FileNotFoundError("❌ No scraped_results_*.csv file found in outputs/ folder.")

    latest_file = max(csv_files, key=os.path.getctime)
    print(f"📂 Using latest data file: {latest_file}")
    
    # 2. CALL the unified function
    # This function now does ALL the cleaning, feature engineering,
    # scaling, and saving of scalers/encoders.
    try:
        X_train, X_test, y_train, y_test = clean_and_feature_engineer(latest_file)
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return

    print("✅ Preprocessing and train-test split complete.")

    # 3. DEFINE the models
    models = {
        "rf": RandomForestRegressor(n_estimators=100, random_state=42),
        "lr": LinearRegression()
    }

    results = []
    
    # Check if data is empty after split
    if X_train.empty or y_train.empty:
        print("❌ Training data is empty after preprocessing. Cannot train models.")
        return

    # 4. TRAIN models on the correctly processed data
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            
            # Ensure X_test has the same columns as X_train
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
            
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            results.append((name, rmse, r2))
            joblib.dump(model, f"model/{name}_model.pkl")
            print(f"✅ {name.upper()} model saved. RMSE={rmse:.4f}, R2={r2:.4f}")
        
        except Exception as e:
             print(f"❌ Error training model {name}: {e}")

    # 5. SAVE results
    pd.DataFrame(results, columns=["Model", "RMSE", "R2"]).to_csv(
        "model/model_performance.csv", index=False
    )
    print("📈 Model performance saved to model/model_performance.csv")
    print("🎯 Training completed successfully.")


if __name__ == "__main__":
    train_and_save()