# ml_pipeline/tune_rf_optuna.py
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
    }
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    return -cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=3).mean()
