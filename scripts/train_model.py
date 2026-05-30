import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from xgboost import XGBRegressor

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_exchange_rates.csv"
MODELS_DIR = ROOT_DIR / "models"
METRICS_PATH = MODELS_DIR / "model_metrics.csv"

# Load dataset
df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0)

# Define currencies to predict
currencies = ["EUR", "INR", "GBP", "JPY", "AUD"]

# Create model folder
MODELS_DIR.mkdir(parents=True, exist_ok=True)
metrics_rows = []

for currency in currencies:

    print(f"\nTraining model for {currency}")

    y = df[currency]

    feature_columns = [
        f"{currency}_lag1",
        f"{currency}_lag2",
        f"{currency}_lag3",
        f"{currency}_lag7",
        f"{currency}_lag14",
        f"{currency}_MA7",
        f"{currency}_MA14",
        f"{currency}_return1",
        f"{currency}_return7",
        f"{currency}_momentum7",
        f"{currency}_volatility7",
        "day_of_week",
        "month",
        "year",
    ]
    X = df[feature_columns]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    baseline_model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )

    main_model = XGBRegressor(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )

    baseline_model.fit(X_train, y_train)
    baseline_predictions = baseline_model.predict(X_test)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    baseline_mape = np.mean(np.abs((y_test.values - baseline_predictions) / y_test.values)) * 100

    print(
        f"RandomForest baseline MAE: {baseline_mae:.6f} | "
        f"RMSE: {baseline_rmse:.6f} | MAPE: {baseline_mape:.4f}%"
    )

    main_model.fit(X_train, y_train)
    main_predictions = main_model.predict(X_test)
    main_rmse = np.sqrt(mean_squared_error(y_test, main_predictions))
    main_mae = mean_absolute_error(y_test, main_predictions)
    main_mape = np.mean(np.abs((y_test.values - main_predictions) / y_test.values)) * 100

    print(
        f"XGBoost main MAE: {main_mae:.6f} | "
        f"RMSE: {main_rmse:.6f} | MAPE: {main_mape:.4f}%"
    )

    metrics_rows.extend([
        {
            "Currency": currency,
            "Model": "XGBoost (Main)",
            "MAE": main_mae,
            "RMSE": main_rmse,
            "MAPE (%)": main_mape,
        },
        {
            "Currency": currency,
            "Model": "Random Forest (Baseline)",
            "MAE": baseline_mae,
            "RMSE": baseline_rmse,
            "MAPE (%)": baseline_mape,
        },
    ])

    final_main_model = clone(main_model)
    final_baseline_model = clone(baseline_model)

    final_main_model.fit(X, y)
    final_baseline_model.fit(X, y)

    joblib.dump(final_main_model, MODELS_DIR / f"{currency}_main_model.pkl")
    joblib.dump(final_baseline_model, MODELS_DIR / f"{currency}_rf_baseline.pkl")

pd.DataFrame(metrics_rows).to_csv(METRICS_PATH, index=False)

print(f"\nModel metrics saved to {METRICS_PATH}")
print("All currency models trained successfully!")
