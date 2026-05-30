from pathlib import Path

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd


app = FastAPI()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_exchange_rates.csv"
MODELS_DIR = ROOT_DIR / "models"
CURRENCIES = {"EUR", "INR", "GBP", "JPY", "AUD"}


def build_forecast_features(currency, history, forecast_date, feature_columns):
    series = pd.Series(history)
    returns = series.pct_change()

    row = pd.DataFrame([{
        f"{currency}_lag1": history[-1],
        f"{currency}_lag2": history[-2],
        f"{currency}_lag3": history[-3],
        f"{currency}_lag7": history[-7],
        f"{currency}_lag14": history[-14],
        f"{currency}_MA7": series.tail(7).mean(),
        f"{currency}_MA14": series.tail(14).mean(),
        f"{currency}_return1": returns.iloc[-1],
        f"{currency}_return7": history[-1] / history[-8] - 1,
        f"{currency}_momentum7": history[-1] - history[-8],
        f"{currency}_volatility7": returns.tail(7).std(),
        "day_of_week": forecast_date.dayofweek,
        "month": forecast_date.month,
        "year": forecast_date.year,
    }])

    return row[feature_columns]


@app.get("/")
def home():
    return {
        "message": "Currency Prediction API",
        "currencies": sorted(CURRENCIES),
    }


@app.get("/predict/{currency}")
def predict(currency: str, days: int = 1):
    currency = currency.upper()

    if currency not in CURRENCIES:
        raise HTTPException(
            status_code=400,
            detail=f"Currency must be one of: {', '.join(sorted(CURRENCIES))}",
        )

    if days < 1 or days > 30:
        raise HTTPException(
            status_code=400,
            detail="days must be between 1 and 30",
        )

    model = joblib.load(MODELS_DIR / f"{currency}_main_model.pkl")
    df = pd.read_csv(DATA_PATH, index_col=0)
    history = df[currency].tail(30).tolist()
    forecast_date = pd.to_datetime(df.index[-1])
    predictions = []

    for day in range(1, days + 1):
        forecast_date = forecast_date + pd.offsets.BDay(1)
        features = build_forecast_features(
            currency,
            history,
            forecast_date,
            model.feature_names_in_,
        )
        prediction = float(model.predict(features)[0])
        predictions.append({
            "forecast_number": day,
            "date": forecast_date.strftime("%Y-%m-%d"),
            "weekday": forecast_date.day_name(),
            "predicted_exchange_rate": prediction,
        })
        history.append(prediction)

    return {
        "currency": currency,
        "model": "XGBoost",
        "predictions": predictions,
    }
