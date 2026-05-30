import pandas as pd
import joblib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_exchange_rates.csv"
MODELS_DIR = ROOT_DIR / "models"

# Ask user which currency to predict
currency = input("Enter currency (EUR, INR, GBP, JPY, AUD): ").upper()

# Load dataset
df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0)

# Load corresponding model
model = joblib.load(MODELS_DIR / f"{currency}_main_model.pkl")

# Get feature names
features = model.feature_names_in_

# Number of future days
future_days = int(input("Enter number of forecast days: "))

predictions = []
history = df[currency].tail(30).tolist()
forecast_date = pd.to_datetime(df.index[-1])

for i in range(future_days):

    forecast_date = forecast_date + pd.offsets.BDay(1)
    series = pd.Series(history)
    returns = series.pct_change()

    current_row = pd.DataFrame([{
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

    pred = model.predict(current_row[features])[0]

    predictions.append({
        "date": forecast_date.strftime("%Y-%m-%d"),
        "weekday": forecast_date.day_name(),
        "prediction": pred,
    })
    history.append(pred)

print("\nFuture Forecast:")

for i, item in enumerate(predictions):
    print(f"{item['date']} ({item['weekday']}): {item['prediction']}")
