import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "exchange_rates.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_exchange_rates.csv"
TARGET_CURRENCIES = ["EUR", "INR", "GBP", "JPY", "AUD"]

# Load dataset
df = pd.read_csv(RAW_DATA_PATH, index_col=0)

# Convert index to datetime
df.index = pd.to_datetime(df.index)

# Sort dataset
df = df.sort_index()

# Fill missing values
df = df.ffill()

# Keep only the currencies predicted by the project.
currencies = TARGET_CURRENCIES
df = df[currencies]

feature_frames = []

# Create lag, trend, return, and volatility features
for currency in currencies:
    return1 = df[currency].pct_change(1)
    feature_frames.append(pd.DataFrame({
        f"{currency}_lag1": df[currency].shift(1),
        f"{currency}_lag2": df[currency].shift(2),
        f"{currency}_lag3": df[currency].shift(3),
        f"{currency}_lag7": df[currency].shift(7),
        f"{currency}_lag14": df[currency].shift(14),
        f"{currency}_MA7": df[currency].shift(1).rolling(7).mean(),
        f"{currency}_MA14": df[currency].shift(1).rolling(14).mean(),
        f"{currency}_return1": return1.shift(1),
        f"{currency}_return7": df[currency].pct_change(7).shift(1),
        f"{currency}_momentum7": df[currency].shift(1) - df[currency].shift(8),
        f"{currency}_volatility7": return1.rolling(7).std().shift(1),
    }, index=df.index))

df = pd.concat([df, *feature_frames], axis=1)

# Add time features
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month
df["year"] = df.index.year

# Remove missing rows
df = df.dropna()

# Save processed dataset
PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(PROCESSED_DATA_PATH)

print("Feature engineering completed successfully")
print("Dataset shape:", df.shape)
