import requests
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "exchange_rates.csv"

url = "https://api.frankfurter.app/2018-01-01..?from=USD"

response = requests.get(url, timeout=20)
response.raise_for_status()
data = response.json()

rates = data["rates"]

records = []

for date, value in rates.items():
    row = {"Date": date}
    row.update(value)
    records.append(row)

df = pd.DataFrame(records)

df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)

RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(RAW_DATA_PATH, index=False)

print("Dataset updated till latest available date!")
