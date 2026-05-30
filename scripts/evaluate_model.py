import pandas as pd
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT_DIR / "models" / "model_metrics.csv"

results = pd.read_csv(METRICS_PATH)
print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
