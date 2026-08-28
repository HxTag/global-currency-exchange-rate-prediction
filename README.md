# Currency Exchange Rate Prediction

This project collects historical exchange rates, builds lag-based features, trains an XGBoost forecasting model with a Random Forest baseline, and serves predictions through Streamlit and FastAPI.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Data And Training

Run these from the project root:

```powershell
python scripts\data_collection.py
python scripts\preprocessing.py
python scripts\train_model.py
```

`train_model.py` trains XGBoost as the main model and Random Forest as the baseline for each forecast currency. Both models use only that currency's lag, moving average, return, momentum, volatility, and date features. This avoids relying on same-day values for other currencies that are not known during future forecasting.

The dashboard evaluates models with MAE, RMSE, and MAPE.

## Run The Apps

```powershell
streamlit run dashboard\streamlit_app.py
```

For the API:

```powershell
uvicorn api.app:app --reload
```

API docs are available at `http://127.0.0.1:8000/docs`. Example prediction endpoint:

```text
http://127.0.0.1:8000/predict/INR?days=7
```

## Notes

- Live exchange rates are fetched from Frankfurter. If the API is unavailable, the dashboard still shows forecasts from saved data.
- Forecasts are educational and should not be treated as financial advice.

## License

This project is licensed under the [MIT License](LICENSE).
