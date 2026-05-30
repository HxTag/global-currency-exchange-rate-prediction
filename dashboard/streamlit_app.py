from pathlib import Path
from datetime import datetime, timezone
import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_exchange_rates.csv"
MODELS_DIR = ROOT_DIR / "models"
METRICS_PATH = MODELS_DIR / "model_metrics.csv"
FORECAST_HISTORY_DAYS = 30

CURRENCIES = {
    "🇺🇸 United States - US Dollar (USD)": "USD",
    "🇪🇺 European Union - Euro (EUR)": "EUR",
    "🇮🇳 India - Indian Rupee (INR)": "INR",
    "🇬🇧 United Kingdom - Pound Sterling (GBP)": "GBP",
    "🇯🇵 Japan - Japanese Yen (JPY)": "JPY",
    "🇦🇺 Australia - Australian Dollar (AUD)": "AUD",
}

CURRENCY_LABELS = {code: label for label, code in CURRENCIES.items()}
CURRENCY_NAMES = {
    "USD": "US Dollar",
    "EUR": "Euro",
    "INR": "Indian Rupee",
    "GBP": "Pound Sterling",
    "JPY": "Japanese Yen",
    "AUD": "Australian Dollar",
}
CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "€",
    "INR": "₹",
    "GBP": "£",
    "JPY": "¥",
    "AUD": "A$",
}


def format_conversion_value(amount, currency_code):
    symbol = CURRENCY_SYMBOLS[currency_code].replace("$", r"\$")
    return f"{amount}{symbol} {CURRENCY_NAMES[currency_code]}"


def fetch_rate(from_currency, to_currency, amount=None):
    if from_currency == to_currency:
        return {
            "rate": 1.0,
            "converted_amount": amount if amount is not None else 1.0,
            "provider": "Same currency",
            "updated_at": "Now",
        }

    fixer_api_key = os.getenv("FIXER_API_KEY")
    if fixer_api_key:
        params = (
            f"access_key={fixer_api_key}"
            f"&from={from_currency}"
            f"&to={to_currency}"
            f"&amount={amount if amount is not None else 1}"
        )

        try:
            response = requests.get(
                f"https://data.fixer.io/api/convert?{params}",
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            data = {}

        if data.get("success"):
            rate = data["info"]["rate"]
            timestamp = data.get("info", {}).get("timestamp")
            updated_at = "Latest available"
            if timestamp:
                updated_at = datetime.fromtimestamp(
                    timestamp,
                    tz=timezone.utc,
                ).strftime("%Y-%m-%d %H:%M:%S UTC")

            return {
                "rate": rate,
                "converted_amount": data["result"],
                "provider": "Fixer",
                "updated_at": updated_at,
            }

    exchange_rate_api_key = os.getenv("EXCHANGERATE_API_KEY")
    if exchange_rate_api_key:
        url = (
            f"https://v6.exchangerate-api.com/v6/{exchange_rate_api_key}"
            f"/pair/{from_currency}/{to_currency}"
        )
        if amount is not None:
            url = f"{url}/{amount}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            data = {}

        if data.get("result") == "success":
            rate = data["conversion_rate"]
            return {
                "rate": rate,
                "converted_amount": data.get(
                    "conversion_result",
                    amount * rate if amount is not None else rate,
                ),
                "provider": "ExchangeRate-API",
                "updated_at": data.get("time_last_update_utc", "Latest available"),
            }

    params = f"from={from_currency}&to={to_currency}"
    if amount is not None:
        params = f"amount={amount}&{params}"

    try:
        response = requests.get(
            f"https://api.frankfurter.app/latest?{params}",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException:
        return None

    rate = data.get("rates", {}).get(to_currency)
    if rate is None:
        return None

    return {
        "rate": rate if amount is None else rate / amount,
        "converted_amount": rate,
        "provider": "Frankfurter",
        "updated_at": data.get("date", "Latest working day"),
    }


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


def model_display_name(model):
    class_name = model.__class__.__name__
    names = {
        "XGBRegressor": "XGBoost",
        "RandomForestRegressor": "Random Forest",
    }
    return names.get(class_name, class_name)


def format_error_value(value):
    if abs(value) < 0.01:
        return f"{value:.6f}"
    if abs(value) < 1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def forecast_tree_model(model, df, currency, forecast_days):
    features = model.feature_names_in_
    history = df[currency].tail(FORECAST_HISTORY_DAYS).tolist()
    forecast_date = pd.to_datetime(df.index[-1])
    predictions = []

    for _ in range(forecast_days):
        forecast_date = forecast_date + pd.offsets.BDay(1)
        current_row = build_forecast_features(
            currency,
            history,
            forecast_date,
            features,
        )
        prediction = model.predict(current_row)[0]
        predictions.append({
            "date": forecast_date,
            "weekday": forecast_date.day_name(),
            "prediction": prediction,
        })
        history.append(prediction)

    return predictions


st.set_page_config(
    page_title="Global Currency Analytics Dashboard",
    page_icon="$",
    layout="wide",
)

st.title("Global Currency Analytics Dashboard")

tab1, tab2 = st.tabs(["Currency Prediction", "Currency Converter"])

with tab1:
    st.header("Currency Forecast")

    currency_label = st.selectbox(
        "Select Currency",
        list(CURRENCIES.keys())[1:],
    )
    currency = CURRENCIES[currency_label]
    forecast_days = st.slider("Forecast Days", 1, 30, 7)

    df = pd.read_csv(DATA_PATH, index_col=0)

    live_rate = fetch_rate("USD", currency)
    st.subheader("Latest Exchange Rate")

    if live_rate is not None:
        st.metric(f"{CURRENCY_LABELS['USD']} to {CURRENCY_LABELS[currency]}", live_rate["rate"])
        st.caption(f"Source: {live_rate['provider']} | Updated: {live_rate['updated_at']}")
    else:
        st.warning("Live rate is unavailable right now. Forecasting still works with saved data.")

    st.subheader("Historical Trend")
    st.line_chart(df[currency])

    if st.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            main_model = joblib.load(MODELS_DIR / f"{currency}_main_model.pkl")
            rf_model = joblib.load(MODELS_DIR / f"{currency}_rf_baseline.pkl")
            main_model_name = model_display_name(main_model)

            main_predictions = forecast_tree_model(
                main_model,
                df,
                currency,
                forecast_days,
            )
            rf_predictions = forecast_tree_model(
                rf_model,
                df,
                currency,
                forecast_days,
            )

            st.subheader("Rate Comparison")
            col1, col2, col3 = st.columns(3)
            col1.metric("Latest", live_rate["rate"] if live_rate is not None else "N/A")
            col2.metric(main_model_name, round(main_predictions[0]["prediction"], 4))
            col3.metric("Random Forest", round(rf_predictions[0]["prediction"], 4))

            st.subheader("Model Performance")
            metrics_df = pd.read_csv(METRICS_PATH)
            metrics_df = metrics_df[metrics_df["Currency"] == currency].copy()
            metrics_df = pd.DataFrame({
                "Model": metrics_df["Model"],
                "Average Error (MAE)": metrics_df["MAE"].map(format_error_value),
                "RMSE": metrics_df["RMSE"].map(format_error_value),
                "Error (%)": metrics_df["MAPE (%)"].map(lambda value: f"{value:.3f}%"),
            }).reset_index(drop=True)
            st.dataframe(
                metrics_df,
                use_container_width=True,
                hide_index=True,
            )

            forecast_df = pd.DataFrame({
                "Date": [item["date"].strftime("%Y-%m-%d") for item in main_predictions],
                f"{main_model_name} (Main)": [item["prediction"] for item in main_predictions],
                "Random Forest (Baseline)": [item["prediction"] for item in rf_predictions],
            })

            st.subheader("Forecast Results")
            st.dataframe(forecast_df)

            st.download_button(
                "Download CSV",
                forecast_df.to_csv(index=False),
                file_name=f"{currency}_forecast.csv",
            )

            st.subheader("Forecast Visualization")
            historical_data = df[currency].tail(FORECAST_HISTORY_DAYS)
            forecast_dates = [
                historical_data.index[-1],
                *[item["date"].strftime("%Y-%m-%d") for item in main_predictions],
            ]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data,
                name="Historical",
                line=dict(color="blue"),
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=[historical_data.iloc[-1], *[item["prediction"] for item in main_predictions]],
                name=f"{main_model_name} (Main)",
                line=dict(color="red"),
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=[historical_data.iloc[-1], *[item["prediction"] for item in rf_predictions]],
                name="Random Forest (Baseline)",
                line=dict(color="orange"),
            ))
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Currency Converter")

    amount = st.number_input("Amount", value=1.0)
    from_label = st.selectbox("From Currency", list(CURRENCIES.keys()))
    to_label = st.selectbox("To Currency", list(CURRENCIES.keys()))

    from_currency = CURRENCIES[from_label]
    to_currency = CURRENCIES[to_label]

    if st.button("Convert"):
        conversion = fetch_rate(from_currency, to_currency, amount)

        if conversion is not None:
            st.success(
                f"{format_conversion_value(amount, from_currency)} = "
                f"{format_conversion_value(conversion['converted_amount'], to_currency)}"
            )
            st.caption(
                f"Source: {conversion['provider']} | "
                f"Updated: {conversion['updated_at']}"
            )
        else:
            st.error("Conversion failed")
