import requests

SUPPORTED_CURRENCIES = {"USD", "EUR", "INR", "GBP", "JPY", "AUD"}
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
    return f"{amount}{CURRENCY_SYMBOLS[currency_code]} {CURRENCY_NAMES[currency_code]}"

# Ask user for input
amount = float(input("Enter amount: "))
from_currency = input("From currency (USD, EUR, INR, GBP, JPY, AUD): ").upper()
to_currency = input("To currency (USD, EUR, INR, GBP, JPY, AUD): ").upper()

if from_currency not in SUPPORTED_CURRENCIES or to_currency not in SUPPORTED_CURRENCIES:
    supported = ", ".join(sorted(SUPPORTED_CURRENCIES))
    raise ValueError(f"Currency must be one of: {supported}")

if from_currency == to_currency:
    print(
        f"\n{format_conversion_value(amount, from_currency)} ="
        f"{format_conversion_value(amount, to_currency)}"
    )
    raise SystemExit

# API request
url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_currency}&to={to_currency}"
response = requests.get(url, timeout=10)
response.raise_for_status()

data = response.json()

converted = data["rates"][to_currency]

print(
    f"\n{format_conversion_value(amount, from_currency)} = "
    f"{format_conversion_value(converted, to_currency)}"
)
