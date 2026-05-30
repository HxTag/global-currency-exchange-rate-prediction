import requests

# Choose base currency
base = "USD"

# Choose currency you want
symbol = "INR"

# API request
url = f"https://api.frankfurter.app/latest?from={base}&to={symbol}"

response = requests.get(url, timeout=10)
response.raise_for_status()

data = response.json()

rate = data["rates"][symbol]

print(f"Current {base} to {symbol} rate:", rate)
