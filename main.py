import os

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

from .extract import get_stock_prices

# Example usage:
api_key = os.environ.get("API_KEY")  # Replace with your Alpha Vantage API key
symbol = input("Enter Stock Symbol:")  # Replace with the desired stock ticker symbol
stock_prices = get_stock_prices(symbol, api_key)
print(
    f"Retrieved {len(stock_prices)} stock prices for {symbol}."
    if (len(stock_prices) > 0)
    else f"Could not retrieve stock prices for {symbol}."
)
