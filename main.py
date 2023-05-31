import os

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

from utils.extract import get_stock_prices


api_key = os.environ.get("API_KEY")  # Replace with your Alpha Vantage API key
symbol = input("Enter Stock Symbol:")  # Replace with the desired stock ticker symbol
isTraining = input("Is this for training? (y/n): ")

if isTraining.lower() == "y":
    from utils.train import train_model
    plot = input("Plot the model? (y/n): ")
    if plot.lower() == "y":
        train_model(get_stock_prices(symbol, api_key), symbol, True)
    else:
        train_model(get_stock_prices(symbol, api_key), symbol)
else:
    stock_prices = get_stock_prices(symbol, api_key)
    print(
        f"Retrieved {len(stock_prices)} stock prices for {symbol}."
        if (len(stock_prices) > 0)
        else f"Could not retrieve stock prices for {symbol}."
    )
