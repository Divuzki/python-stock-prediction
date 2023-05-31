import os
import numpy as np
import pandas as pd


from utils.extract import get_stock_prices

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


api_key = os.environ.get("API_KEY")  # Alpha Vantage API key
symbol = input("Enter Stock Symbol:")  # desired stock ticker symbol
symbol = symbol.upper()
isTraining = input("Is this for training? (y/n): ")

if isTraining.lower() == "y":
    from utils.train import train_model

    plot = input("Plot the model? (y/n): ")
    if plot.lower() == "y":
        train_model(get_stock_prices(symbol, api_key), symbol, True)
    else:
        train_model(get_stock_prices(symbol, api_key), symbol)
else:
    from tensorflow.keras.models import load_model

    stock_prices = get_stock_prices(symbol, api_key)
    model = load_model("merged_model.h5")
    predicted_prices = model.predict(pd.DataFrame(stock_prices[:-1]).to_numpy().astype(np.float32), batch_size=1)
    predicted_prices = predicted_prices.reshape(-1, 1)
    df_predicted = pd.DataFrame(predicted_prices, columns=["Predicted Price"])

    # Print the predicted prices
    print(df_predicted)
    print(
        f"Retrieved {len(stock_prices)} stock prices for {symbol}."
        if (len(stock_prices) > 0)
        else f"Could not retrieve stock prices for {symbol}."
    )
