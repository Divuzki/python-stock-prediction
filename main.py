import requests
import json

def get_stock_prices(symbol, api_key):
    """
    Retrieves historical stock price data from Alpha Vantage API.

    Args:
        symbol (str): Ticker symbol of the stock (e.g., 'AAPL' for Apple Inc.).
        api_key (str): Your Alpha Vantage API key.

    Returns:
        list: A list of dictionaries containing the historical stock price data.
    """

    base_url = 'https://www.alphavantage.co/query'
    function = 'TIME_SERIES_DAILY'  # Adjust if you need a different time series function
    output_size = 'full'  # Adjust if you want a partial output size

    # Construct the API request URL
    url = f"{base_url}?function={function}&symbol={symbol}&outputsize={output_size}&apikey={api_key}"

    try:
        response = requests.get(url)
        data = json.loads(response.text)

        if 'Time Series (Daily)' in data:
            time_series = data['Time Series (Daily)']
            stock_prices = []

            for date, price in time_series.items():
                stock_prices.append({
                    'date': date,
                    'open': float(price['1. open']),
                    'high': float(price['2. high']),
                    'low': float(price['3. low']),
                    'close': float(price['4. close']),
                    'volume': int(price['5. volume'])
                })

            return stock_prices

        elif 'Error Message' in data:
            print(f"Error: {data['Error Message']}")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

    return []

# Example usage:
api_key = 'YOUR_API_KEY'  # Replace with your Alpha Vantage API key
symbol = 'AAPL'  # Replace with the desired stock ticker symbol

stock_prices = get_stock_prices(symbol, api_key)
for price in stock_prices:
    print(price)
