import os
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.



api_key = os.environ.get("API_KEY")  # Alpha Vantage API key

# Load the trained model
model = load_model("merged_model.h5")

# Function to fetch stock data using Alpha Vantage API
def fetch_stock_data(symbol):
    api_key = "YOUR_API_KEY"  # Replace with your Alpha Vantage API key
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='compact')
    return data

# Function to preprocess the stock data
def preprocess_stock_data(data):
    # Add your preprocessing steps here
    # e.g., scaling, normalization, feature engineering, etc.
    return data

# Function to extract input features from stock data
def extract_input_features(data):
    # Add your feature extraction steps here
    # e.g., selecting relevant columns, reshaping data, etc.
    return data

# Get stock symbol input from the user
symbol = input("Enter the stock symbol: ")

# Fetch stock data using Alpha Vantage API
stock_data = fetch_stock_data(symbol)

# Preprocess the stock data
preprocessed_data = preprocess_stock_data(stock_data)

# Extract input features from preprocessed data
input_features = extract_input_features(preprocessed_data)

# Perform the prediction
predicted_prices = model.predict(input_features)

def process_predicted_prices(predicted_prices):
    # Process the predicted prices here
    # You can implement any further analysis or decision-making steps

    # Example: Convert the predicted prices to a pandas DataFrame
    df_predicted = pd.DataFrame(predicted_prices, columns=["Predicted Price"])

    # Print the predicted prices
    print(df_predicted)

    # You can perform additional actions such as saving the predictions to a file,
    # making trading decisions based on certain conditions, or visualizing the predictions.

    # Placeholder for any further processing or decision-making steps
    # Add your code here

    return


# Process the predicted prices or make further decisions based on the prediction
process_predicted_prices(predicted_prices)  # Placeholder for your further processing steps

