import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2


def train_model(stock_prices, stock_symbol=None):
    # Step 1: Load the historical stock price data
    # Assuming you have already retrieved the historical stock prices using the get_stock_prices() function

    # Convert the stock prices into a Pandas DataFrame
    df = pd.DataFrame(stock_prices)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Step 2: Prepare the data for training
    # Select the features and target variable
    features = ['open', 'high', 'low', 'close', 'volume']
    target = 'close'

    # Extract the features and target variable from the DataFrame
    X = df[features].values
    y = df[target].values

    # Scale the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

    # Step 3: Define and train the LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # Step 4: Evaluate the model
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    model.save(f"models/{stock_symbol}.h5")
