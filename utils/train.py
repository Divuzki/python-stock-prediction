import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt

# plot_loss() function
def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def train_model(stock_prices, stock_symbol, plot=False):
    # Step 1: Load the historical stock price data
    # Assuming you have already retrieved the historical stock prices using the get_stock_prices() function

    # Convert the stock prices into a Pandas DataFrame
    df = pd.DataFrame(stock_prices)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Step 2: Prepare the data for training
    features = ['open', 'high', 'low', 'close', 'volume']
    target = 'close'
    X = df[features].values
    y = df[target].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

    # Step 3: Define and train the LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], 1), kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)))  # Update input_shape
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    

    # Step 4: Evaluate the model
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    save_model = input("Do you want to save the model? (y/n): ")

    if save_model.lower() == 'y':
        model.save(f"models/{stock_symbol}_prediction_model.h5")
        print(f"Model saved to models/{stock_symbol}_prediction_model.h5")

    # Plot the training loss and validation loss over epochs
    if plot:
        plot_loss(history)
        print("Loss curve plotted")
 
