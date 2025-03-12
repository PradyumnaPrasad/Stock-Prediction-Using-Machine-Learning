import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras_tuner import HyperModel, RandomSearch
import streamlit as st
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2


np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Fetch stock data
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime.now()
user_input = st.text_input('Enter the stock symbol:', 'AAPL')  
df = yf.download(user_input, start=start, end=end)

df = df.reset_index(drop=False)



# Check if data is fetched successfully

if df.empty:
    st.error(f"No data found for the symbol: {user_input}. Please check the symbol and try again.")
    st.stop()  

# Prepare the data
df['Date'] = df.index
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Include additional features
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

# Calculate RSI
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)  
loss = -delta.where(delta < 0, 0)  

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Drop rows with NaN values
df.dropna(inplace=True)

# Select features
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Day', 'Month', 'Year', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI']
data = df[features].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create training and testing datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 10:]

# Create datasets for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), :]
        X.append(a)
        Y.append(data[i + time_step, 1:3])  
    return np.array(X), np.array(Y)

time_step = 10
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Define a HyperModel for KerasTuner
class LSTMHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(LSTM(
            units=hp.Int('units', min_value=64, max_value=256, step=64),  # Reduced max_value
            return_sequences=True,
            input_shape=(time_step, X_train.shape[2]),
            kernel_regularizer=l2(0.01)  # Added L2 regularization
        ))

        model.add(Dropout(0.2))  
        model.add(LSTM(
            units=hp.Int('units', min_value=64, max_value=512, step=64),  # Increased range
            return_sequences=True
        ))
        model.add(Dropout(0.2))  
        model.add(LSTM(units=hp.Int('units', min_value=64, max_value=512, step=64),  # Increased range
            return_sequences=False))

        model.add(Dropout(0.3))  # Increased dropout rate
        model.add(LSTM(
            units=hp.Int('units', min_value=64, max_value=256, step=64),  # Reduced max_value
            return_sequences=False,
            kernel_regularizer=l2(0.01)))  # Added L2 regularization
        model.add(Dropout(0.3))  # Increased dropout rate
        model.add(Dense(25))
        model.add(Dense(2))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            ),
            loss='mean_absolute_error'
        )
        return model

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

hypermodel = LSTMHyperModel()


model_file = 'model.h5'
if os.path.exists(model_file):
    # Load the existing model
    best_model = load_model(model_file)
else:
    # hyperparameter tuning
    tuner = RandomSearch(
        hypermodel,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=2,
        directory='hyperparameter_tuning',
        project_name='lstm_stock_prediction'
    )

    tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    best_model = tuner.get_best_models(num_models=1)[0]


    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping])  # Increased epochs

    # Fit the best model with early stopping and learning rate scheduler
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Increased patience
    lr_scheduler_callback = LearningRateScheduler(lr_scheduler)
    best_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping, lr_scheduler_callback])


    # Save the trained model
    best_model.save(model_file)

# Predicting the model
y_predicted = best_model.predict(X_test)
y_test = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 1)), y_test, np.zeros((y_test.shape[0], data.shape[1] - 3))), axis=1))[:, 1:3]
y_predicted = scaler.inverse_transform(np.concatenate((np.zeros((y_predicted.shape[0], 1)), y_predicted, np.zeros((y_predicted.shape[0], data.shape[1] - 3))), axis=1))[:, 1:3]

# Calculate the offset
last_original_value = y_test[-1]
last_predicted_value = y_predicted[-1]
offset = last_original_value - last_predicted_value

# Prepare the input data for the next prediction using the last 10 days
last_10_days = scaled_data[-10:]
last_10_days = np.array(last_10_days).reshape(1, -1, X_train.shape[2])

# Predict the next time step (tomorrow's 'High', 'Low' prices)
predicted_prices = best_model.predict(last_10_days)
predicted_prices = scaler.inverse_transform(np.concatenate((np.zeros((predicted_prices.shape[0], 1)), predicted_prices, np.zeros((predicted_prices.shape[0], data.shape[1] - 3))), axis=1))[:, 1:3]

# Adjust the predicted prices by adding the offset
adjusted_predicted_prices = predicted_prices + offset

# Predict tomorrow's opening price using Ridge regression
df['Prev Close'] = df['Close'].shift(1)
df.dropna(inplace=True)
X_open = df[['Prev Close']]
y_open = df['Open']

# Split the data into training and test sets
X_train_open, X_test_open, y_train_open, y_test_open = train_test_split(X_open, y_open, test_size=0.2, random_state=42)

# Scale the features for the Ridge regression model
scaler_open = StandardScaler()
X_train_open_scaled = scaler_open.fit_transform(X_train_open)
X_test_open_scaled = scaler_open.transform(X_test_open)

# Train the Ridge regression model
model_open = Ridge(alpha=1.0)
model_open.fit(X_train_open_scaled, y_train_open)

# Predict the opening value for tomorrow
last_close_value = df[['Prev Close']].iloc[-1].values.reshape(-1, 1)
last_close_value_scaled = scaler_open.transform(last_close_value)
predicted_opening_value = model_open.predict(last_close_value_scaled)


prev_close_value = df['Close'].iloc[-1]
min_open = prev_close_value * 1.005
max_open = prev_close_value * 1.01
predicted_opening_value = np.clip(predicted_opening_value, min_open, max_open)

# Insert the predicted opening value into adjusted_predicted_prices
adjusted_predicted_prices = np.insert(adjusted_predicted_prices, 0, predicted_opening_value, axis=1)

# Calculate the predicted closing price as the average of predicted high and low prices
predicted_closing_value = (adjusted_predicted_prices[0][1] + adjusted_predicted_prices[0][2]) / 2

# Ensure open value is not lower than predicted low value
if adjusted_predicted_prices[0][0] < adjusted_predicted_prices[0][2]:
    adjusted_predicted_prices[0][0] = adjusted_predicted_prices[0][2]
if adjusted_predicted_prices[0][0] > adjusted_predicted_prices[0][1]:
    adjusted_predicted_prices[0][0] = adjusted_predicted_prices[0][1]

# Display today's high and low
todays_high = df['High'].iloc[-1]
todays_low = df['Low'].iloc[-1]
st.subheader('Today\'s Prices')
st.write(f"High: {todays_high}")
st.write(f"Low: {todays_low}")

# Display the predicted prices
st.subheader('Tomorrow\'s Predicted Prices')
st.write(f"Open: {adjusted_predicted_prices[0][0]}")
st.write(f"High: {adjusted_predicted_prices[0][1]}")
st.write(f"Low: {adjusted_predicted_prices[0][2]}")
st.write(f"Close: {predicted_closing_value}")  

# Plotting the prediction vs original for 'High', 'Low', and 'Close' prices
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test[:, 0], 'g', label='Original High Price')
plt.plot(y_predicted[:, 0], 'orange', label='Predicted High Price')
plt.plot(y_test[:, 1], 'purple', label='Original Low Price')
plt.plot(y_predicted[:, 1], 'pink', label='Predicted Low Price')
plt.axhline(y=predicted_closing_value, color='red', linestyle='--', label='Predicted Closing Price')  
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Prepare data for plotting tomorrow's prices
last_known_data = np.array(data[-10:])
# Ensure predicted_data has the same number of columns as last_known_data
predicted_data = np.zeros_like(last_known_data[0])
predicted_data[0] = adjusted_predicted_prices[0][0]  # Predicted open price
predicted_data[1] = adjusted_predicted_prices[0][1]  # Predicted high price
predicted_data[2] = adjusted_predicted_prices[0][2]  # Predicted low price
predicted_data[3] = predicted_closing_value  # Predicted closing price

# Concatenate known and predicted data for plotting
plot_data = np.vstack((last_known_data, predicted_data.reshape(1, -1)))

# Plotting the predicted prices for tomorrow
st.subheader('Graph of Predicted Prices for Tomorrow')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(plot_data[:, 0], 'b', label='Predicted Open Price for Tomorrow')
plt.plot(plot_data[:, 1], 'g', label='Predicted High Price for Tomorrow')
plt.plot(plot_data[:, 2], 'purple', label='Predicted Low Price for Tomorrow')
plt.axhline(y=predicted_closing_value, color='red', linestyle='--', label='Predicted Closing Price for Tomorrow')  # Add predicted closing price line
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)
