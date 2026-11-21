import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras_tuner import HyperModel, RandomSearch
import streamlit as st
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os
from html import escape
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.regularizers import l2
import time
from datetime import timedelta

st.set_page_config(page_title='StockScope | Price Forecasting', page_icon='📈', layout='wide')
st.markdown('''<style>
.stApp {background:linear-gradient(135deg,#f0f5ff 0%,#faf7ff 45%,#f0faf8 100%);color:#172b4d;}
.block-container {max-width:1280px;padding-top:4rem;padding-bottom:3rem;}
h1,h2,h3 {color:#152b49;letter-spacing:-.025em;}
h1 {font-size:2.8rem !important;}
[data-testid="stMetric"] {background:white;border:1px solid #e0e7f1;border-radius:14px;padding:20px 22px;box-shadow:0 3px 12px #17335805;}
[data-testid="stMetricLabel"] {color:#62738d;}
[data-testid="stMetricValue"] {font-size:1.85rem;letter-spacing:-.025em;}
[data-testid="stColumn"]:nth-child(4n+1) [data-testid="stMetric"] {background:#eef2ff;border-top:4px solid #6366f1;}
[data-testid="stColumn"]:nth-child(4n+2) [data-testid="stMetric"] {background:#e9faf5;border-top:4px solid #0d9488;}
[data-testid="stColumn"]:nth-child(4n+3) [data-testid="stMetric"] {background:#fff6e7;border-top:4px solid #eaa12c;}
[data-testid="stColumn"]:nth-child(4n+4) [data-testid="stMetric"] {background:#fff0f5;border-top:4px solid #db5b91;}
[data-testid="stSidebar"] {background:#fff;border-right:1px solid #e0e7f1;}
[data-testid="stPlotlyChart"] {background:white;border:1px solid #e0e7f1;border-radius:16px;padding:12px;}
.eyebrow {display:inline-block;font-size:11px;font-weight:700;letter-spacing:.14em;color:#6350b5;background:#ece7ff;border:1px solid #ddd4ff;border-radius:24px;padding:7px 14px;margin-bottom:10px;}
.intro {font-size:16px;color:#62738d;margin-bottom:22px;}
.session-bar {background:linear-gradient(110deg,#e2eaff,#ede6ff,#ddf5ec);border:1px solid #dce5fd;border-radius:12px;padding:16px 20px;color:#36558d;margin:10px 0 26px;}
.section-label {font-size:11px;letter-spacing:.14em;color:#6350b5;font-weight:700;margin-top:24px;border-left:3px solid #8b5cf6;padding-left:10px;}
</style>''', unsafe_allow_html=True)
with st.sidebar:
    st.markdown('### 📈 StockScope')
    st.caption('STOCK PRICE FORECASTING')
    st.divider()
    with st.form('symbol_selection'):
        user_input = st.text_input('Stock symbol', 'AAPL', help='Yahoo Finance symbol, e.g. AAPL, MSFT or INFY.NS').strip().upper()
        st.form_submit_button('Analyze stock', use_container_width=True, type='primary')
    st.caption('Examples: AAPL · MSFT · GOOGL')
    st.markdown('**Chart window**')
    chart_sessions = st.select_slider('Recent history', options=[60, 120, 250], value=120,
                                      format_func=lambda n: f'{n} sessions')
    st.caption('Applies to price history, volume and RSI. The model comparison uses its full test segment.')
    st.divider()
    st.markdown('**Inside the model**')
    st.caption('LSTM predicts high and low from a 10-session window. Ridge regression estimates open. Close is derived from the predicted range.')
    st.markdown('**Data source**')
    st.caption('Yahoo Finance · Daily observations since 2010. Prices follow the data provider’s quote currency.')
    st.divider()
    st.caption('Academic project demonstration')

st.markdown('<div class="eyebrow">MACHINE LEARNING / MARKET ANALYSIS</div>', unsafe_allow_html=True)
st.title('Stock price forecasting')
st.markdown('<div class="intro">Explore historical patterns. Compare model predictions. Preview the next session.</div>', unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner='Fetching stock history...')
def fetch_history(symbol):
    return yf.download(symbol, start=datetime.datetime(2010, 1, 1), end=datetime.datetime.now(), progress=False)

@st.cache_resource(show_spinner='Loading the saved LSTM model...')
def saved_model(path):
    return load_model(path)

np.random.seed(42)#To ensure the randomness here , that is to avoid shuffling of dataset everytime I run this code
random.seed(42)
tf.random.set_seed(42)

# Fetch stock data
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime.now()
if not user_input or any(c.isspace() for c in user_input) or ',' in user_input:
    st.info('Enter one stock symbol in the sidebar to begin.')
    st.stop()
try:
    df = fetch_history(user_input)
except Exception as exc:
    st.error(f'Could not fetch stock history. Please try again. ({type(exc).__name__})')
    st.stop()

# Recent yfinance versions return a (Price, Ticker) MultiIndex even for one ticker.
# Flatten it so Open/High/Low/Close are numeric Series as this script expects.
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index(drop=False)



# Check if data is fetched successfully

if df.empty:
    st.error(f"No data found for the symbol: {user_input}. Please check the symbol and try again.")
    st.stop()  

# Prepare the data
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
if len(df) < 80:
    st.warning('This symbol does not have enough usable history for a 10-session model window. Try AAPL or MSFT.')
    st.stop()
history_dates = df['Date'].copy()

# Select features
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Day', 'Month', 'Year', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI']
data = df[features].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# training and testing
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 10:]


#sequence  for lstm and data for ridge regression
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

# HYPERMODEL
class LSTMHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(LSTM(
            units=hp.Int('units_1', min_value=64, max_value=128, step=64),
            return_sequences=True,
            input_shape=(time_step, X_train.shape[2]),
            kernel_regularizer=l2(0.01)
        ))
        
        model.add(Dropout(0.2))
        model.add(LSTM(
            units=hp.Int('units_2', min_value=64, max_value=128, step=64),
            return_sequences=False
        ))
        
        model.add(Dropout(0.2))
        model.add(Dense(hp.Int('dense_units', min_value=16, max_value=32, step=8)))
        model.add(Dense(2))
        
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(
                hp.Choice('learning_rate', values=[1e-2, 1e-3])
            ),
            loss='mean_absolute_error'
        )
        return model

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.9  # Simplified decay for compatibility

hypermodel = LSTMHyperModel()


model_file = 'model.h5'
if os.path.exists(model_file):
    # Load the existing model
    best_model = saved_model(model_file)
else:
    # hyperparameter tuning
    tuner = RandomSearch(
        hypermodel,
        objective='val_loss',
        max_trials=5,  # Reduced from 10
        executions_per_trial=1,  # Reduced from 2
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


if adjusted_predicted_prices[0][0] < adjusted_predicted_prices[0][2]:
    adjusted_predicted_prices[0][0] = adjusted_predicted_prices[0][2]
if adjusted_predicted_prices[0][0] > adjusted_predicted_prices[0][1]:
    adjusted_predicted_prices[0][0] = adjusted_predicted_prices[0][1]

# Dashboard output: show the source session and keep forecasts visibly distinct.
latest_date = df['Date'].iloc[-1].strftime('%d %b %Y')
st.markdown(f'<div class="session-bar"><b>{escape(user_input)}</b> &nbsp; / &nbsp; Latest available session: {latest_date} &nbsp; / &nbsp; Daily data</div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">01 / OBSERVED MARKET DATA</div>', unsafe_allow_html=True)
st.subheader("Today's Prices")
st.caption(f'Latest available trading session: {latest_date}. Weekends and holidays may have no new observations.')
for col, label in zip(st.columns(4), ('Open', 'High', 'Low', 'Close')):
    col.metric(label, f"{float(df[label].iloc[-1]):,.2f}")

st.markdown('<div class="section-label">02 / MODEL OUTLOOK</div>', unsafe_allow_html=True)
st.subheader("Tomorrow's Predicted Prices")
st.caption('Forecast for the next trading session after the latest observation. Close is the midpoint of predicted high and low.')
for col, label, value in zip(st.columns(4), ('Open', 'High', 'Low', 'Close'), (*adjusted_predicted_prices[0], predicted_closing_value)):
    col.metric(f'Predicted {label}', f"{float(value):,.2f}")

def style_chart(fig, xlabel):
    fig.update_layout(template='plotly_white', height=410, paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                      margin=dict(t=30, b=40, l=20, r=20), font=dict(family='Arial', color='#526681', size=12),
                      legend=dict(orientation='h', y=1.13, x=0), hovermode='x unified',
                      xaxis_title=xlabel, yaxis_title='Price (quote currency)')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='#edf1f7', zeroline=False)
    return fig

st.markdown('<div class="section-label">03 / HISTORICAL COMPARISON</div>', unsafe_allow_html=True)
st.subheader('Prediction vs Original')
st.caption('Actual high and low against LSTM outputs on the final chronological segment. This chart alone does not establish forecast accuracy.')
comparison = go.Figure()
test_dates = history_dates.iloc[train_size:train_size + len(y_test)]
for values, name, color, dash in (
    (y_test[:, 0], 'Actual High', '#087f8c', 'solid'),
    (y_predicted[:, 0], 'Predicted High', '#6366f1', 'dash'),
    (y_test[:, 1], 'Actual Low', '#d4902a', 'solid'),
    (y_predicted[:, 1], 'Predicted Low', '#dc647d', 'dash'),
):
    comparison.add_scatter(x=test_dates, y=values, mode='lines', name=name, line=dict(color=color, dash=dash, width=2))
st.plotly_chart(style_chart(comparison, 'Trading session'), use_container_width=True)

st.markdown('<div class="section-label">04 / NEXT SESSION PREVIEW</div>', unsafe_allow_html=True)
st.subheader('Graph of Predicted Prices for Tomorrow')
st.caption('Ten observed sessions followed by one forecast. The shaded area and dashed lines identify the forecast.')
forecast_fig = go.Figure()
labels = list(history_dates.iloc[-10:].dt.strftime('%d %b %Y')) + ['Next session']
next_values = (*adjusted_predicted_prices[0], predicted_closing_value)
for i, (name, color) in enumerate(zip(('Open', 'High', 'Low', 'Close'), ('#6366f1', '#087f8c', '#d4902a', '#dc647d'))):
    values = data[-10:, i]
    forecast_fig.add_scatter(x=labels[:10], y=values, mode='lines+markers', name=name,
                             line=dict(color=color, width=2), marker=dict(size=5), legendgroup=name)
    forecast_fig.add_scatter(x=labels[-2:], y=[values[-1], next_values[i]], mode='lines+markers',
                             line=dict(color=color, width=2, dash='dash'), marker=dict(size=8, symbol='diamond'),
                             name=f'{name} forecast', legendgroup=name, showlegend=False)
forecast_fig.add_vrect(x0=9.5, x1=10.5, fillcolor='#eef2ff', opacity=.6, line_width=0, layer='below')
forecast_fig.update_xaxes(type='category', categoryorder='array', categoryarray=labels)
st.plotly_chart(style_chart(forecast_fig, 'Observed sessions / next-session forecast'), use_container_width=True)

st.markdown('<div class="section-label">05 / MARKET EXPLORER</div>', unsafe_allow_html=True)
st.subheader('Price history & moving averages')
st.caption('Each candle shows a session’s open, high, low and close. Moving averages smooth the observed closing prices; they are not forecasts.')
recent = df.tail(chart_sessions)
price_fig = go.Figure(go.Candlestick(
    x=recent['Date'], open=recent['Open'], high=recent['High'], low=recent['Low'], close=recent['Close'],
    name='Observed OHLC', increasing_line_color='#0d9488', decreasing_line_color='#e45c83',
))
for field, name, color in [('SMA_20', '20-session SMA', '#6366f1'), ('SMA_50', '50-session SMA', '#eaa12c')]:
    price_fig.add_scatter(x=recent['Date'], y=recent[field], name=name, mode='lines', line=dict(color=color, width=2))
style_chart(price_fig, 'Trading session')
price_fig.update_layout(xaxis_rangeslider_visible=False, height=440)
st.plotly_chart(price_fig, use_container_width=True)

volume_col, rsi_col = st.columns(2)
with volume_col:
    st.subheader('Trading volume')
    st.caption('Shares traded per session. Colors indicate close versus open.')
    volume_fig = go.Figure(go.Bar(x=recent['Date'], y=recent['Volume'], name='Volume',
        marker_color=np.where(recent['Close'] >= recent['Open'], '#23b5a0', '#ec86aa')))
    style_chart(volume_fig, 'Trading session')
    volume_fig.update_layout(yaxis_title='Shares traded', height=340, showlegend=False)
    st.plotly_chart(volume_fig, use_container_width=True)
with rsi_col:
    st.subheader('RSI · 14 sessions')
    st.caption('Momentum indicator on a 0–100 scale; 30 and 70 are reference levels.')
    rsi_fig = go.Figure(go.Scatter(x=recent['Date'], y=recent['RSI'], name='RSI',
        mode='lines', line=dict(color='#8b5cf6', width=2.5)))
    rsi_fig.add_hrect(y0=30, y1=70, fillcolor='#ede9fe', opacity=.5, line_width=0)
    for level, color in [(30, '#0d9488'), (70, '#db5b91')]:
        rsi_fig.add_hline(y=level, line_dash='dot', line_color=color)
    style_chart(rsi_fig, 'Trading session')
    rsi_fig.update_layout(yaxis_title='RSI', yaxis_range=[0, 100], height=340, showlegend=False)
    st.plotly_chart(rsi_fig, use_container_width=True)

st.markdown('<div class="section-label">06 / MODEL DIAGNOSTICS</div>', unsafe_allow_html=True)
st.subheader('Where predictions differ from observations')
st.caption('Prediction minus actual price on the displayed test segment. Above zero means an overestimate; below zero means an underestimate.')
residuals = y_predicted - y_test
error_fig = go.Figure()
for i, name, color in [(0, 'High error', '#6366f1'), (1, 'Low error', '#db5b91')]:
    error_fig.add_scatter(x=test_dates, y=residuals[:, i], name=name, mode='lines', line=dict(color=color, width=1.5))
error_fig.add_hline(y=0, line_color='#64748b', line_dash='dash')
style_chart(error_fig, 'Trading session')
error_fig.update_layout(yaxis_title='Predicted minus actual (quote currency)', height=340)
st.plotly_chart(error_fig, use_container_width=True)
st.caption('These errors describe the existing model and preprocessing on this dataset. They are not an independent benchmark: the scaler uses the full downloaded history and the saved model’s training split is not documented.')
with st.expander('How to read this dashboard'):
    st.write('Solid lines show observed prices. Dashed lines show model outputs. The saved LSTM is reused; the displayed outlook also includes the original project’s offset and opening-price constraints. These outputs are a classroom demonstration, not independently validated forecasts.')
st.divider()
st.caption('StockScope · LSTM + Ridge Regression · Academic demonstration')
