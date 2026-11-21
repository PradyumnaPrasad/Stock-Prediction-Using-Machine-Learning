
# Stock Price Predictor (LSTM + Ridge Regression)

This project is a **Streamlit** web application that uses **deep learning (LSTM)** and **machine learning (Ridge Regression)** to predict the next day's stock prices (Open, High, Low, Close) based on historical stock data fetched from Yahoo Finance.

## 🔍 Features

* Fetches historical stock data for any ticker symbol.
* Computes technical indicators: SMA, EMA, RSI.
* Preprocesses and scales data using `MinMaxScaler`.
* Predicts:

  * **Open** price using Ridge Regression.
  * **High**, **Low**, and derived **Close** prices using LSTM.
* Uses **Keras Tuner** to optimize LSTM hyperparameters.
* Automatically saves and loads the best LSTM model (`model.h5`).
* Shows a side-by-side comparison of predicted vs actual prices.
* Interactive and intuitive UI using **Streamlit**.

---

## 🧠 Technologies Used

* `Python`
* `Streamlit` - UI
* `yfinance` - Fetching stock data
* `Keras` / `TensorFlow` - LSTM deep learning
* `Keras Tuner` - Hyperparameter tuning
* `Scikit-learn` - Ridge regression and scaling
* `Matplotlib` - Plotting predictions

---

## ⚙️ Installation & Setup

1. **Create and activate a Python 3.11 environment (Windows PowerShell):**

   ```bash
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run the actual app file:**

   ```bash
   streamlit run stock_prediction.py
   ```

   The app is available at `http://localhost:8501`. It fetches market data from Yahoo Finance, so it needs an internet connection. The included `model.h5` is loaded when present; hyperparameter tuning and training happen only if it is missing.

Python 3.11 and TensorFlow 2.18 are used because the checked-in HDF5 model was saved with Keras 3.7; the original TensorFlow 2.15 pin cannot load it. This is a research/education demo, not investment advice. Forecasts are uncertain and should not guide financial decisions.

---

## 🧪 How It Works

* **LSTM**: Trained on sequences of past 10 days to predict next day's High & Low.
* **Ridge Regression**: Predicts Open price using the previous day's Close.
* **Close price** is derived as the average of predicted High and Low.
* Keras Tuner uses Random Search to find optimal LSTM configuration.
* Early stopping and learning rate scheduling ensure efficient training.
* The model is persisted as `model.h5` to avoid retraining every time.

---

## 📊 Example Output

After entering a stock ticker like `AAPL`, you’ll see:

* Today’s High and Low
* Predicted Open, High, Low, and Close for tomorrow
* Plots comparing predicted vs actual prices

---

## 📁 File Structure

```
.
├── stock_prediction.py                # Main Streamlit app
├── model.h5              # Included saved LSTM model
├── hyperparameter_tuning/
│   └── lstm_stock_prediction/  # Keras Tuner files
├── requirements.txt
└── README.md
```

---

## 🚀 Future Improvements

* Add support for more indicators (MACD, Bollinger Bands).
* Incorporate sentiment analysis from news headlines.
* Provide weekly or monthly predictions.
* Add explainability using SHAP or attention layers.

---

## 🧑‍💻 Author

**Pradyumna**
2nd Year AIML Engineering Student
Building ML and GenAI tools for real-world impact.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---
