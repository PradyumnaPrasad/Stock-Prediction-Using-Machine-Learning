
# 📈 Stock Price Predictor (LSTM + Ridge Regression)

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

1. **Clone the repo:**

   ```bash
   git clone https://github.com/yourusername/stock-price-predictor.git
   cd stock-price-predictor
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:

   ```text
   numpy
   pandas
   yfinance
   scikit-learn
   keras
   keras-tuner
   tensorflow
   matplotlib
   streamlit
   ```

4. **Run the app:**

   ```bash
   streamlit run app.py
   ```

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
├── model.h5              # Saved best LSTM model (created after training)
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
