📌 Overview
This project uses Long Short-Term Memory (LSTM) networks to predict stock prices for global stock markets. It fetches historical stock data using Yahoo Finance based on a given ticker, start date, and end date. The model then predicts future stock prices and calculates the returns for the given date range.

🔧 Features
Fetches real-time stock prices using Yahoo Finance.
Allows users to input stock ticker, start date, and end date.
Uses LSTM deep learning model for future stock price prediction.
Computes returns between the selected start and end dates.

🚀 How to Use
Install Dependencies
pip install -r requirements.txt

To run the Project
python prediction.py

Input Required:
Stock ticker symbol (e.g., DMART.NS, HDFCBANK.NS, etc..,)
For the indian stocks we have to add suffix .NS
Start date and end date for fetching historical prices
Model predicts future stock prices and calculates returns

🛠 Technologies Used
Python
TensorFlow / Keras (LSTM Model)
Yahoo Finance API (yfinance)
Pandas, NumPy, Matplotlib

Fetching stock data for: AAPL  
Start Date: 2023-01-01 | End Date: 2024-01-01  
Predicted Future Price: $185.60  
Total Return: +12.5%  

📝 Key Features
Add different machine learning models for comparison.
Implement a web-based UI using Flask or Streamlit.
Improve prediction accuracy with more data preprocessing techniques.

