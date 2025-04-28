import numpy as np
import pandas as pd
import datetime as dt
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.requests import StockBarsRequest 
from alpaca.data import TimeFrame
"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt """
""" from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense """
import requests
import json
import os

# Load environment variables from .env file
load_dotenv()
alpaca_api_key = os.environ["APCA-API-KEY-ID"]
alpaca_secret_key = os.environ["APCA-API-SECRET-KEY"]

stock_client = StockHistoricalDataClient(
    api_key = alpaca_api_key,
    secret_key = alpaca_secret_key,
)

request_params = StockBarsRequest(
    symbol_or_symbols="SPY",
    timeframe=TimeFrame.Day,
    start=dt.datetime(2024, 11, 1),
    end=dt.datetime.now().date(),
    pagination=True
)

try:
    bars = stock_client.get_stock_bars(request_params)
    df = bars.df #dataframe from pandas
    print(df)

except Exception as e:
    print(f"Error fetching data: {str(e)}")











