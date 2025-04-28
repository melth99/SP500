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

class DataConfig:
    alpaca_api_key = os.environ["APCA-API-KEY-ID"]
    alpaca_secret_key = os.environ["APCA-API-SECRET-KEY"]
    now = dt.datetime.now()
    startTraining = now - dt.timedelta(days=60)
    symbol = "SPY"

    stock_client = StockHistoricalDataClient(
        api_key = alpaca_api_key,
        secret_key = alpaca_secret_key,
    )


class FetchStockData:
    def __init__(self, symbol, start, end, stock_client):
        self.stock_client = stock_client
        self.request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            pagination=True
        )
    def fetch_data(self,stock_client):
        try:
            bars = stock_client.get_stock_bars(self.request_params)
            df = bars.df #dataframe from pandas
            print(df)

        except Exception as e:
            print(f"Error fetching data: {str(e)}")

class PrepareData:
    def __init__(self, df):
        self.df = df
        self.scaler = MinMaxScaler()
    def prepare_data(self):
        # utc timestamp | open | high | low | close | vwap
        self.df = self.df.sort_index()
        self.df = self.df.dropna() #gets rid of missing values in df
        return self.df
    def split_data(self):
        #split data into train and test sets
        train_data = self.df.iloc[:int(0.8*len(self.df))] #80% of data for training
        test_data = self.df.iloc[int(0.2*len(self.df)):] #20% of data for testing (per cross validation info)
        return train_data, test_data
    
    def scale_data(self): #this method essentially preps this data for the LSTM model
        # transforming reduces noise, endcoding, normalization, handles missing values
        
        # fitting Avoids data leakage: By fitting only on the training data, you prevent information from the test set leaking into the model,
        # which could artificially inflate performance metrics
        #Fitting is the step where the transformation “learns” how to process the data;
        
        # transforming then applies this learned process to new data
        
        train_data = self.scaler.fit_transform(train_data) #uses both fit & transform
        #The fit(data) method is used to compute the mean and std dev 
        # for a given feature to be used further for scaling.
        test_data = self.scaler.transform(test_data)
        # transform(data) method is used to perform scaling using 
        # mean and std dev calculated using the .fit() method.
        
    
def main():
    fetch_stock_data = FetchStockData(DataConfig.symbol, DataConfig.startTraining, DataConfig.now, DataConfig.stock_client)
    df_time_series = fetch_stock_data.fetch_data(DataConfig.stock_client)
    prepare_data = PrepareData(df_time_series)
    data_prepared = prepare_data.prepare_data()
    



if __name__ == "__main__":
    main()
