import numpy as np
import pandas as pd
import datetime as dt
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.requests import StockBarsRequest
from alpaca.data import TimeFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense 
import requests
import json
import os

# Load environment variables from .env file
load_dotenv()

class DataConfig:
    def __init__(self):
        self.alpaca_api_key = os.environ["APCA-API-KEY-ID"]
        self.alpaca_secret_key = os.environ["APCA-API-SECRET-KEY"]
        self.now = dt.datetime.now()
        self.startTraining = self.now - dt.timedelta(days=60)
        self.symbol = "SPY"
        self.stock_client = StockHistoricalDataClient(
            api_key = self.alpaca_api_key,
            secret_key = self.alpaca_secret_key,
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
        self.columns = ["open", "high", "low", "close", "volume", "vwap", "timestamp"]
        self.column_num = len(self.columns)
    def fetch_data(self,stock_client):
        try:
            bars = stock_client.get_stock_bars(self.request_params)
            df = bars.df #dataframe from pandas
            return df
            #print(df)

        except Exception as e:
            print(f"Error fetching data: {str(e)}")

class PrepareData:
    def __init__(self, df):
        self.df = df
        self.scaler = MinMaxScaler()
        self.train_data = None
        self.test_data = None
    def prepare_data(self):
        # utc timestamp | open | high | low | close | vwap
        self.df = self.df.sort_index()
        self.df = self.df.dropna() #gets rid of missing values in df

    def split_data(self):
        #split data into train and test sets
        train_data = self.df.iloc[:int(0.8*len(self.df))] #first 80% of data for training
        test_data = self.df.iloc[int(0.2*len(self.df)):] #last 20% of data for testing (per cross validation info)
        return train_data, test_data
    
    def scale_data(self): #this method essentially preps this data for the LSTM model
        # transforming reduces noise, endcoding, normalization, handles missing values
        
        # fitting Avoids data leakage: By fitting only on the training data, you prevent information from the test set leaking into the model,
        # which could artificially inflate performance metrics
        #Fitting is the step where the transformation “learns” how to process the data;
        
        # transforming then applies this learned process to new data
        
        self.train_data = self.scaler.fit_transform(self.train_data) #uses both fit & transform
        #The fit(data) method is used to compute the mean and std dev 
        # for a given feature to be used further for scaling.
        self.test_data = self.scaler.transform(self.test_data)
        # transform(data) method is used to perform scaling using 
        # mean and std dev calculated using the .fit() method.
        
class CreateLSTMModel:
    def __init__(self, train_data, test_data, column_num, neuron_num):
        self.train_data = train_data
        self.test_data = test_data
        self.column_num = column_num
        self.neuron_num = neuron_num
        self.stock_model = Sequential()

    def create_model(self, column_num):
        # Reshape input data for LSTM [samples, timesteps, features]
        self.train_data = self.train_data.reshape((self.train_data.shape[0], 1, self.train_data.shape[1]))
        self.test_data = self.test_data.reshape((self.test_data.shape[0], 1, self.test_data.shape[1]))

        # First LSTM layer
        self.stock_model.add(LSTM(
            units=self.neuron_num,
            return_sequences=True,
            input_shape=(1, self.column_num),
            name="lstm_1"
        ))

        # Additional LSTM layers with decreasing number of neurons
        i = 2
        neuron_loop = self.neuron_num // 2
        while neuron_loop >= 2:
            self.stock_model.add(LSTM(
                units=neuron_loop,
                return_sequences=(neuron_loop > 2),  # Only return sequences if not the last LSTM layer
                name=f"lstm_{i}"
            ))
            neuron_loop = neuron_loop // 2
            i += 1

        # Final Dense layer for prediction
        self.stock_model.add(Dense(1, activation="linear", name="output_layer"))

        # Print model summary
        self.stock_model.summary()

def main():
    config = DataConfig()
    fetch_stock_data = FetchStockData(config.symbol, config.startTraining, config.now, config.stock_client)
    df_time_series = fetch_stock_data.fetch_data(config.stock_client)
    print("Raw data fetched:\n", df_time_series.head())

    prepare_data = PrepareData(df_time_series)
    prepare_data.prepare_data()
    train_data, test_data = prepare_data.split_data()
    prepare_data.train_data = train_data
    prepare_data.test_data = test_data
    prepare_data.scale_data()

    neuron_num = 32
    create_model = CreateLSTMModel(prepare_data.train_data, prepare_data.test_data, fetch_stock_data.column_num, neuron_num)
    create_model.create_model(fetch_stock_data.column_num)
    
    # Compile the model
    create_model.stock_model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mae"]
    )
    
    # Train the model
    history = create_model.stock_model.fit(
        create_model.train_data,
        create_model.train_data[:, -1],  # 
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_mae = create_model.stock_model.evaluate(
        create_model.test_data,
        create_model.test_data[:, -1]
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")


if __name__ == "__main__":
    main()
