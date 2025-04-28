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
        self.columns = ["open", "high", "low", "close", "volume", "vwap", "timestamp"]
        self.column_num = len(self.columns)
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
        
class CreateLSTMModel:
    #this class creates the LSTM model
    
    def __init__(self, train_data, test_data, columns, neuron_num):
        self.train_data = train_data
        self.test_data = test_data
        self.stock_model = Sequential()
        self.train_data_shape = columns
        self.neuron_num =  neuron_num #starting here! can change later
    
    def one_step_ahead_prediction(self):
        #this method creates a one step ahead prediction
        #it takes the last 60 days of training data and uses it to predict the next day's closing price
        pass
    def multi_step_ahead_prediction(self):
        #this method creates a multi step ahead prediction
        #it takes the last # days of training data and uses it to predict the next 
        # Y number of days of closing prices
        pass
    
    def create_model(self, column_num): #not explicity called, but initial layer has
        #7 features for the 7 columns in the training data relfected in the input_shape
        #input layer has NO neurons it just passes the data to the first LSTM layer

        self.stock_model.add(Dense(32,activation="relu", input_shape=(column_num,), name="layer_1"))
        i = 2
        neuron_loop = self.neuron_num
        while i > 1: #allows for dynamic neuron number when fine-tuning later!
            neuron_loop = neuron_loop/2
            i += 1
            self.stock_model.add(Dense(neuron_loop,self.neuron_num,activation="relu", name=f"layer_{i}"))
        self.stock_model.add(Dense(1,activation="relu", name="output_layer"))
        
        """           
        self.stock_model.add(Dense(16,activation="relu", name="layer_2"))
        self.stock_model.add(Dense(8,activation="relu", name="layer_3"))
        self.stock_model.add(Dense(4,activation="relu", name="layer_4"))
        self.stock_model.add(Dense(2,activation="relu", name="layer_5")) """
        
        
        
        
        
        
        
        
        
        
        
        
def main():
    fetch_stock_data = FetchStockData(DataConfig.symbol, DataConfig.startTraining, DataConfig.now, DataConfig.stock_client)
    df_time_series = fetch_stock_data.fetch_data(DataConfig.stock_client)
    prepare_data = PrepareData(df_time_series)
    neuron_num = 32 #starting here! can change later will also change organization of this later!
    create_model = CreateLSTMModel(prepare_data.train_data, prepare_data.test_data, fetch_stock_data.column_num, neuron_num)
    
    
    
    """ multi_day_prediction = input("Do you want to predict for multiple days? (y/n)")
    if multi_day_prediction == "y" or multi_day_predictiction == "Y" or multi_day_prediction == "yes" or multi_day_prediction == "Yes" or multi_day_prediction == "True":
        multi_day_prediction = True
        print("How many days would you like to predict for?")
        num_days = int(input())

    elif multi_day_prediction == "n" or multi_day_predictiction == "N" or multi_day_prediction == "no" or multi_day_prediction == "No" or multi_day_prediction == "False":
        multi_day_prediction = False
    else:
        print("Invalid input")
        return
    if multi_day_prediction:
        multi_step_ahead_prediction = CreateLSTMModel(prepare_data.train_data, prepare_data.test_data)
        multi_step_ahead_prediction.multi_step_ahead_prediction()
    else: #predicting for tommorrow
        one_step_ahead_prediction = CreateLSTMModel(prepare_data.train_data, prepare_data.test_data)
        one_step_ahead_prediction.one_step_ahead_prediction()
    
    
 """


if __name__ == "__main__":
    main()
