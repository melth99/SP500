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
from data.auto_data import AutoData

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
        self.request_today = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=dt.datetime.now().date(),
            pagination=True
        ) #todays value we can compare to
        self.columns = ["open", "high", "low", "close", "volume", "vwap", "timestamp"]
        self.column_num = len(self.columns)
    def fetch_data(self,stock_client):
        try:
            bars = stock_client.get_stock_bars(self.request_params)
            today_bars = stock_client.get_stock_bars(self.request_today)
            df = bars.df #dataframe from pandas
            return df, today_bars
            #print(df)

        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            
class PrepareData:
    def __init__(self, df, today_bars):
        self.df = df
        self.today_bars = today_bars
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
    
    def today(self):
        self.unscaled_price_at_market_close = self.test_data[-1][6]
        return self.unscaled_price_at_market_close
    
    def scale_data(self): #this method essentially preps this data for the LSTM model
        # transforming reduces noise, endcoding, normalization, handles missing values
        
        # fitting Avoids data leakage: By fitting only on the training data, you prevent information from the test set leaking into the model,
        # which could artificially inflate performance metrics
        #Fitting is the step where the transformation "learns" how to process the data;
        
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
class FindAnswer: #remmeber 6 features # lets do sequences of 10 days to predict 11th day
    def __init__(self, model, test_data, sequence_length, scaler, today_price):
        self.model = model
        self.prediction = None
        self.sequence_length = sequence_length #10 days (for out 60 day model right now)
        self.n_features = 7 #constant from our alpaca data no need to create variable
        self.latest_window = test_data[-sequence_length:] #grabs last "sequence_length" rows of data
        self.latest_sequence = None
        self.scaler = scaler
        self.actual_row = None
        self.predicted_price_actual = None
        self.real_world_answer = None
        self.dummy_input = None
        self.today_price = today_price
        
   
            
            
    def find_latest_sequence(self):
        print('latest_window', self.latest_window)
        self.latest_sequence = self.latest_window.reshape(1,self.sequence_length,self.n_features)
        self.prediction = self.model.predict(self.latest_sequence)
        

        return self.prediction # needs to be un-scaled

    def de_scale(self):
        self.dummy_input = np.zeros((1, 7))  # 7 = number of features 
        self.dummy_input[0, -1] = self.prediction[0][0]  # only the 'close' price is predicted
        self.scaled_price_at_market_close = self.latest_window[-1, -1]
        #adding 0s to the other features to just estime price or 'close' for simplicity


        self.actual_row = self.scaler.inverse_transform(self.dummy_input)
        self.predicted_price_actual = self.actual_row[0, -1]  # extract just the 'close' price
        return self.predicted_price_actual

    def print_answer(self):
        self.difference = self.predicted_price_actual - self.today_price #prediction for tmr vs price today
        self.difference_percentage = (self.difference / self.today_price) * 100
        self.real_world_answer = f"The predicted price is {self.predicted_price_actual}, while the price todayis {self.today_price}. The difference is {self.difference:.2f}, which is {self.difference_percentage:.2f}%"
        return self.real_world_answer
        
def create_sequences(data, sequence_length): #purposefully outside of class ^
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length][-1])
    return np.array(x), np.array(y)

def main():
    config = DataConfig()
    fetch_stock_data = FetchStockData(config.symbol, config.startTraining, config.now, config.stock_client)
    df, request_today = fetch_stock_data.fetch_data(config.stock_client)
    
    # converting today bars to a DataFrame
    today_df = request_today.df
    print('Today\'s data:', today_df)
    
    preparedata = PrepareData(df, request_today)
    preparedata.prepare_data()
    train_data, test_data = preparedata.split_data()
    preparedata.train_data = train_data
    preparedata.test_data = test_data
    preparedata.scale_data()

    neuron_num = 32
    create_model = CreateLSTMModel(preparedata.train_data, preparedata.test_data, fetch_stock_data.column_num, neuron_num)
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
        create_model.train_data[:, -1],
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
    
    # Get today's price from the DataFrame
    today_price = today_df['close'].iloc[-1]
    print('Today\'s closing price:', today_price)
    
    # Make prediction
    sequence_length = 10
    find_answer = FindAnswer(create_model.stock_model, preparedata.test_data, sequence_length, preparedata.scaler, today_price)
    prediction = find_answer.find_latest_sequence()
    predicted_price = find_answer.de_scale()
    print('Today\'s closing price:', today_price)
    print("Predicted price for tomorrow:", predicted_price)
    print(find_answer.print_answer())

if __name__ == "__main__": 
    main()

