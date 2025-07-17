from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

import datetime as dt
import schedule
import time
import subprocess
import os
from dotenv import load_dotenv
import sys
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current directory
parent_dir = os.path.dirname(current_dir)  # Get parent directory
sys.path.append(parent_dir)  # Add parent directory to Python path

from main import DataConfig

load_dotenv()

class AutoData:
    
    
    """[ ////////this is the format of calendar data json frm api we need entire program to run only
        if open true & close false then it is a weekend break
        {
            "date": "string",
            "open": "string",
            "close": "string",
            "settlement_date": "string"
        }
        ]
    """
    
    def __init__(self, df=None, request_today=None, today_price=None, predicted_price=None, test_mae=None, test_loss=None, val_mae=None, val_loss=None):
        self.trading_client = TradingClient(os.environ["APCA-API-KEY-ID"], os.environ["APCA-API-SECRET-KEY"])
        self.df = df
        self.request_today = request_today
        self.test_row = {
            "Test Date (YYYY-MM-DD)": str(dt.datetime.now().date()),
            "Test Day Of Week": str(dt.datetime.now().weekday()),
            "Test Price": today_price,
            "Predicted Value": predicted_price,
            "Test Diff": (predicted_price - today_price) if predicted_price and today_price else None,
            "Test Change(%)": ((predicted_price - today_price) / today_price * 100) if predicted_price and today_price else None,
            "Test Diff Sign(+/-)": (predicted_price - today_price) if predicted_price and today_price else None,
            "Test MAE": test_mae,
            "Test Loss": test_loss
        }
        self.val_row = {
            "Val Date (YYYY-MM-DD)": str(dt.datetime.now().date()),
            "Val Day Of Week": str(dt.datetime.now().weekday()),
            "Val Price": predicted_price,
            "Val Diff": (predicted_price - today_price) if predicted_price and today_price else None,
            "Val Change(%)": ((predicted_price - today_price) / today_price * 100) if predicted_price and today_price else None,
            "Val MAE": val_mae,
            "Val Loss": val_loss
        }
        self.notes_row = {
            "Post-Break?(T/F)": None,
            'Weekend Break?(T/F)': None,
            'Else Break(T/F)': None,
            'Model Ver': None,
            'Notes': None
        }
        self.val_date = None
        self.test_date = None
        self.calendar = None

        #might not be nessesary
        """ def after_a_break(self):
            if self.val_date and self.test_date:
                self.calendar = self.trading_client.get_calendar(
                    GetCalendarRequest(start=self.val_date, end=self.test_date)
                )
                return self.calendar
            return None """

    def auto_data(self):
        #this is where we will add the code to automate
        print("Running scheduled task...")
            
        subprocess.run(["python", "main.py"])
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def format_data(self):
        #formatting on how user recieves data
        pass
    
    def delivery(self):
        #formatting on how user recieves data
        pass
    
    def trigger_run(self): #used for testing. trigger code to run on command 
        pass
    def plot_data(self):
        #not worried about this for now
        pass
    
    def logging(self):
        logging.info("Running scheduled task...")
        logging.info("Running scheduled task...")
        logging.info("Running scheduled task...")
        logging.info("Running scheduled task...")
        
    def test_logging(self):
        # Test method - to be implemented
        pass
        
    def test_does_main_run(self):
        # Test method - to be implemented
        pass
        
    def test_auto_data(self):
        # Test method - to be implemented  
        pass


def main():
    config = DataConfig()
    auto_data = AutoData(
        df=None,
        request_today=None,
        today_price=None,
        predicted_price=None,
        test_mae=None,
        test_loss=None,
        val_mae=None,
        val_loss=None
    )
    print(auto_data.calendar)
    
    



if __name__ == "__main__":
    main()  