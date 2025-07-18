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
import smtplib
import unittest
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl


with smtplib.SMTP("domain.org") as smtp:
    smtp.noop() #this is just to smtp statement quits
    
who_wants_this = []
message = ''
    
#%(asctime)s - Adds a timestamp showing when the log message was created
#%(message)s - The actual log message you write in your code """

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current directory
parent_dir = os.path.dirname(current_dir)  # Get parent directory
sys.path.append(parent_dir)  # Add parent directory to Python path

from main import DataConfig
from main import pass_to_auto_data

load_dotenv()
host_name = 'stonks'
email_address = os.getenv('EMAIL-SENDER')
email_pw = os.getenv('EMAIL-SENDER-PW')




class AutoData(unittest.TestCase, smtp, host_name):
    
    
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
        try:
            print("Running scheduled task...")
            result = subprocess.run(["python", "main.py"], capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"Error running main.py: {result.stderr}")
                return False
            return True
        except Exception as e:
            logging.error(f"Exception in auto_data: {e}")
            return False

    def format_data(self):
        #formatting on how user recieves data
        delivery_message = 'Hi there \n The price of the S&P 500 is $' + str(pass_to_auto_data['today_price']) + ' \n The predicted price for tomorrow is $' + str(pass_to_auto_data['predicted_price']) + ' \n The difference is $' + str(pass_to_auto_data['difference']) + ' \n The difference percentage is ' + str(pass_to_auto_data['difference_percentage']) + '%'
        
        
    

    def trigger_run(self): #used for testing. trigger code to run on command 
        pass
    def plot_data(self):
        #not worried about this for now
        pass
    

        
    def test_logging(self):
        # Test method - to be implemented
        pass
        
    def test_does_main_run(self):
        # Test method - to be implemented
        pass
        
    def test_auto_data(self, unittest.TestCase):
        # Test method - to be implemented  
        self.auto_data()
        self.assertTrue(pass_to_auto_data)
        print(pass_to_auto_data)

class Emails(email_address, email_pw, smtp, host_name):
    def __init__(self, email_address, email_pw):
        self.email_address = email_address
        self.email_pw = email_pw
        self.host_name = host_name
        self.smtp = smtp

        
        def delivery(self):
            #formatting on how user recieves data
            smtp.ehlo(host_name)
            smtp.extn(host_name) # returns true if smtp server recieves request with host name :)
            
            
            
            smtp.quit()
            
        def logging(self):
            logging.info("Running scheduled task...")
            logging.info("Running scheduled task...")
            logging.info("Running scheduled task...")
            logging.info("Running scheduled task...")
                    
    

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