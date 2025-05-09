from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
class AutoData:
    
    def __init__(self, df, request_today):
        self.trading_client = TradingClient(os.environ["APCA-API-KEY-ID"], os.environ["APCA-API-SECRET-KEY"])
        # separating rows into 3 chunks for accessing later 
        self.df = df
        self.request_today = request_today
        self.test_row = {
        "Test Date (YYYY-MM-DD)": str(dt.datetime.now().date()),
        "Test Day Of Week": str(dt.datetime.now().weekday()),
        "Test Price": today_price,
        "Predicted Value": predicted_price,
        "Test Diff": predicted_price - today_price,
        "Test Change(%)": ((predicted_price - today_price) / today_price) * 100,
        "Test Diff Sign(+/-)": predicted_price - today_price,
        "Test MAE": test_mae, #from terminal
        "Test Loss": test_loss #from terminal
        }
        self.val_row = { #calculates dependent variables next day "validation"
        "Val Date (YYYY-MM-DD)": str(dt.datetime.now().date()),
        "Val Day Of Week": str(dt.datetime.now().weekday()),
        "Val Price": predicted_price, #next day calculation
        "Val Diff": predicted_price - today_price, #next day calculation
        "Val Change(%)": ((predicted_price - today_price) / today_price) * 100,
        "Val MAE": val_mae, #from terminal
        "Val Loss": val_loss #from terminal
        }
        # add if else statement somewhere to only run when test_date == val_date[-1] or something like that
        self.notes_row = { #values that show context of prediction data
            "Post-Break?(T/F)": None, #if after weekend or holiday
            'Weekend Break?(T/F)': None, #if after weekend or holiday
            'Else Break(T/F)': None, #if after weekend or holiday
            'Model Ver': None, #determine later......
            'Notes': None #any significant events effecting stock (might use chatgpt to do this later)
        }
        
    # only runs if val_row[Price] is not None
    def after_a_break(self): #tracking if US stock exchange is open between two adjactent days
        self.val_date = self.val_row["Val Date (YYYY-MM-DD)"]
        self.test_date = self.test_row["Test Date (YYYY-MM-DD)"]
        self.calendar = self.trading_client.get_calendar(GetCalendarRequest(start=self.val_date, end=self.test_date))
       
        
    def auto_data(self):
        self.df = self.df.sort_index()
        
    
    def updateTD_txt(self):
        with open('TD.txt', 'w') as file:
            
    def plot_data(self):
        plt.plot(self.df['close'])
        plt.show()
