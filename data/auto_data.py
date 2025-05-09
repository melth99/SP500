
class AutoData:
    
    
    def __init__(self, df, request_today):
        
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
        "Test MAE": test_mae,
        "Test Loss": test_loss
        }
        self.val_row = row = {
        "Val Date (YYYY-MM-DD)": str(dt.datetime.now().date()),
        "Val Day Of Week": str(dt.datetime.now().weekday()),
        "Val Price": today_price,
        "Predicted Value": predicted_price,
        "Val Diff": predicted_price - today_price,
        "Val Change(%)": ((predicted_price - today_price) / today_price) * 100,
        "Val MAE": test_mae,
        "Val Loss": test_loss
        }
        self.notes_row = row = {
            "Post-Break?(T/F)": 'T/F', #if after weekend or holiday
            'Model Ver': 'Model Ver',
            'Notes': 'Notes'
        }
        
    def auto_data(self):
        self.df = self.df.sort_index()
        
    
    def updateTD_txt(self):
        with open('TD.txt', 'w') as file:
            
    def plot_data(self):
        plt.plot(self.df['close'])
        plt.show()
