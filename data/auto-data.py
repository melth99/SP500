
class AutoData:
    
    def __init__(self, df, request_today):
        self.df = df
        self.request_today = request_today
        
    def auto_data(self):
        self.df = self.df.sort_index()
    
    def updateTD_txt(self):
        with open('TD.txt', 'w') as file:
