""" import numpy as np
import pandas as pd
import datetime as dt
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.requests import StockBarsRequest
from alpaca.data import TimeFrame

import requests
import json
import os



load_dotenv()
client = StockHistoricalDataClient(os.environ["APCA-API-KEY-ID"], os.environ["APCA-API-SECRET-KEY"])
request = StockLatestQuoteRequest(symbol_or_symbols="SPY")
latest_quote = client.get_stock_latest_quote(request)
price = latest_quote['SPY'].ask_price  # or .bid_price

print("Latest SPY price:", price)
 """
import builtins
help(builtins)