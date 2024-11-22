import yfinance as yf
import pandas as pd


# Step 1: Download Stock Market Data (S&P 500)
 # S&P 500
start_date = '2018-01-01'
end_date = '2024-06-30'

ticker = '^GSPC' 
stock_data = yf.download(ticker, start=start_date, end=end_date)
stock_data.to_csv('sp500_data.csv')

# Step 2: Download Cryptocurrency Data (Bitcoin)


ticker = 'BTC-USD'  
bitcoin_data = yf.download(ticker, start=start_date, end=end_date)
bitcoin_data.to_csv('bitcoin_data.csv')

# Step 3: Download Commodity Data (Gold)
ticker_gold = 'GC=F'  # Gold Futures
gold_data = yf.download(ticker_gold, start=start_date, end=end_date)
gold_data.to_csv('gold_data.csv')

# Step 4: Download Currency Market Data 
ticker_currency = 'EURUSD=X'  
eur_usd_data = yf.download(ticker_currency, start=start_date, end=end_date)
eur_usd_data.to_csv('eur_usd_data.csv')


