import yfinance as yf


# Define the S&P500 symbol and date range
ticker = '^GSPC'  # S&P 500
start_date = '2006-01-01'
end_date = '2024-01-01'

# Download historical data
index_data = yf.download(ticker, start=start_date, end=end_date)
# Export to CSV
index_data.to_csv('SP500_data.csv')

