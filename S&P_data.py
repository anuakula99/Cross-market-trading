import yfinance as yf

# Define the ticker symbol and date range
ticker = '^GSPC'  # S&P 500
start_date = '2022-01-01'
end_date = '2023-01-01'

# Download historical data
data = yf.download(ticker, start=start_date, end=end_date)

# Export to CSV
data.to_csv('SP500_data.csv')
