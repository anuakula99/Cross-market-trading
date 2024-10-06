import pandas as pd

# Load historical stock data (e.g., S&P 500) from Yahoo Finance
data = pd.read_csv('sp500_data.csv')

# Check number of rows in the dataset
num_steps = len(data)
print(f"Number of steps (trading days): {num_steps}")


