import pandas as pd
import tech_indicators
from tech_indicators import add_technical_indicators

df_1 = pd.read_csv(f'sp500_data.csv', index_col='Date', parse_dates=True)
df_2 = pd.read_csv(f'bitcoin_data.csv', index_col='Date', parse_dates=True)
df_3 = pd.read_csv(f'gold_data.csv', index_col='Date', parse_dates=True)
df_4 = pd.read_csv(f'eur_usd_data.csv', index_col='Date', parse_dates=True)

training_data_time_range = ('2018-01-01', '2023-12-31')
test_data_time_range = ('2024-01-01', '2024-06-30')

df_1 = add_technical_indicators(df_1)
df_2 = add_technical_indicators(df_2)
df_3 = add_technical_indicators(df_3)
df_4 = add_technical_indicators(df_4)
train_data_stocks = df_1.loc[training_data_time_range[0]:training_data_time_range[1]]
train_data_crypto= df_2.loc[training_data_time_range[0]:training_data_time_range[1]] 
train_data_commodity = df_3.loc[training_data_time_range[0]:training_data_time_range[1]]
train_data_currency = df_4.loc[training_data_time_range[0]:training_data_time_range[1]]

test_data_stocks = df_1.loc[test_data_time_range[0]:test_data_time_range[1]]
test_data_crypto = df_2.loc[test_data_time_range[0]:test_data_time_range[1]]
test_data_commodity = df_3.loc[test_data_time_range[0]:test_data_time_range[1]]
test_data_currency = df_4.loc[test_data_time_range[0]:test_data_time_range[1]]


#print('Shape of training data for crypto:', training_data_crypto.shape)
#print('Shape of testing data for crypto:', test_data_crypto.shape)


