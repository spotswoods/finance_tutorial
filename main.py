import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

def get_rolling_mean(stock_data, window):
    return stock_data['Close'].rolling(window).mean()

def get_rolling_std(stock_data, window):
    return stock_data['Close'].rolling(window).std()

def get_bollinger_bands(stock_data_rm, stock_data_rstd):
    upper_band = stock_data_rm + stock_data_rstd*2
    lower_band = stock_data_rm - stock_data_rstd*2
    return upper_band, lower_band


def compute_daily_returns(stock_data):
    stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
    
    daily_returns = stock_data.copy()
    daily_returns = stock_data['Close'].pct_change()
    # Use .iloc for integer-location based indexing
    daily_returns.iloc[0] = 0
    return daily_returns


def test_run():
    end_date = '2023-12-31'
    start_date = '2023-01-01'
    tickers = 'SSAB-B.ST'
    #Read data
    df = yf.download(tickers, start=start_date, end=end_date)
    
    daily_returns = compute_daily_returns(df)
    
    
    daily_returns.hist(bins=20)
    plt.plot()
    
if __name__ == '__main__':
    test_run()
    
# Daily returns 
# daily_ret[Today] =
# (price[today] / price[today-1]) - 1