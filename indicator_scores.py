import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm.notebook import tqdm
import itertools

def calculate_scores(df):
    # Calculate scores for each indicator
    df['MACD_score'] = np.tanh(df['MACD_diff'] / df['MACD_diff'].std())  # Normalize using tanh to bound between -1 and 1
    df['RSI_score'] = df['RSI'].apply(lambda x: -np.tanh((x-50)/10))  # Shift and scale RSI scores
    df['Volume_score'] = np.tanh((df['Volume'] / df['Volume_avg']) - 1)  # Volume compared to average
    df['EMA200_score'] = np.tanh((df['Close'] - df['EMA200']) / df['EMA200'].std())  # Distance from EMA200
    
    # Aggregate scores
    df['Aggregated_Score'] = df[['MACD_score', 'RSI_score', 'Volume_score', 'EMA200_score']].mean(axis=1)
    
    return df

def download_stock_data(symbols, start_date='2010-01-01'):
    stock_data = {}
    for symbol in symbols:
        df = yf.download(symbol, start=start_date)
        stock_data[symbol] = df
    return stock_data

def backtest(df, buy_threshold, sell_threshold):
    cash = 0
    position = 0
    buy_price = 0
    
    for index, row in df.iterrows():
        if row['Aggregated_Score'] > buy_threshold and position == 0:
            position = 1
            buy_price = row['Close']
        elif row['Aggregated_Score'] < sell_threshold and position == 1:
            cash += row['Close'] - buy_price
            position = 0
    
    if position == 1:
        cash += df.iloc[-1]['Close'] - buy_price
    return cash

def test_multiple_stocks(stock_data, buy_threshold, sell_threshold):
    results = []
    for symbol, df in stock_data.items():
        pnl = backtest(df, buy_threshold, sell_threshold)
        results.append(pnl)
    return sum(results) / len(results)  # Return average P&L





# Calculate MACD
def prepare_data(df):
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_diff'] = df['MACD'] - df['Signal']

    # Calculate RSI
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Analyze Volume
    df['Volume_avg'] = df['Volume'].rolling(window=20).mean()
    df['Volume_score'] = df['Volume'] / df['Volume_avg']


    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # Bollinger Bands
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA20'] + (df['STD20'] * 2)
    df['Lower_Band'] = df['SMA20'] - (df['STD20'] * 2)

    # Stochastic Oscillator
    low_14, high_14 = df['Low'].rolling(window=14).min(), df['High'].rolling(window=14).max()
    df['Stochastic'] = ((df['Close'] - low_14) / (high_14 - low_14)) * 100

    # ATR
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = np.abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    # Scoring example (customize as needed)

    df = calculate_scores(df)
    return df


symbols = ['AAPL', 'MSFT', 'GOOGL']
# Load data
stock_data = download_stock_data(symbols)


for symbol in symbols:
    stock_data[symbol] = prepare_data(stock_data[symbol])

# plt.figure(figsize=(14, 7))
# plt.xlabel('Date')
# plt.ylabel('Close Price')
# plt.title('Stock Price with Highlighted Buy Signals')
# plt.grid(True)

# # Plot the closing price
# plt.plot(df.index, df['Close'], label='Close Price', color='skyblue')

# # Highlight points where the aggregated score is above 0.15
# # We'll use red dots to mark these points on the plot
# buy_signals = df[df['Aggregated_Score'] > 0.15]
# plt.scatter(buy_signals.index, buy_signals['Close'], color='green', label='Buy Signal', marker='^', alpha=1)

# # Highlight sell signals where the aggregated score is below -0.15
# # Use red down markers for sell signals
# sell_signals = df[df['Aggregated_Score'] < -0.15]
# plt.scatter(sell_signals.index, sell_signals['Close'], color='red', label='Sell Signal', marker='v', alpha=1)

# # Improve readability of the x-axis dates
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.gcf().autofmt_xdate()  # Rotation

# # Show legend
# plt.legend(

# plt.show()

buy_thresholds = np.arange(0.05, 0.3, 0.05)
sell_thresholds = np.arange(-0.3, -0.05, 0.05)

results = []
for buy_threshold, sell_threshold in tqdm(itertools.product(buy_thresholds, sell_thresholds), desc='Optimizing'):
    avg_pnl = test_multiple_stocks(stock_data, buy_threshold, sell_threshold)
    results.append(((buy_threshold, sell_threshold), avg_pnl))

best_thresholds, best_avg_pnl = max(results, key=lambda x: x[1])
print(f"Optimal Buy Threshold: {best_thresholds[0]}, Optimal Sell Threshold: {best_thresholds[1]}, Maximum Average P&L: {best_avg_pnl}")