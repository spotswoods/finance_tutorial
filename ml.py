import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random



# Assuming you have a ReplayBuffer class defined


Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])


class StockTradingEnv(gym.Env):
    """A simple stock trading environment"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        
        self.df = df.reset_index()
        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(df.columns) - 1,), dtype=np.float16)
        
        self.current_step = 0
        self.done = False
        
        # Initialize the attributes for tracking trades and profit
        self.positions = []  # To track open positions
        self.total_profit = 0  # Total profit or loss
        self.trade_history = []  # To track the history of trades

    def reset(self):
        self.current_step = 0
        self.done = False
        self.positions = []
        self.profits = 0
        return self._next_observation()

    def _next_observation(self):
    # Include 'Trend' in observation vector
        relevant_features = ['Close', 'RSI', 'MACD', 'Signal_Line', 'Short_MAvg', 'Long_MAvg', 'Trend', 'Open', 'High', 'Low', 'Volume']  # Update as per your setup
        obs = self.df.loc[self.current_step, relevant_features].values
        
        # Convert observation to numpy array of type float32
        obs = np.array(obs, dtype=np.float32)
        
        return obs

    def step(self, action):
        self.current_step += 1
        self.peak = float('-inf')
        self.drawdown = 0
        
        if self.current_step >= len(self.df) - 1:
            self.done = True

        # Get the current price of the stock
        current_price = self.df.loc[self.current_step, 'Close']
        reward = 0
        profit_or_loss = 0
        transaction_cost = 0.01  # Example cost per trade
        if action == 1:  # Buy
            self.positions.append(current_price)
        elif action == 2:  # Attempt to Sell
            if self.positions:  # Sell with positions
                buy_price = self.positions.pop(0)
                profit_or_loss = current_price - buy_price - transaction_cost
                self.total_profit += profit_or_loss
            else:
                profit_or_loss = -0.001  # Penalty for trying to sell without positions

        self.peak = max(self.peak, self.total_profit)
    
    # Calculate drawdown
        self.drawdown = (self.peak - self.total_profit) / self.peak if self.peak > 0 else 0
        
        # Apply a drawdown penalty if the drawdown exceeds a threshold
        DRAWDOWN_THRESHOLD = 0.2  # Example threshold
        if self.drawdown > DRAWDOWN_THRESHOLD:
            reward = -self.drawdown  # Consider tuning this penalty
        # Adjust the reward based on drawdown
        
        
        # Calculate the reward
        reward = np.tanh(profit_or_loss) if action == 2 else np.tanh(-0.0001) if action == 0 else 0
        
        # Construct the next observation
        trade_outcome = 1 if profit_or_loss > 0 else 0
        self.trade_history.append((self.current_step, action, profit_or_loss, trade_outcome))


        obs = self._next_observation()

        return obs, reward, self.done, {}
    
    def render(self, mode='human', close=False):
        profit = self.profits
        print(f'Step: {self.current_step}, Profit: {profit}')

# Assume 'data' is the preprocessed DataFrame with stock data
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)  # internal memory (deque)
        self.experience = Experience
        self.positions = []  # To track open positions
        self.total_profit = 0  # Total profit or loss
        self.trade_history = []  # To track the history of trades

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class DQN(nn.Module):
    """Deep Q-Network"""
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),  # Ensure state_dim matches the environment's state size
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def fetch_data(ticker, period="5y"):
# Fetch historical data for AAPL
    ticker = "AAPL"
    data = yf.download(ticker, period="90d")
    data.sort_values(by='Date', inplace=True)
    
    return data
# Display the first few rows of the data
def calculate_moving_averages(data):
    short_window = 50
    long_window = 200
    
    data['Short_MAvg'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MAvg'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    
    # Determine trend direction: 1 for uptrend, -1 for downtrend, 0 for sideways
    data['Trend'] = 0  # Initialize as sideways
    data.loc[data['Short_MAvg'] > data['Long_MAvg'], 'Trend'] = 1  # Uptrend
    data.loc[data['Short_MAvg'] < data['Long_MAvg'], 'Trend'] = -1  # Downtrend
    
    # Optionally, define 'sideways' more strictly based on the difference between moving averages
    # This is an example and might need adjustments
    threshold = 0.01  # Define based on your criteria
    close_to_each_other = abs(data['Short_MAvg'] - data['Long_MAvg']) / data['Close'] <= threshold
    data.loc[close_to_each_other, 'Trend'] = 0  # Sideways
    
    return data

# Plot the closing prices
def preprocess_data(data):
    data['RSI'] = calculate_RSI(data)
    macd, signal = calculate_MACD(data)
    data = calculate_moving_averages(data)
    data['MACD'] = macd
    data['Signal_Line'] = signal
    data.dropna(inplace=True)
    
    features = ['Close', 'RSI', 'MACD', 'Signal_Line', 'Short_MAvg', 'Long_MAvg', 'Trend', 'Open', 'High', 'Low', 'Volume']
    
    # Initialize and apply the scaler
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])

    return data  # Ensure you return the modified DataFrame


# Let's drop NaN values that might have been introduced during calculation
 


def calculate_RSI(data, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    RS = gain / loss
    return 100 - (100 / (1 + RS))

def calculate_MACD(data, span1=12, span2=26, signal_span=9):
    """Calculate Moving Average Convergence Divergence (MACD)"""
    exp1 = data['Close'].ewm(span=span1, adjust=False).mean()
    exp2 = data['Close'].ewm(span=span2, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal_span, adjust=False).mean()
    
    return macd, signal_line

def plot_cumulative_profits(total_profits):
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(total_profits), label='Agent Cumulative Profit')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Profit')
    plt.title('Agent Cumulative Profits Over Episodes')
    plt.legend()
    plt.show()

def plot_trade_outcomes(trade_outcomes):
    wins = sum(trade_outcomes)
    losses = len(trade_outcomes) - wins

    plt.figure(figsize=(7, 4))
    plt.bar(['Wins', 'Losses'], [wins, losses], color=['green', 'red'])
    plt.title('Trade Outcomes')
    plt.show()

def train():
    # Parameters
    episodes = 500
    batch_size = 32
    gamma = 0.99  # Discount factor for future rewards
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_min = 0.001
    epsilon_decay_rate = 0.995
    epsilon = epsilon_start

    # Environment and Model setup
    data = fetch_data('SSAB-B.ST', '90d')
    print(data.head())  # Check the first few rows
    print(data.tail())  # Check the last few rows
    processed_data = preprocess_data(data)
    env = StockTradingEnv(processed_data)
    state_dim = len(['Close', 'RSI', 'MACD', 'Signal_Line', 'Short_MAvg', 'Long_MAvg', 'Trend', 'Open', 'High', 'Low', 'Volume'])  # Now 7
    action_dim = env.action_space.n
    model = DQN(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters())
    replay_buffer = ReplayBuffer(10000)
    total_profits = []  # Track total profits per episode
    trade_outcomes = []  # Track wins (1) and losses (0)
    for episode in range(episodes):
        state = env.reset()
        episode_profit = 0
        episode_trades = []
        epsilon = max(epsilon_min, epsilon * epsilon_decay_rate)    
        while True:
            # Epsilon-greedy action selection
            if random.random() > epsilon:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = model(state_tensor).argmax().item()
            else:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            

            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                # Unpack batched data
                # Compute loss and update model

            next_state, reward, done, info = env.step(action)  # Ensure 'info' contains trade outcome if applicable

            episode_profit += reward  # Assuming reward is profit from the step
            if 'trade_outcome' in info:
                episode_trades.append(info['trade_outcome'])
            if done:
                break
        
        epsilon = max(epsilon_final, epsilon_decay_rate * epsilon)  # Decrease epsilon
        total_profits.append(episode_profit)
        trade_outcomes.extend(episode_trades)  # Assuming binary outcomes for simplicity
        print(f'Episode: {episode+1}, Total Reward: {episode_profit}')
    return total_profits, trade_outcomes
    # Save model
    torch.save(model.state_dict(), 'dqn_model.pth')
if __name__ == "__main__":
    total_profits, trade_outcomes = train()
    plot_cumulative_profits(total_profits)
    plot_trade_outcomes(trade_outcomes)
    