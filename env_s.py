import gymnasium 
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TradingEnv_s(gymnasium.Env):
    # ... existing code ...
    
    def __init__(self, market_data, risk_free_rate=0.0):
        super(TradingEnv_s, self).__init__()
        
        
        self.market_data = market_data
        self.risk_free_rate = risk_free_rate  # Risk-free rate for Sharpe Ratio
        self.n_features = len(self.market_data.columns)
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.obs_shape = self.n_features + 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float64)
        
        # Initialize account values
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = 0
        self.max_steps = len(self.market_data) - 1
        self.episode_rewards = []
        self.daily_returns = []
        self.wealth = []
        self.money= []
        
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = 0
        self.total_reward = 0
       
        self.daily_returns = []
        
        self.money = []
       
        return self._next_observation(), {}
        
    def _next_observation(self):  
        frame = np.zeros(self.obs_shape)
        idx = 0
        if self.current_step < len(self.market_data)-1:
           frame[idx:idx+self.n_features] = self.market_data.iloc[self.current_step].values
        elif len(self.market_data) > 0:
           frame[idx:idx+self.n_features] = self.market_data.iloc[-1].values
           
        frame[-4] = self.balance
        frame[-3] = self.net_worth
        frame[-2] = self.max_net_worth
        frame[-1] = self.shares_held
        
        return frame  
    
    def step(self, action):
        self.current_step += 1
        action = action
        if isinstance(action, tuple):
           action = action[0]
        
     
        
        current_price = self.market_data.iloc[self.current_step]['Close']
        
        # Execute buy or sell action
        if action > 0:  # Buy
            shares_to_buy = int(self.balance * action / current_price)
            cost = shares_to_buy * current_price
            self.balance -= cost
            self.shares_held += shares_to_buy
        elif action < 0:  # Sell
            shares_to_sell = int(self.shares_held * abs(action))
            sale = shares_to_sell * current_price
            self.balance += sale
            self.shares_held -= shares_to_sell
            self.total_shares_sold += shares_to_sell
            self.total_sales_value += sale
        
        # Update net worth
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        self.max_net_worth = max(self.net_worth, self.max_net_worth)
        
        # Calculate daily return and add to the list
        daily_return = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth > 0 else 0
        self.daily_returns.append(daily_return)
        self.money.append(self.net_worth)
        
        # Calculate Profitability (normalized) and Sharpe Ratio (normalized)
        profitability_reward = (self.net_worth - self.initial_balance) / self.initial_balance
        sharpe_ratio_reward = self.calculate_sharpe_ratio()
        
        # Combined reward with 50-50 weightage
        reward = 0.5 * profitability_reward + 0.5 * sharpe_ratio_reward
        reward = reward *10
        self.total_reward += reward 
        
        
        # Determine if the episode is done
        done = self.net_worth <= 0 or self.current_step >= self.max_steps
        obs = self._next_observation()
        
        if done == True:
           self.episode_rewards.append(self.total_reward)
           self.wealth.append(self.net_worth)
        
        return obs, reward, done, False, {}
    
    def calculate_sharpe_ratio(self):
        """Calculates the Sharpe Ratio of the daily returns."""
        if len(self.daily_returns) < 2:  # Not enough data to calculate volatility
            return 0
        
        avg_return = np.mean(self.daily_returns)
        std_dev_return = np.std(self.daily_returns)
        
        # Sharpe Ratio = (Mean Return - Risk-Free Rate) / Std Dev of Return
        sharpe_ratio = (avg_return - self.risk_free_rate) / std_dev_return if std_dev_return > 0 else 0
        return sharpe_ratio
    
    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance:.2f}')
        print(f'Shares held: {self.shares_held}')
        print(f'Net worth: {self.net_worth:.2f}')
        print(f'Profit: {profit:.2f}')
    
    def plot_moving_average_rewards(self, window=10):
        moving_avg = pd.Series(self.episode_rewards).rolling(window=window).mean()
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards, label="Episode Reward")
        plt.plot(moving_avg, label=f"{window}-Episode Moving Average", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward per Episode with Moving Average")
        plt.legend()
        plt.savefig("train_reward.png")
        
    def plot_worth(self, window=10):
        moving_avg = pd.Series(self.wealth).rolling(window=window).mean()
        plt.figure(figsize=(10, 6))
        plt.plot(moving_avg, label=f"{window}-Episode Moving Average", color="orange")
        plt.plot(self.wealth)
        plt.xlabel("Episode")
        plt.ylabel("net_worth")
        plt.title("Wealth accumulated")
        
        plt.savefig("net_worth.png")
        
    def portfolio(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.money)
        plt.xlabel("Timestep")
        plt.ylabel("net_worth")
        plt.title("Wealth accumulated")
  
        plt.savefig("portfolio.png")
    
    def close(self):
        pass

