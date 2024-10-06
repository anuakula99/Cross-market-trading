import gymnasium as gym
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
  def __init__(self, df):
    super(StockTradingEnv, self).__init__()
    self.df = df
    self.current_step = 0
    self.initial_balance = 10000
    self.balance = self.initial_balance
    self.shares_hold = 0
    self.net_worth = self.initial_balance
    self.episode = 0
    self.epi_rewards = []

    # Action space: [0: Hold, 1: Buy, 2: Sell]
    self.action_space = gym.spaces.Discrete(3)

    # Observation space: Prices, Technical Indicators, etc.
    self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float64)

  def reset(self, seed=None):
    self.episode +=1
    self.total_reward = 0
    self.current_step = 0
    self.balance = self.initial_balance
    self.shares_held = 0
    self.net_worth = self.initial_balance
    # Return the initial observation and an empty info dictionary
    return self._next_observation(), {}

  def _next_observation(self):
    obs = np.array([
        self.df.iloc[self.current_step]['Open'],
        self.df.iloc[self.current_step]['High'],
        self.df.iloc[self.current_step]['Low'],
        self.df.iloc[self.current_step]['Close'],
        self.df.iloc[self.current_step]['Volume']
    ])
    return obs

  def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']

     # Perform action
        if action == 1:  # Buy
            self.shares_held = self.balance / current_price
            self.balance = 0
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0
        
        # Update net worth
        self.net_worth = self.balance + self.shares_held * current_price

        # Reward is the change in portfolio value
        self.reward = self.net_worth - self.initial_balance
        self.total_reward += self.reward
        

        self.current_step += 1

        self.done = self.current_step >= len(self.df) - 1
        self.truncated = False  # Not handling truncation here
        
        if self.done == True or self.truncated == True:
          
           self.epi_rewards.append(self.total_reward)

           if self.episode==1 or self.episode % 50 == 0:
               StockTradingEnv.render(self)

         # Return next observation, reward, done, truncated, and an empty info dictionary
        return self._next_observation(), self.reward, self.done, self.truncated, {}

        
  def render(self, mode='human'):
     print(f'Episode: {self.episode},Return in episode: {self.total_reward}')
     
    
  def close(self):
        pass

        








