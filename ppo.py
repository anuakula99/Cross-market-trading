from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import gymnasium


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import env_cr
from env_cr import TradingEnv_cr

import load_data
from load_data import train_data_stocks,train_data_crypto, train_data_commodity

class PPOAgent:
     def __init__(self, env, total_timesteps):
          check_env(env)
          self.model = PPO("MlpPolicy", env, verbose=1)
          self.model.learn(total_timesteps=total_timesteps)
          self.model.save('ppo agent_crypto1')
          
     def predict(self, obs):
          action, _ = self.model.predict(obs)
          return action
         
 
def create_env_and_train_agent(data, total_timesteps):
    
     env = TradingEnv_cr(data)
     
     ppo_agent = PPOAgent(env, total_timesteps)
     
     return env, ppo_agent
     
total_timesteps = 1000000
env, ppo_agent = create_env_and_train_agent(train_data_crypto, total_timesteps)

# After training is complete
env.plot_moving_average_rewards(window=10)
env.plot_worth()


