import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


import trading_env
from trading_env import StockTradingEnv

# Load historical stock data (e.g., S&P 500) from Yahoo Finance
data = pd.read_csv('sp500_data.csv')  # Replace with your actual file path

# Ensure the data is in the correct format and sorted by date
data = data.sort_values('Date').reset_index(drop=True)

# Set up the Gymnasium environment
env = StockTradingEnv(data)

check_env(env)

print("Environment is valid!")


device = "cuda" 

# Set up the PPO agent
model = PPO("MlpPolicy", env, verbose=1, device=device)

# Train the agent
model.learn(total_timesteps=400000)

# Save the trained model
model.save("ppo_stock_trading.zip")

Episode_rewards = env.epi_rewards
plt.figure(figsize=(9,6))
running_avg = np.empty(len(Episode_rewards))
for t in range(len(Episode_rewards)):
        running_avg[t] = np.mean(Episode_rewards[max(0, t-100):(t+1)])
plt.plot(Episode_rewards,color="teal",alpha=0.2)
plt.plot(running_avg,color="teal",linewidth=2.0)
plt.ylabel("Reward Units")
plt.xlabel("No. of Episode")
plt.title("Proximal policy optimization (PPO)")
plt.grid()
plt.legend()
plt.savefig('rewards')


