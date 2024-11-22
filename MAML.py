import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch.optim as optim

import load_data
from load_data import train_data_stocks, train_data_commodity, train_data_crypto

# Assuming these are your custom environments
import env_s, env_co, env_cr
from env_s import TradingEnv_s
from env_co import TradingEnv_co
from env_cr import TradingEnv_cr

# Create instances of your custom environments
env_1 = TradingEnv_s(train_data_stocks)  # stock environment
env_2 = TradingEnv_co(train_data_commodity)  # commodity environment
env_3 = TradingEnv_cr(train_data_crypto)  # crypto environment

# Wrap the environments in DummyVecEnv (Stable-Baselines3 requirement)
env_1 = DummyVecEnv([lambda: env_1])
env_2 = DummyVecEnv([lambda: env_2])
env_3 = DummyVecEnv([lambda: env_3])

# Normalize environments for better performance (Stable-Baselines3 requires this)
env_1 = VecNormalize(env_1, norm_obs=True, norm_reward=True)
env_2 = VecNormalize(env_2, norm_obs=True, norm_reward=True)
env_3 = VecNormalize(env_3, norm_obs=True, norm_reward=True)

# Meta-policy Network (PPO for meta-policy update)
class MetaPolicyNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MetaPolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, output_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Assuming action space is bounded between [-1, 1]

# Wrapper for PPO task-specific adaptation (using Stable-Baselines3)
class PPOAgent:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.model = PPO("MlpPolicy", env, verbose=1, device=device) 

    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)

    def get_action(self, state):
        action, _states = self.model.predict(state, deterministic=True)
        return action

    def get_log_prob(self, state, action):
        # Get the log probability of the action taken from the state
        # Extract log_prob from the action distribution
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.unsqueeze(0).to(self.device) 
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32).to(self.device)
        else:
            action = action.to(self.device)
        action_dist = self.model.policy.get_distribution(state)
        log_prob = action_dist.log_prob(action)
        return log_prob

# PPO with Clipping for Meta-Policy Update
def meta_ppo_update(meta_policy, old_meta_policy, trajectories, clip_epsilon=0.2):
    old_log_probs, rewards = zip(*trajectories)
    rewards = torch.tensor(rewards)
    log_probs = torch.stack(old_log_probs)

    # Calculate ratio of new to old policy probabilities
    ratio = torch.exp(log_probs - old_log_probs.detach())
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    # Calculate PPO loss and update meta-policy
    ppo_loss = -torch.min(ratio * rewards, clipped_ratio * rewards).mean()

    meta_optimizer.zero_grad()
    ppo_loss.backward()
    meta_optimizer.step()

    return ppo_loss

# Main training loop
def main():
    # Define input and output dimensions for meta-policy network
    input_dim = env_1.observation_space.shape[0]  # Assuming all environments have the same state space
    output_dim = env_1.action_space.shape[0]  # Assuming all environments have the same action space
    
    # Initialize meta-policy network and old meta-policy
    meta_policy = MetaPolicyNetwork(input_dim, output_dim)
    old_meta_policy = MetaPolicyNetwork(input_dim, output_dim)
    old_meta_policy.load_state_dict(meta_policy.state_dict())  # Initialize old meta-policy with current policy

    # Meta-policy optimizer
    meta_optimizer = torch.optim.Adam(meta_policy.parameters(), lr=0.001)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize PPO agent for task-specific adaptation
    task_agent_1 = PPOAgent(env_1,device)
    task_agent_2 = PPOAgent(env_2,device)
    task_agent_3 = PPOAgent(env_3,device)

    # Training loop
    n_iterations = 100
    for iteration in range(n_iterations):
        task_losses = []

        # Train task-specific policies using PPO (one for each environment)
        task_agent_1.train(total_timesteps=10000)
        task_agent_2.train(total_timesteps=10000)
        task_agent_3.train(total_timesteps=10000)

        # After task-specific update, collect experiences (for meta-policy update)
        trajectories_1 = []
        trajectories_2 = []
        trajectories_3 = []
        
        # Collect data for meta-policy update from each environment
        for _ in range(10):  # Number of episodes to collect for meta-policy update
            state = env_1.reset()[0]
            done = False
            log_probs = []
            rewards = []
            while not done:
                action = task_agent_1.get_action(state)
                log_prob = task_agent_1.get_log_prob(state, action)  # Use get_log_prob here
                log_probs.append(log_prob)
                state, reward, done, _ = env_1.step(action)
                rewards.append(reward)
            trajectories_1.append((log_probs, sum(rewards)))

        for _ in range(10):
            state = env_2.reset()[0]
            done = False
            log_probs = []
            rewards = []
            while not done:
                action = task_agent_2.get_action(state)
                log_prob = task_agent_2.get_log_prob(state, action)  # Use get_log_prob here
                log_probs.append(log_prob)
                state, reward, done,  _ = env_2.step(action)
                rewards.append(reward)
            trajectories_2.append((log_probs, sum(rewards)))

        for _ in range(10):
            state = env_3.reset()[0]
            done = False
            log_probs = []
            rewards = []
            while not done:
                action = task_agent_3.get_action(state)
                log_prob = task_agent_3.get_log_prob(state, action)  # Use get_log_prob here
                log_probs.append(log_prob)
                state, reward, done,  _ = env_3.step(action)
                rewards.append(reward)
            trajectories_3.append((log_probs, sum(rewards)))

        # Perform PPO update on meta-policy using collected trajectories
        all_trajectories = trajectories_1 + trajectories_2 + trajectories_3
        meta_loss = meta_ppo_update(meta_policy, old_meta_policy, all_trajectories)

        # Save meta-policy weights periodically
        if (iteration + 1) % 10 == 0:
            torch.save(meta_policy.state_dict(), f"meta_weights_iteration_{iteration + 1}.pth")
            print(f"Meta weights saved at iteration {iteration + 1}")

        print(f"Iteration {iteration + 1}/{n_iterations}, Meta Loss: {meta_loss.item()}")

if __name__ == "__main__":
    main()

