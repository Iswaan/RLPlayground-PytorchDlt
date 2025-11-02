import gym
import torch
from agents.a2c_agent import ActorCriticNetwork, A2CAgent
from agents.ppo_agent import PPONetwork, PPOAgent
from agents.dqn_agent import DQNNetwork, DQNAgent
from utils.plot_utils import plot_rewards
import json
import os

# ---- Configuration ----
env_name = 'LunarLander-v3'
episodes = 500
device = 'cpu'  # Change to 'cuda' if GPU available
os.makedirs('models', exist_ok=True)

# ---- Initialize environment ----
env = gym.make(env_name)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# ---- Define agents ----
agents = {
    'A2C': A2CAgent(env, ActorCriticNetwork(input_dim, output_dim, hidden_layers=[128,128]), device=device, save_path='models/'),
    'PPO': PPOAgent(env, PPONetwork(input_dim, output_dim, hidden_layers=[128,128]), device=device, save_path='models/'),
    'DQN': DQNAgent(env, DQNNetwork(input_dim, output_dim, hidden_layers=[128,128]), device=device, save_path='models/')
}

best_agent_name = None
best_agent_reward = -float('inf')

# ---- Train each agent ----
for name, agent in agents.items():
    print(f"\n==== Training {name} ====")
    agent.train(episodes=episodes)
    avg_reward = sum(agent.episode_rewards)/len(agent.episode_rewards)
    print(f"Average reward for {name}: {avg_reward:.2f}")

    # Track the overall best agent
    max_reward = max(agent.episode_rewards)
    if max_reward > best_agent_reward:
        best_agent_reward = max_reward
        best_agent_name = name
        best_agent_file = f"models/best_{name}.pth"

print("\n==== Training Completed ====")
print(f"Best agent overall: {best_agent_name} with reward {best_agent_reward}")

# ---- Save best agent info ----
best_info = {
    'name': best_agent_name,
    'reward': best_agent_reward,
    'model_file': best_agent_file
}
with open('models/best_agent.json', 'w') as f:
    json.dump(best_info, f)

# ---- Plot comparison of rewards ----
for name, agent in agents.items():
    plot_rewards(agent.episode_rewards, f"{name}_rewards.png")
