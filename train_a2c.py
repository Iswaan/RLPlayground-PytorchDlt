import gymnasium as gym
import torch
import yaml
from agents.a2c_agent import A2CAgent, ActorCriticNetwork

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

env_name = config.get('env_name', 'LunarLander-v3')
env = gym.make(env_name)

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# Use defaults if a2c block is missing
a2c_cfg = config.get('a2c', {}) or {}
hidden_layers = a2c_cfg.get('hidden_layers', [128, 128])
lr = float(a2c_cfg.get('lr', 7e-4))
gamma = float(a2c_cfg.get('gamma', 0.99))
activation = getattr(torch.nn, a2c_cfg.get('activation', 'Tanh'))

network = ActorCriticNetwork(input_dim, output_dim, hidden_layers, activation)

# Create the agent instance
agent = A2CAgent(
    env,
    network,
    lr=lr,
    gamma=gamma,
    device=config.get('device', 'cpu'),
    save_path='results/',
    config=config # <--- THIS LINE WAS MISSING. Pass the config dictionary here.
)

# Train the agent
print(f"--- Starting A2C Training (Episodes: {config.get('train_episodes', 500)}) ---")
agent.train(episodes=config.get('train_episodes', 500))
print("--- A2C Training Complete ---")