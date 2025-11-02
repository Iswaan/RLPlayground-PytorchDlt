import argparse
import gymnasium as gym
import torch
import yaml
import os
from agents.ppo_agent import PPONetwork, PPOAgent

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help='Run a fast debug training session')
args = parser.parse_args()

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

agent_name = 'ppo'
env_name = config.get('env_name', 'LunarLander-v3')
train_episodes = config.get('train_episodes', 2000)

# Safe device selection
device = config['device']
if device == 'cuda' and not torch.cuda.is_available():
    print("CUDA requested but not available. Falling back to CPU.")
    device = 'cpu'

print(f"Starting {agent_name.upper()} training on device: {device}")

# --- Environment Initialization ---
try:
    env = gym.make(env_name)
except Exception as e:
    print(f"Error initializing environment {env_name}: {e}")
    exit()

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Network config
net_cfg = config.get(agent_name, {})
# defaults (cast numeric values to proper types)
hidden_layers = net_cfg.get('hidden_layers', [128, 128])
lr = float(net_cfg.get('lr', 3e-4))
gamma = float(net_cfg.get('gamma', 0.99))
clip_epsilon = float(net_cfg.get('clip_epsilon', 0.2))
activation = getattr(torch.nn, net_cfg.get('activation', 'ReLU'))
network = PPONetwork(obs_dim, action_dim, hidden_layers, activation)

# Initialize agent
agent = PPOAgent(env, network,
                 lr=lr,
                 gamma=gamma,
                 clip_epsilon=clip_epsilon,
                 device=device,
                 save_path='results/',
                 config=config)

if args.debug:
    print("DEBUG mode enabled: reducing n_steps/n_epochs and train_episodes for fast runs")
    # reduce workload for faster iteration
    agent.n_steps = max(64, getattr(agent, 'n_steps', 2048) // 8)
    agent.n_epochs = max(1, getattr(agent, 'n_epochs', 10) // 5)
    train_episodes = min(train_episodes, 20)

# Train agent
agent.train(episodes=train_episodes)