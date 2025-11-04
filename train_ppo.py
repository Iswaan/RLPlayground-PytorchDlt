import argparse
import gymnasium as gym
import torch
import yaml
import os
from agents.ppo_agent import PPONetwork, PPOAgent

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help='Run a fast debug training session')
parser.add_argument('--save_path', type=str, default=None, help='Directory to save results.') # ADDED
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

# ADDED: Environment parameter handling
env_kwargs = {}
if "LunarLander" in env_name:
    env_params = config.get('environment_params', {}).get(env_name, {})
    if 'gravity' in env_params:
        env_kwargs['gravity'] = env_params['gravity']
    if 'enable_wind' in env_params and env_params['enable_wind']:
        env_kwargs['enable_wind'] = True
        env_kwargs['wind_power'] = env_params.get('wind_power', 0.0)

# Environment Initialization
try:
    env = gym.make(env_name, **env_kwargs) # MODIFIED: Pass env_kwargs here
except Exception as e:
    print(f"Error initializing environment {env_name}: {e}")
    exit()

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Network config
net_cfg = config.get(agent_name, {})
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
                 save_path=args.save_path or 'results/', # MODIFIED: Use arg or fallback
                 config=config)

if args.debug:
    print("DEBUG mode enabled: reducing n_steps/n_epochs and train_episodes for fast runs")
    agent.n_steps = max(64, getattr(agent, 'n_steps', 2048) // 8)
    agent.n_epochs = max(1, getattr(agent, 'n_epochs', 10) // 5)
    train_episodes = min(train_episodes, 20)

# Train agent
agent.train(episodes=train_episodes)