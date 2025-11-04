import gymnasium as gym
import torch
import yaml
from agents.a2c_agent import A2CAgent, ActorCriticNetwork
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default=None, help='Directory to save results.')
args = parser.parse_args()

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

env_name = config.get('env_name', 'LunarLander-v3')

# ADDED: Environment parameter handling
env_kwargs = {}
if "LunarLander" in env_name:
    env_params = config.get('environment_params', {}).get(env_name, {})
    if 'gravity' in env_params:
        env_kwargs['gravity'] = env_params['gravity']
    if 'enable_wind' in env_params and env_params['enable_wind']:
        env_kwargs['enable_wind'] = True
        env_kwargs['wind_power'] = env_params.get('wind_power', 0.0)

env = gym.make(env_name, **env_kwargs) # Pass env_kwargs here

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

a2c_cfg = config.get('a2c', {}) or {}
hidden_layers = a2c_cfg.get('hidden_layers', [128, 128])
lr = float(a2c_cfg.get('lr', 7e-4))
gamma = float(a2c_cfg.get('gamma', 0.99))
activation = getattr(torch.nn, a2c_cfg.get('activation', 'Tanh'))

network = ActorCriticNetwork(input_dim, output_dim, hidden_layers, activation)

agent = A2CAgent(
    env=env,
    network=network,
    lr=lr,
    gamma=gamma,
    device=config.get('device', 'cpu'),
    save_path=args.save_path or 'results/',
    config=config
)

print(f"--- Starting A2C Training (Episodes: {config.get('train_episodes', 500)}) ---")
agent.train(episodes=config.get('train_episodes', 500))
print("--- A2C Training Complete ---")