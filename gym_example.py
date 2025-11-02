import gymnasium as gym
import yaml
import os

# Load config
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Config file not found, using default environment")
    config = {'env_name': 'LunarLander-v2'}

env_name = config.get('env_name', 'LunarLander-v2')
env = gym.make(env_name)

print(f"Environment: {env_name}")
print(f"Action space: {env.action_space}")