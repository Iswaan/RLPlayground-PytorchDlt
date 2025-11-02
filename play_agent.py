import gymnasium as gym
import torch
import yaml
import os
import time
import glob
from utils.video_utils import record_video_wrapper # Ensure this matches your utils/video_utils.py

# Load config from the project root (assuming play_agent.py is in project root)
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

# --- Configuration Settings ---
env_name = config['env_name']
agent_name = config.get('play_agent_type', 'ppo').lower()
device = config['device']
model_path = config.get('load_model_path', f'results/best_{agent_name}.pth') # Default still points to results/
                                                                                # UI will provide path from pre_trained_models/

# IMPORTANT: render_mode and record_video_flag are now controlled by the Streamlit app via config
render_mode = config.get('render_mode', "human")
record_video_flag = config.get('record_video', False) # Default to False, Streamlit enables

# --- Dynamic Import for Agent and Network ---
try:
    module = __import__(f'agents.{agent_name}_agent', fromlist=['*'])
    if agent_name == 'a2c':
        AgentClass = getattr(module, 'A2CAgent')
        NetworkClass = getattr(module, 'ActorCriticNetwork')
    else: # For PPO and DQN
        AgentClass = getattr(module, agent_name.upper() + 'Agent')
        NetworkClass = getattr(module, agent_name.upper() + 'Network')
except (ImportError, AttributeError) as e:
    print(f"Error: Could not import {agent_name.upper()}Agent or Network. Check file names and class definitions.")
    print(f"Original error: {e}")
    exit()

# --- Environment Setup - Now handling dynamic parameters ---
env_kwargs = {}
if "LunarLander" in env_name: # Check if it's a LunarLander environment
    env_params = config.get('environment_params', {}).get(env_name, {})
    if 'gravity' in env_params:
        env_kwargs['gravity'] = env_params['gravity']
    if 'enable_wind' in env_params and env_params['enable_wind']:
        env_kwargs['enable_wind'] = True
        env_kwargs['wind_power'] = env_params.get('wind_power', 0.0)
    # Note: LunarLander-v2 and -v3 natively support these kwargs.
    # If using a custom environment that wraps LunarLander, ensure its __init__
    # method accepts and passes these kwargs.

try:
    # Pass render_mode and environment-specific kwargs to gym.make
    env = gym.make(env_name, render_mode=render_mode, **env_kwargs)
except gym.error.VersionNotFound as e:
    print(f"Warning: Failed to load environment '{env_name}'. Trying version 'v2'...")
    if "v3" in env_name and "LunarLander" in env_name:
        env_name = env_name.replace("v3", "v2")
        env = gym.make(env_name, render_mode=render_mode, **env_kwargs)
    else:
        raise e

# --- Network and Agent Initialization ---
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
net_cfg = config.get(agent_name, {})

# Pass hidden_layers if defined in config, otherwise let NetworkClass use its default
hidden_layers_cfg = net_cfg.get('hidden_layers')
if hidden_layers_cfg:
    network = NetworkClass(obs_dim, action_dim, hidden_layers_cfg)
else:
    network = NetworkClass(obs_dim, action_dim)

agent = AgentClass(env, network, device=device)
agent.agent_name = agent_name

# --- Load Model ---
# Model path is expected to be relative to the project root for play_agent.py
# Streamlit app should provide this path correctly.
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    env.close()
    exit()

print(f"Loading best model for {agent_name.upper()} from {model_path}")
agent.load_model(model_path)
agent.network.eval() # Set to evaluation mode

# --- Video Recording Wrapper ---
video_dir = None
episodes_to_play_count = config.get('episodes_to_play', 1) # Control number of episodes to record
if record_video_flag:
    # The record_video_wrapper function will return the wrapped environment and the directory where videos are saved
    env_to_play, video_dir = record_video_wrapper(env, env_name, agent_name, episodes_to_record=episodes_to_play_count)
else:
    env_to_play = env

# --- Play Loop ---
print(f"Starting playback for {episodes_to_play_count} episodes on {env_name}...")

for ep in range(episodes_to_play_count):
    state, _ = env_to_play.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = agent.act(state) # Agent acts deterministically because network.eval() is set

        next_state, reward, terminated, truncated, _ = env.step(action) # MODIFIED: Use original 'env' here for consistency, or env_to_play.step() if wrapper handles it
        done = terminated or truncated # Episode ends if terminated OR truncated

        total_reward += reward
        state = next_state
        steps += 1

        # Small delay for visual effect if running locally with human render_mode
        if render_mode == "human":
            time.sleep(0.01)

    print(f"Episode {ep + 1}: Reward = {total_reward:.2f}, Steps = {steps}")

env.close() # Close the environment
print("Playback complete.")

# --- Crucial: Print the path to the recorded video for Streamlit to capture ---
if record_video_flag and video_dir:
    time.sleep(1) # Give the system a moment to finalize the video file
    mp4_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if mp4_files:
        print(f"VIDEO_PATH_FOR_STREAMLIT: {mp4_files[0]}")