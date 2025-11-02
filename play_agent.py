import gymnasium as gym 
import torch
import yaml
import os
import time
from utils.video_utils import record_video
import glob

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# --- Configuration Settings ---
env_name = config['env_name']
# Added 'play_agent_type' to config for user control, defaults to 'ppo'
agent_name = config.get('play_agent_type', 'ppo').lower() 
device = config['device']
model_path = config.get('load_model_path', f'results/best_{agent_name}.pth')
render_mode = "human" # Display the environment window
record_video_flag = config.get('record_video', True) # Added flag for video recording

# --- Dynamic Import for Agent and Network ---
try:
    module = __import__(f'agents.{agent_name}_agent', fromlist=['*'])
    # Some agent files follow different naming conventions for network classes.
    # Provide a small mapping for known cases, otherwise fall back to <AGENT>Network.
    if agent_name == 'a2c':
        AgentClass = getattr(module, 'A2CAgent')
        NetworkClass = getattr(module, 'ActorCriticNetwork')
    else:
        AgentClass = getattr(module, agent_name.upper() + 'Agent')
        NetworkClass = getattr(module, agent_name.upper() + 'Network')
except (ImportError, AttributeError) as e:
    print(f"Error: Could not import {agent_name.upper()}Agent or Network. Check file names and class definitions.")
    print(f"Original error: {e}")
    exit()

# --- Environment Setup ---
try:
    env = gym.make(env_name, render_mode=render_mode)
except gym.error.VersionNotFound as e:
    print(f"Warning: Failed to load environment '{env_name}'. Trying version 'v2'...")
    if "v3" in env_name and "LunarLander" in env_name:
        env_name = env_name.replace("v3", "v2")
        env = gym.make(env_name, render_mode=render_mode)
    else:
        raise e

# --- Network and Agent Initialization ---
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
net_cfg = config.get(agent_name, {})

network = NetworkClass(obs_dim, action_dim, net_cfg.get('hidden_layers', [64, 64])) 
agent = AgentClass(env, network, device=device) 
agent.agent_name = agent_name 

# --- Load Model ---
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    env.close()
    exit()

print(f"Loading best model for {agent_name.upper()} from {model_path}")
agent.load_model(model_path)
agent.network.eval()

# --- Video Recording Wrapper (if enabled) ---
# --- Video Recording Wrapper (if enabled) ---
video_output_path = None # Initialize variable
if record_video_flag:
    # The record_video function from video_utils already returns a wrapped env and the path
    env_to_play, video_dir = record_video_wrapper(env, env_name, agent_name) # Call your wrapper
    # The actual mp4 file path will be inside video_dir, like video_dir/rl-video-episode-0.mp4
    # We need a way to find the actual .mp4 file. It's usually the first one created.
    import glob
    # We'll just grab the *first* video created for simplicity in app.py
    # You could also modify record_video_wrapper to return the exact mp4 path
    # or have play_agent.py loop through episodes and print each path.
else:
    env_to_play = env

# ... (rest of the play loop) ...

# --- ADD THIS AT THE VERY END OF play_agent.py ---
if record_video_flag and video_dir:
    # Wait a moment for the video file to be finalized
    time.sleep(1)
    mp4_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if mp4_files:
        print(f"VIDEO_PATH_FOR_STREAMLIT: {mp4_files[0]}") # Print path for Streamlit to capture

# --- Play Loop ---
episodes_to_play = 5 
print(f"Starting playback for {episodes_to_play} episodes on {env_name}...")

for ep in range(episodes_to_play):
    # Use the potentially wrapped environment
    state, _ = env_to_play.reset() 
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action = agent.act(state) 
        
        # Use the env_to_play step method
        next_state, reward, terminated, truncated, _ = env_to_play.step(action)
        done = terminated or truncated
        
        total_reward += reward
        state = next_state
        steps += 1
        
        time.sleep(0.01)

    print(f"Episode {ep + 1}: Reward = {total_reward:.2f}, Steps = {steps}")

env.close()
print("Playback complete.")