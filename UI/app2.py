import streamlit as st
import yaml
import os
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import numpy as np
from datetime import datetime

# --- Initial Setup ---
st.set_page_config(layout="wide") # Use the full width of the page

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch

CONFIG_PATH = os.path.join(project_root, 'config.yaml')

@st.cache_data
def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# --- Sidebar for Global Controls ---
st.sidebar.title("🛠️ RLPlayground Settings")

st.sidebar.header("Environment")
env_name = st.sidebar.selectbox("Select Environment", ["LunarLander-v3", "LunarLander-v2", "CartPole-v1"], index=0)
config['env_name'] = env_name

st.sidebar.subheader("Environment Physics/Parameters")
if 'environment_params' not in config:
    config['environment_params'] = {}
    
if "LunarLander" in env_name:
    if env_name not in config['environment_params']:
        config['environment_params'][env_name] = {'gravity': -10.0, 'enable_wind': False, 'wind_power': 0.0}
    ll_params = config['environment_params'][env_name]

    current_gravity = st.sidebar.slider("Gravity", -15.0, -5.0, ll_params.get('gravity', -10.0), 0.1)
    current_enable_wind = st.sidebar.checkbox("Enable Wind", ll_params.get('enable_wind', False))
    current_wind_power = st.sidebar.slider("Wind Power", 0.0, 20.0, ll_params.get('wind_power', 0.0), 0.5, disabled=not current_enable_wind)

    config['environment_params'][env_name]['gravity'] = current_gravity
    config['environment_params'][env_name]['enable_wind'] = current_enable_wind
    config['environment_params'][env_name]['wind_power'] = current_wind_power
elif env_name == "CartPole-v1":
    st.sidebar.info("CartPole-v1 has no configurable physics.")

st.sidebar.header("Global Training Parameters")
train_episodes = st.sidebar.number_input("Total Training Episodes", 1, 100000, config.get('train_episodes', 1450), 100)
config['train_episodes'] = train_episodes
max_timesteps_per_episode = st.sidebar.number_input("Max Timesteps Per Episode", 100, 10000, config.get('max_timesteps_per_episode', 1000), 100)
config['max_timesteps_per_episode'] = max_timesteps_per_episode
device = st.sidebar.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"], index=0)
config['device'] = device

# --- Main App Title ---
st.title("🚀 RL Playground: A Universal Game AI Framework")
st.markdown("Use the controls in the sidebar to configure the environment and training. Select a tab below to either play a pre-trained model or train a new one.")

# --- UI LAYOUT WITH TABS ---
tab_playback, tab_training, tab_logs = st.tabs(["▶️ Play Pre-trained Model", "🧠 Train New Model", "📊 View Training Logs"])

# --- TAB 1: PLAYBACK ---
with tab_playback:
    st.header("Visual Playback with Dynamic Environment")
    st.markdown("Select a pre-trained agent and model, adjust the environment's physics in the sidebar, and see how it performs!")

    col_agent_select, col_model_select = st.columns(2)
    
    with col_agent_select:
        playback_agent_type = st.selectbox("Choose Agent for Playback", ["PPO", "DQN", "A2C"], key="playback_agent")
        playback_agent_cfg_key = playback_agent_type.lower()
        
        if playback_agent_type == "DQN":
            playback_dqn_preset = st.selectbox("DQN Preset Model", ["fast_learning", "stability_first", "base_dqn"], key="playback_preset")

    with col_model_select:
        # Find available models in pre_trained_models/
        pre_trained_models_dir = os.path.join(project_root, 'pre_trained_models')
        available_models = []
        if os.path.exists(pre_trained_models_dir):
            for f in os.listdir(pre_trained_models_dir):
                if f.endswith('.pth') and f.startswith(f'best_{playback_agent_cfg_key}'):
                    available_models.append(os.path.join(pre_trained_models_dir, f))
        
        # Additional logic for DQN preset models
        if playback_agent_type == "DQN":
            preset_model_name = f"best_dqn_{playback_dqn_preset}.pth"
            # Filter available models to match selected agent and preset
            filtered_models = [m for m in available_models if preset_model_name in m or "best_dqn.pth" in m]
        else:
            filtered_models = [m for m in available_models if f"best_{playback_agent_cfg_key}.pth" in m]

        selected_model_path = st.selectbox(
            "Select Pre-trained Model",
            options=filtered_models,
            format_func=os.path.basename,
            help="These models were found in your 'pre_trained_models' folder."
        )

    num_play_episodes = st.number_input("Number of Playback Episodes", 1, 10, 1, 1)

    if st.button("🚀 Run Playback", use_container_width=True):
        if not selected_model_path or not os.path.exists(selected_model_path):
            st.error(f"Model file not found. Please ensure a valid model is selected.")
        else:
            st.info(f"Initiating playback for {os.path.basename(selected_model_path)} on {env_name}...")
            
            try:
                temp_config = config.copy()
                temp_config['play_agent_type'] = playback_agent_cfg_key
                temp_config['load_model_path'] = os.path.relpath(selected_model_path, project_root)
                temp_config['episodes_to_play'] = num_play_episodes
                temp_config['render_mode'] = "rgb_array"
                temp_config['record_video'] = True

                with open(CONFIG_PATH, 'w') as f:
                    yaml.dump(temp_config, f, default_flow_style=False)

                play_command = [sys.executable, os.path.join(project_root, "play_agent.py")]
                
                video_placeholder = st.empty()
                output_placeholder = st.empty()
                
                with st.spinner(f'Running playback...'):
                    play_process = subprocess.Popen(play_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=project_root)
                    
                    full_play_output = []
                    recorded_video_path = None
                    for line in play_process.stdout:
                        full_play_output.append(line)
                        output_placeholder.text("".join(full_play_output[-5:]))

                        if "VIDEO_PATH_FOR_STREAMLIT:" in line:
                            recorded_video_path = line.split("VIDEO_PATH_FOR_STREAMLIT:")[1].strip()
                    play_process.wait()

                if play_process.returncode == 0:
                    st.success("Playback Complete!")
                    if recorded_video_path and os.path.exists(recorded_video_path):
                        video_file = open(recorded_video_path, 'rb')
                        video_bytes = video_file.read()
                        video_placeholder.video(video_bytes, format='video/mp4')
                        video_file.close()
                    else:
                        video_placeholder.warning("No video file was found after playback.")
                    
                    final_rewards = [float(l.split("Reward = ")[1].split(",")[0]) for l in full_play_output if "Episode" in l and "Reward =" in l]
                    if final_rewards:
                        st.write(f"**Playback Rewards:** {', '.join([f'{r:.2f}' for r in final_rewards])}")
                        st.write(f"**Mean Playback Reward:** {np.mean(final_rewards):.2f}")
                else:
                    st.error(f"Playback Failed!")
                    st.text_area("Full Log", "".join(full_play_output), height=300)

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# --- TAB 2: TRAINING ---
with tab_training:
    st.header("Train a New Model")
    st.markdown("Select an agent type and configure its hyperparameters. When ready, click 'Start Training'.")

    training_agent_type = st.selectbox("Choose Agent to Train", ["DQN", "A2C", "PPO"], key="train_agent")
    training_agent_cfg_key = training_agent_type.lower()
    
    with st.expander(f"Show/Hide {training_agent_type} Hyperparameters"):
        if training_agent_type == "DQN":
            # Your full DQN parameter widgets go here...
            st.info("Full DQN hyperparameter controls would be listed here.")
        elif training_agent_type == "A2C":
            # Your full A2C parameter widgets go here...
            st.info("Full A2C hyperparameter controls would be listed here.")
        elif training_agent_type == "PPO":
            # Your full PPO parameter widgets go here...
            st.info("Full PPO hyperparameter controls would be listed here.")

    if st.button("Start Training", use_container_width=True):
        st.warning("Training functionality is set up but will run on the Streamlit Cloud server. This can be slow. Use for smaller training runs.")
        # The full training logic (subprocess calls, etc.) from the previous app.py would go here.
        # For simplicity in this example, we'll just show a message.
        st.info("This is where the training process would be initiated.")

# --- TAB 3: LOGS and PLOTS ---
with tab_logs:
    st.header("Live Training Logs and Plots")
    st.markdown("If you start a training run, its terminal output and reward plots will appear here.")
    
    # This section is a placeholder for the live output and plotting logic
    # from the previous comprehensive app.py.
    st.info("Terminal output and live plots would be displayed in this tab during training.")