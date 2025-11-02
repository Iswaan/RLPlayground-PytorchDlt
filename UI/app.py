import streamlit as st
import yaml
import os
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime

# --- START OF MODIFIED PATH CONFIGURATION ---
# Add the project root to the Python path
# This assumes app.py is in a 'UI' subfolder of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Correctly reference config.yaml from the new UI/ folder
CONFIG_PATH = os.path.join(project_root, 'config.yaml')
# --- END OF MODIFIED PATH CONFIGURATION ---

# Import torch after setting sys.path
import torch # Ensure torch is imported after sys.path is set, as it might be needed by other imports

# --- Configuration Loading (from config.yaml) ---
@st.cache_data # Cache the config loading for performance
def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
st.sidebar.title("RLPlayground Settings")

# --- Environment Selection ---
env_name = st.sidebar.selectbox(
    "Select Environment",
    ["LunarLander-v2", "CartPole-v1"], # Add more environments as you develop adapters
    index=0 # Default to LunarLander-v2
)
config['env_name'] = env_name # Update config dynamically

# --- Global Training Parameters ---
st.sidebar.header("Global Training Parameters")
train_episodes = st.sidebar.number_input(
    "Total Training Episodes",
    min_value=10, max_value=100000, value=config.get('train_episodes', 1450), step=100
)
config['train_episodes'] = train_episodes

max_timesteps_per_episode = st.sidebar.number_input(
    "Max Timesteps Per Episode",
    min_value=100, max_value=10000, value=config.get('max_timesteps_per_episode', 1000), step=100
)
config['max_timesteps_per_episode'] = max_timesteps_per_episode

device = st.sidebar.selectbox(
    "Device",
    ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
    index=0 if not torch.cuda.is_available() else (1 if config.get('device') == 'cuda' else 0)
)
config['device'] = device

# --- Agent Selection ---
st.header("RL Agent Training & Evaluation")
agent_type = st.selectbox(
    "Choose Agent Type",
    ["DQN", "A2C", "PPO"]
)

# --- Hyperparameter Input (Agent-Specific) ---
st.subheader(f"{agent_type} Hyperparameters")
agent_cfg_key = agent_type.lower()
current_agent_config = config.get(agent_cfg_key, {})

if agent_type == "DQN":
    # Presets dropdown
    dqn_preset_name = st.selectbox(
        "DQN Preset",
        ["base_dqn", "stability_first", "fast_learning"], # "base_dqn" implies using global dqn params
        index=0
    )

    if dqn_preset_name != "base_dqn":
        dqn_conf = config['presets'][dqn_preset_name]['dqn']
    else:
        dqn_conf = current_agent_config # Use the main dqn block

    # Display/Allow modification of DQN specific params
    st.write(f"Parameters for {dqn_preset_name}:")
    dqn_lr = st.number_input("Learning Rate (DQN)", value=float(dqn_conf.get('lr', 2.5e-4)), format="%e")
    dqn_gamma = st.slider("Gamma (DQN)", value=float(dqn_conf.get('gamma', 0.99)), min_value=0.0, max_value=1.0, step=0.01)
    dqn_batch_size = st.number_input("Batch Size (DQN)", value=int(dqn_conf.get('batch_size', 256)), min_value=16, step=16)
    # ... add more DQN params here, mirroring config.yaml structure
    dqn_buffer_size = st.number_input("Buffer Size (DQN)", value=int(dqn_conf.get('buffer_size', 100000)), min_value=1000, step=1000)
    dqn_min_replay_size = st.number_input("Min Replay Size (DQN)", value=int(dqn_conf.get('min_replay_size', 20000)), min_value=100, step=100)
    dqn_update_every = st.number_input("Update Every (DQN)", value=int(dqn_conf.get('update_every', 4)), min_value=1, step=1)
    dqn_tau = st.number_input("Tau (DQN)", value=float(dqn_conf.get('tau', 1e-3)), format="%e")
    dqn_double_dqn = st.checkbox("Double DQN", value=bool(dqn_conf.get('double_dqn', True)))
    dqn_clip_grad = st.number_input("Clip Grad (DQN)", value=float(dqn_conf.get('clip_grad', 0.5)), min_value=0.0, step=0.1)
    dqn_epsilon_start = st.number_input("Epsilon Start (DQN)", value=float(dqn_conf.get('epsilon_start', 1.0)), min_value=0.0, max_value=1.0, step=0.01)
    dqn_epsilon_end = st.number_input("Epsilon End (DQN)", value=float(dqn_conf.get('epsilon_end', 0.01)), min_value=0.0, max_value=1.0, step=0.001, format="%f")
    dqn_epsilon_decay = st.number_input("Epsilon Decay (DQN)", value=float(dqn_conf.get('epsilon_decay', 0.997)), min_value=0.0, max_value=1.0, step=0.001, format="%f")
    dqn_target_update_every = st.number_input("Target Update Every (DQN)", value=int(dqn_conf.get('target_update_every', 1000)), min_value=1, step=100)


    # Update the config object being passed
    if dqn_preset_name != "base_dqn":
        config['presets'][dqn_preset_name]['dqn']['lr'] = dqn_lr
        config['presets'][dqn_preset_name]['dqn']['gamma'] = dqn_gamma
        config['presets'][dqn_preset_name]['dqn']['batch_size'] = dqn_batch_size
        config['presets'][dqn_preset_name]['dqn']['buffer_size'] = dqn_buffer_size
        config['presets'][dqn_preset_name]['dqn']['min_replay_size'] = dqn_min_replay_size
        config['presets'][dqn_preset_name]['dqn']['update_every'] = dqn_update_every
        config['presets'][dqn_preset_name]['dqn']['tau'] = dqn_tau
        config['presets'][dqn_preset_name]['dqn']['double_dqn'] = dqn_double_dqn
        config['presets'][dqn_preset_name]['dqn']['clip_grad'] = dqn_clip_grad
        config['presets'][dqn_preset_name]['dqn']['epsilon_start'] = dqn_epsilon_start
        config['presets'][dqn_preset_name]['dqn']['epsilon_end'] = dqn_epsilon_end
        config['presets'][dqn_preset_name]['dqn']['epsilon_decay'] = dqn_epsilon_decay
        config['presets'][dqn_preset_name]['dqn']['target_update_every'] = dqn_target_update_every
    else:
        config['dqn']['lr'] = dqn_lr
        config['dqn']['gamma'] = dqn_gamma
        config['dqn']['batch_size'] = dqn_batch_size
        config['dqn']['buffer_size'] = dqn_buffer_size
        config['dqn']['min_replay_size'] = dqn_min_replay_size
        config['dqn']['update_every'] = dqn_update_every
        config['dqn']['tau'] = dqn_tau
        config['dqn']['double_dqn'] = dqn_double_dqn
        config['dqn']['clip_grad'] = dqn_clip_grad
        config['dqn']['epsilon_start'] = dqn_epsilon_start
        config['dqn']['epsilon_end'] = dqn_epsilon_end
        config['dqn']['epsilon_decay'] = dqn_epsilon_decay
        config['dqn']['target_update_every'] = dqn_target_update_every

elif agent_type == "A2C":
    a2c_lr = st.number_input("Learning Rate (A2C)", value=float(current_agent_config.get('lr', 7e-4)), format="%e")
    a2c_gamma = st.slider("Gamma (A2C)", value=float(current_agent_config.get('gamma', 0.99)), min_value=0.0, max_value=1.0, step=0.01)
    a2c_activation = st.selectbox("Activation (A2C)", ["Tanh", "ReLU"], index=0 if current_agent_config.get('activation', 'Tanh') == 'Tanh' else 1)
    a2c_value_coef = st.number_input("Value Coefficient (A2C)", value=float(current_agent_config.get('value_coef', 0.5)), min_value=0.0, step=0.01)
    a2c_entropy_coef = st.number_input("Entropy Coefficient (A2C)", value=float(current_agent_config.get('entropy_coef', 0.01)), min_value=0.0, step=0.001, format="%f")

    config['a2c']['lr'] = a2c_lr
    config['a2c']['gamma'] = a2c_gamma
    config['a2c']['activation'] = a2c_activation
    config['a2c']['value_coef'] = a2c_value_coef
    config['a2c']['entropy_coef'] = a2c_entropy_coef

elif agent_type == "PPO":
    ppo_lr = st.number_input("Learning Rate (PPO)", value=float(current_agent_config.get('lr', 3e-4)), format="%e")
    ppo_gamma = st.slider("Gamma (PPO)", value=float(current_agent_config.get('gamma', 0.99)), min_value=0.0, max_value=1.0, step=0.01)
    ppo_n_steps = st.number_input("N Steps (PPO)", value=int(current_agent_config.get('n_steps', 2048)), min_value=32, step=32)
    ppo_n_epochs = st.number_input("N Epochs (PPO)", value=int(current_agent_config.get('n_epochs', 10)), min_value=1, step=1)
    ppo_batch_size = st.number_input("Batch Size (PPO)", value=int(current_agent_config.get('batch_size', 64)), min_value=16, step=16)
    ppo_clip_epsilon = st.number_input("Clip Epsilon (PPO)", value=float(current_agent_config.get('clip_epsilon', 0.2)), min_value=0.0, max_value=0.5, step=0.01)
    ppo_gae_lambda = st.number_input("GAE Lambda (PPO)", value=float(current_agent_config.get('gae_lambda', 0.95)), min_value=0.0, max_value=1.0, step=0.01)
    ppo_ent_coef = st.number_input("Entropy Coefficient (PPO)", value=float(current_agent_config.get('ent_coef', 0.01)), min_value=0.0, step=0.001, format="%f")
    ppo_activation = st.selectbox("Activation (PPO)", ["ReLU", "Tanh"], index=0 if current_agent_config.get('activation', 'ReLU') == 'ReLU' else 1)

    config['ppo']['lr'] = ppo_lr
    config['ppo']['gamma'] = ppo_gamma
    config['ppo']['n_steps'] = ppo_n_steps
    config['ppo']['n_epochs'] = ppo_n_epochs
    config['ppo']['batch_size'] = ppo_batch_size
    config['ppo']['clip_epsilon'] = ppo_clip_epsilon
    config['ppo']['gae_lambda'] = ppo_gae_lambda
    config['ppo']['ent_coef'] = ppo_ent_coef
    config['ppo']['activation'] = ppo_activation


# --- Training Controls ---
st.subheader("Training Control")
col1, col2 = st.columns(2)

if col1.button("Start Training"):
    # Save the current config to config.yaml before starting the process
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    st.text(f"Starting {agent_type} training for {train_episodes} episodes on {env_name}...")
    
    # --- MODIFIED COMMAND FOR SUBPROCESS ---
    script_to_run = f"train_{agent_cfg_key}.py"
    # Ensure the script is called relative to the project root
    command = [sys.executable, os.path.join(project_root, script_to_run)]
    
    current_agent_run_name = agent_cfg_key
    if agent_type == "DQN":
        if dqn_preset_name != "base_dqn":
            command.extend(["--preset_name", dqn_preset_name])
            current_agent_run_name = f"dqn_{dqn_preset_name}"
        else:
            # If "base_dqn" is selected, it implicitly means the main dqn config block
            # For train_dqn.py, if no --preset_name is given, it runs ALL presets.
            # So, if you explicitly select "base_dqn" in the UI and want ONLY that,
            # you would need a mechanism in train_dqn.py to recognize this.
            # For now, it will run all presets if "base_dqn" is selected or no preset is selected.
            st.info("DQN training will run all defined presets. Showing results for the base DQN config.")
            current_agent_run_name = "dqn" # Default agent_name for base DQN
    
    # Use st.spinner for a loading indicator
    with st.spinner(f'Training {agent_type} for {train_episodes} episodes...'):
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1, # Line-buffered output
            cwd=project_root # Set current working directory to project root for subprocess
        )
        
        output_placeholder = st.empty()
        full_output = []
        for line in process.stdout:
            full_output.append(line)
            # Display recent output line by line
            output_placeholder.text("".join(full_output[-10:])) # Show last 10 lines
            # Check for evaluation outputs to trigger plot updates
            if "[EVAL]" in line:
                st.session_state.trigger_plot_update = True # Trigger plot update

        process.wait() # Wait for the process to finish
    
    if process.returncode == 0:
        st.success(f"{agent_type} Training Complete!")
        st.session_state.run_completed = True
        st.session_state.trained_agent_name = current_agent_run_name # Use the specific run name
    else:
        st.error(f"{agent_type} Training Failed! See output above for errors.")
        st.text("".join(full_output)) # Show all output if error

if col2.button("Stop Training"):
    # Implement graceful shutdown if training is a long-running process
    # This usually requires tracking the Popen object and sending a signal
    st.warning("Stopping training functionality not fully implemented yet. Please restart the app or manually stop the process in your terminal if it's running.")

# --- Live Plotting (Requires modifications to your plot_utils.py or direct plotting here) ---
st.subheader("Training Progress (Live Updates)")
chart_placeholder = st.empty()

# Function to load and plot data
def update_plot(agent_name_for_plot):
    csv_path = os.path.join(config['save_dir'], f'{agent_name_for_plot}_rollouts.csv')
    # Use absolute path for save_dir since app.py's cwd will be UI/
    abs_csv_path = os.path.join(project_root, csv_path)

    if os.path.exists(abs_csv_path):
        try:
            df = pd.read_csv(abs_csv_path)
            if 'ep_reward' in df.columns:
                reward_col = 'ep_reward'
            elif 'ep_rew' in df.columns:
                reward_col = 'ep_rew'
            else:
                reward_col = df.columns[2] # Fallback
            
            # Calculate moving average
            ma_window = 100
            df['moving_average'] = df[reward_col].rolling(window=ma_window, min_periods=1).mean()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['episode'], df[reward_col], label='Episode reward', alpha=0.4)
            ax.plot(df['episode'], df['moving_average'], label=f'{ma_window}-episode MA', color='tab:orange', linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title(f'{agent_name_for_plot.upper()} Training Rewards')
            ax.legend()
            ax.grid(True)
            chart_placeholder.pyplot(fig)
            plt.close(fig) # Close the figure to prevent display issues
        except Exception as e:
            chart_placeholder.warning(f"Could not load/plot data from {abs_csv_path}: {e}")
    else:
        chart_placeholder.info(f"No rollout data yet for {agent_name_for_plot} at {abs_csv_path}.")

# Initialize session state for plot updates
if 'trigger_plot_update' not in st.session_state:
    st.session_state.trigger_plot_update = False
if 'run_completed' not in st.session_state:
    st.session_state.run_completed = False
if 'trained_agent_name' not in st.session_state:
    st.session_state.trained_agent_name = agent_cfg_key # Default to generic

# Trigger plot update if training indicates an evaluation was done or on completion
if st.session_state.trigger_plot_update or st.session_state.run_completed:
    update_plot(st.session_state.trained_agent_name)
    st.session_state.trigger_plot_update = False # Reset trigger

# --- Play Agent Section ---
st.subheader("Play Trained Agent")
# Correctly form the default path for loading models
default_model_path = os.path.join(config['save_dir'], f"best_{agent_cfg_key}.pth")
abs_default_model_path = os.path.join(project_root, default_model_path) # Absolute path

play_model_path = st.text_input(
    "Path to Model (.pth)",
    value=abs_default_model_path # Use the absolute default path
)

# --- NEW: Add a checkbox for video recording for play_agent.py ---
record_play_video = st.checkbox("Record Playback Video for UI", value=True)


if st.button("Play Agent"):
    # Ensure the path is absolute for existence check and for saving into config for play_agent.py
    actual_play_model_path = play_model_path if os.path.isabs(play_model_path) else os.path.join(project_root, play_model_path)

    if not os.path.exists(actual_play_model_path):
        st.error(f"Model file not found at {actual_play_model_path}")
    else:
        st.info(f"Starting playback for {actual_play_model_path}...")
        
        try:
            temp_config = config.copy()
            temp_config['play_agent_type'] = agent_cfg_key
            temp_config['load_model_path'] = os.path.relpath(actual_play_model_path, project_root)
            
            # --- MODIFIED: Set render_mode to rgb_array and record_video based on checkbox ---
            temp_config['render_mode'] = "rgb_array" # Important for video recording
            temp_config['record_video'] = record_play_video # Use the checkbox value

            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(temp_config, f, default_flow_style=False)

            play_command = [sys.executable, os.path.join(project_root, "play_agent.py")]
            
            # Use st.spinner for a loading indicator during playback/recording
            with st.spinner(f'Playing {agent_type} and recording video...'):
                play_process = subprocess.Popen(
                    play_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=project_root
                )
                
                play_output_placeholder = st.empty()
                full_play_output = []
                recorded_video_path = None

                for line in play_process.stdout:
                    full_play_output.append(line)
                    play_output_placeholder.text("".join(full_play_output[-5:])) # Show last 5 lines of play output

                    # --- NEW: Capture the video path ---
                    if "VIDEO_PATH_FOR_STREAMLIT:" in line:
                        recorded_video_path = line.split("VIDEO_PATH_FOR_STREAMLIT:")[1].strip()
                        st.success(f"Video recorded to: {recorded_video_path}")
                        break # Once we have the path, we can stop reading output if desired
                
                play_process.wait() # Wait for the play process to finish

            if play_process.returncode == 0:
                st.success("Playback Complete!")
                # --- NEW: Display the recorded video ---
                if recorded_video_path and os.path.exists(recorded_video_path):
                    video_file = open(recorded_video_path, 'rb')
                    video_bytes = video_file.read()
                    st.video(video_bytes, format='video/mp4')
                    video_file.close()
                elif record_play_video:
                    st.warning("Video recording was enabled but no .mp4 file found. Check play_agent.py output.")
            else:
                st.error(f"Playback Failed! See output above for errors.")
                st.text("".join(full_play_output))

        except Exception as e:
            st.error(f"Error during playback: {e}")
            st.warning("Ensure your environment supports 'rgb_array' render mode for recording.")