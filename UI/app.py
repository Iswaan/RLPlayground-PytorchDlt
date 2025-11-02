import streamlit as st
import yaml
import os
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import numpy as np # Needed for np.mean in playback rewards
from datetime import datetime

# Add the project root to the Python path to import agents and utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch # Ensure torch is imported after sys.path is set

# --- Configuration Loading ---
CONFIG_PATH = os.path.join(project_root, 'config.yaml')

@st.cache_data
def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
st.sidebar.title("RLPlayground Settings")

# --- Environment Selection ---
st.sidebar.header("Environment")
env_name = st.sidebar.selectbox(
    "Select Environment",
    ["LunarLander-v3", "CartPole-v1"], # Add more environments as you develop adapters
    index=0 # Default to LunarLander-v2
)
config['env_name'] = env_name # Update config dynamically

# --- Environment Physics/Parameters (Dynamic based on selected env) ---
st.sidebar.subheader("Environment Physics/Parameters")
if 'environment_params' not in config:
    config['environment_params'] = {}
    
if env_name == "LunarLander-v2": # Or "LunarLander-v3"
    # Ensure default parameters exist in config for the selected env
    if env_name not in config['environment_params']:
        config['environment_params'][env_name] = {
            'gravity': -10.0,
            'enable_wind': False,
            'wind_power': 0.0
        }
    ll_params = config['environment_params'][env_name]

    current_gravity = st.sidebar.slider("Gravity",
                                         min_value=-15.0, max_value=-5.0,
                                         value=ll_params.get('gravity', -10.0), step=0.1)
    current_enable_wind = st.sidebar.checkbox("Enable Wind",
                                              value=ll_params.get('enable_wind', False))
    current_wind_power = st.sidebar.slider("Wind Power",
                                            min_value=0.0, max_value=20.0,
                                            value=ll_params.get('wind_power', 0.0), step=0.5,
                                            disabled=not current_enable_wind)

    # Update config object for current session
    config['environment_params'][env_name]['gravity'] = current_gravity
    config['environment_params'][env_name]['enable_wind'] = current_enable_wind
    config['environment_params'][env_name]['wind_power'] = current_wind_power
elif env_name == "CartPole-v1":
    st.sidebar.info("CartPole-v1 currently has no configurable physics parameters.")
    # Add CartPole specific parameters here if needed in the future

# --- Global Training Parameters ---
# These parameters are for the "Start Training" section, which you can choose to use or ignore
# if you only care about playback.
st.sidebar.header("Global Training Parameters")
train_episodes = st.sidebar.number_input(
    "Total Training Episodes",
    min_value=1, max_value=100000, value=config.get('train_episodes', 1450), step=100
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

# --- FIX 1: Define agent_cfg_key based on agent_type ---
agent_cfg_key = agent_type.lower()


# --- Hyperparameter Input (Agent-Specific) - Only for training, won't affect playback ---
st.subheader(f"{agent_type} Hyperparameters (for training)")
# This section is mostly for when you run the "Start Training" button.
# It will dynamically load / save to config.yaml if you change values.
# It will NOT affect the "Play Trained Agent" section which loads pre-trained models.

if agent_cfg_key not in config:
    config[agent_cfg_key] = {}
current_agent_config_dict = config[agent_cfg_key]

dqn_preset_name = None # Initialize dqn_preset_name outside the if block

if agent_type == "DQN":
    dqn_preset_name = st.selectbox(
        "DQN Preset (for training)", # Clarify purpose
        ["base_dqn", "stability_first", "fast_learning"],
        index=0
    )

    if dqn_preset_name != "base_dqn":
        if 'presets' not in config: config['presets'] = {}
        if dqn_preset_name not in config['presets']: config['presets'][dqn_preset_name] = {'dqn': {}}
        if 'dqn' not in config['presets'][dqn_preset_name]: config['presets'][dqn_preset_name]['dqn'] = {}
        dqn_conf = config['presets'][dqn_preset_name]['dqn']
    else:
        if 'dqn' not in config: config['dqn'] = {}
        dqn_conf = config['dqn']

    st.write(f"Parameters for {dqn_preset_name}:")
    dqn_lr = st.number_input("Learning Rate (DQN)", value=float(dqn_conf.get('lr', 2.5e-4)), format="%e")
    dqn_gamma = st.slider("Gamma (DQN)", value=float(dqn_conf.get('gamma', 0.99)), min_value=0.0, max_value=1.0, step=0.01)
    dqn_batch_size = st.number_input("Batch Size (DQN)", value=int(dqn_conf.get('batch_size', 256)), min_value=16, step=16)
    dqn_buffer_size = st.number_input("Buffer Size (DQN)", value=int(dqn_conf.get('buffer_size', 200000)), min_value=1000, step=1000)
    dqn_min_replay_size = st.number_input("Min Replay Size (DQN)", value=int(dqn_conf.get('min_replay_size', 20000)), min_value=100, step=100)
    dqn_update_every = st.number_input("Update Every (DQN)", value=int(dqn_conf.get('update_every', 4)), min_value=1, step=1)
    dqn_tau = st.number_input("Tau (DQN)", value=float(dqn_conf.get('tau', 1e-3)), format="%e")
    dqn_double_dqn = st.checkbox("Double DQN", value=bool(dqn_conf.get('double_dqn', True)))
    dqn_clip_grad = st.number_input("Clip Grad (DQN)", value=float(dqn_conf.get('clip_grad', 0.5)), min_value=0.0, step=0.1)
    dqn_epsilon_start = st.number_input("Epsilon Start (DQN)", value=float(dqn_conf.get('epsilon_start', 1.0)), min_value=0.0, max_value=1.0, step=0.01)
    dqn_epsilon_end = st.number_input("Epsilon End (DQN)", value=float(dqn_conf.get('epsilon_end', 0.01)), min_value=0.0, max_value=1.0, step=0.001, format="%f")
    dqn_epsilon_decay = st.number_input("Epsilon Decay (DQN)", value=float(dqn_conf.get('epsilon_decay', 0.997)), min_value=0.0, max_value=1.0, step=0.001, format="%f")
    dqn_target_update_every = st.number_input("Target Update Every (DQN)", value=int(dqn_conf.get('target_update_every', 1000)), min_value=1, step=100)


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
    if 'a2c' not in config: config['a2c'] = {}
    current_agent_config_dict = config['a2c']
    a2c_lr = st.number_input("Learning Rate (A2C)", value=float(current_agent_config_dict.get('lr', 7e-4)), format="%e")
    a2c_gamma = st.slider("Gamma (A2C)", value=float(current_agent_config_dict.get('gamma', 0.99)), min_value=0.0, max_value=1.0, step=0.01)
    a2c_activation = st.selectbox("Activation (A2C)", ["Tanh", "ReLU"], index=0 if current_agent_config_dict.get('activation', 'Tanh') == 'Tanh' else 1)
    a2c_value_coef = st.number_input("Value Coefficient (A2C)", value=float(current_agent_config_dict.get('value_coef', 0.5)), min_value=0.0, step=0.01)
    a2c_entropy_coef = st.number_input("Entropy Coefficient (A2C)", value=float(current_agent_config_dict.get('entropy_coef', 0.01)), min_value=0.0, step=0.001, format="%f")
    
    # --- FIX 2: Removed stray 'code' and 'Code' lines ---
    config['a2c']['lr'] = a2c_lr
    config['a2c']['gamma'] = a2c_gamma
    config['a2c']['activation'] = a2c_activation
    config['a2c']['value_coef'] = a2c_value_coef
    config['a2c']['entropy_coef'] = a2c_entropy_coef

elif agent_type == "PPO":
    if 'ppo' not in config: config['ppo'] = {}
    current_agent_config_dict = config['ppo']
    ppo_lr = st.number_input("Learning Rate (PPO)", value=float(current_agent_config_dict.get('lr', 3e-4)), format="%e")
    ppo_gamma = st.slider("Gamma (PPO)", value=float(current_agent_config_dict.get('gamma', 0.99)), min_value=0.0, max_value=1.0, step=0.01)
    ppo_n_steps = st.number_input("N Steps (PPO)", value=int(current_agent_config_dict.get('n_steps', 2048)), min_value=32, step=32)
    ppo_n_epochs = st.number_input("N Epochs (PPO)", value=int(current_agent_config_dict.get('n_epochs', 10)), min_value=1, step=1)
    ppo_batch_size = st.number_input("Batch Size (PPO)", value=int(current_agent_config_dict.get('batch_size', 64)), min_value=16, step=16)
    ppo_clip_epsilon = st.number_input("Clip Epsilon (PPO)", value=float(current_agent_config_dict.get('clip_epsilon', 0.2)), min_value=0.0, max_value=0.5, step=0.01)
    ppo_gae_lambda = st.number_input("GAE Lambda (PPO)", value=float(current_agent_config_dict.get('gae_lambda', 0.95)), min_value=0.0, max_value=1.0, step=0.01)
    ppo_ent_coef = st.number_input("Entropy Coefficient (PPO)", value=float(current_agent_config_dict.get('ent_coef', 0.01)), min_value=0.0, step=0.001, format="%f")
    ppo_activation = st.selectbox("Activation (PPO)", ["ReLU", "Tanh"], index=0 if current_agent_config_dict.get('activation', 'ReLU') == 'ReLU' else 1)
    
    # --- FIX 2: Removed stray 'code' and 'Code' lines ---
    config['ppo']['lr'] = ppo_lr
    config['ppo']['gamma'] = ppo_gamma
    config['ppo']['n_steps'] = ppo_n_steps
    config['ppo']['n_epochs'] = ppo_n_epochs
    config['ppo']['batch_size'] = ppo_batch_size
    config['ppo']['clip_epsilon'] = ppo_clip_epsilon
    config['ppo']['gae_lambda'] = ppo_gae_lambda
    config['ppo']['ent_coef'] = ppo_ent_coef
    config['ppo']['activation'] = ppo_activation


# --- Training Controls (Untouched for this specific goal) ---
st.subheader("Training Control")
col1, col2 = st.columns(2)
if col1.button("Start Training"):
    # Save the current config settings (including dynamic env params) to config.yaml
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    st.text(f"Starting {agent_type} training for {train_episodes} episodes on {env_name}...")

    # Uses agent_cfg_key defined above
    script_to_run = f"train_{agent_cfg_key}.py"
    command = [sys.executable, os.path.join(project_root, script_to_run)]

    # Uses agent_cfg_key defined above
    current_agent_run_name = agent_cfg_key
    if agent_type == "DQN":
        if dqn_preset_name != "base_dqn":
            command.extend(["--preset_name", dqn_preset_name])
            current_agent_run_name = f"dqn_{dqn_preset_name}"
        else:
            current_agent_run_name = "dqn"
            
        st.info(f"DQN training will run all defined presets unless a specific preset name ('dqn' or one from config.yaml) is passed. Reporting for '{current_agent_run_name}'.")
        
    with st.spinner(f'Training {agent_type} for {train_episodes} episodes...'):
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=project_root
        )
        
        output_placeholder = st.empty()
        full_output = []
        for line in process.stdout:
            full_output.append(line)
            output_placeholder.text("".join(full_output[-10:]))
            if "[EVAL]" in line:
                st.session_state.trigger_plot_update = True

        process.wait()

    if process.returncode == 0:
        st.success(f"{agent_type} Training Complete!")
        st.session_state.run_completed = True
        st.session_state.trained_agent_name = current_agent_run_name
    else:
        st.error(f"{agent_type} Training Failed! See output above for errors.")
        st.text("".join(full_output))

if col2.button("Stop Training"):
    st.warning("Stopping training functionality not fully implemented yet. Please restart the app or manually stop the process in your terminal if it's running.")


# --- Live Plotting (Untouched for this specific goal) ---
st.subheader("Training Progress (Live Updates)")
chart_placeholder = st.empty()

def update_plot(agent_name_for_plot):
    csv_path = os.path.join(config['save_dir'], f'{agent_name_for_plot}_rollouts.csv')
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
            plt.close(fig)
        except Exception as e:
            chart_placeholder.warning(f"Could not load/plot data from {abs_csv_path}: {e}")
    else:
        chart_placeholder.info(f"No rollout data yet for {agent_name_for_plot} at {abs_csv_path}.")

if 'trigger_plot_update' not in st.session_state:
    st.session_state.trigger_plot_update = False
if 'run_completed' not in st.session_state:
    st.session_state.run_completed = False
if 'trained_agent_name' not in st.session_state:
    # Uses agent_cfg_key defined above
    st.session_state.trained_agent_name = agent_cfg_key

if st.session_state.trigger_plot_update or st.session_state.run_completed:
    update_plot(st.session_state.trained_agent_name)
    st.session_state.trigger_plot_update = False


# --- Play Trained Agent with Visuals and Dynamic Environment ---
st.subheader("Play Trained Agent with Dynamic Environment")

# --- Model Selection Logic (Modified to find models in 'results' for now) ---
all_potential_models = []
# Check the 'results' folder directly for models (this will be on Streamlit Cloud after training there)
results_dir = os.path.join(project_root, 'results')
if os.path.exists(results_dir):
    for f in os.listdir(results_dir):
        # Uses agent_cfg_key defined above
        if f.endswith('.pth') and (f.startswith(f'best_{agent_cfg_key}') or (agent_type == "DQN" and f.startswith('best_dqn_'))):
            full_path = os.path.join(results_dir, f)
            all_potential_models.append(full_path)

# Filter and sort models for the specific agent type and preset
filtered_models = []
for model_path in all_potential_models:
    model_basename = os.path.basename(model_path)
    if agent_type == "DQN":
        # Uses dqn_preset_name defined above
        if dqn_preset_name == "base_dqn" and model_basename == "best_dqn.pth":
            filtered_models.append(model_path)
        elif dqn_preset_name != "base_dqn" and model_basename == f"best_dqn_{dqn_preset_name}.pth":
            filtered_models.append(model_path)
    else: # A2C or PPO
        # Uses agent_cfg_key defined above
        if model_basename == f"best_{agent_cfg_key}.pth":
            filtered_models.append(model_path)

filtered_models = sorted(list(set(filtered_models)), key=lambda x: os.path.basename(x).lower())

initial_model_selection = ""
if filtered_models:
    initial_model_selection = filtered_models[0]

selected_model_from_dropdown = st.selectbox(
    "Select Available Model (.pth)",
    options=filtered_models,
    index=0 if filtered_models else None,
    format_func=lambda x: os.path.basename(x) if x else "No models found",
    # Uses dqn_preset_name defined above
    help=f"Models for {agent_type} (and preset '{dqn_preset_name}' if DQN) found in '{os.path.basename(results_dir)}/'."
)

manual_model_path_input = st.text_input(
    "Or Manually Enter Model Path (.pth)",
    value=selected_model_from_dropdown if selected_model_from_dropdown else "",
    help="Enter the full path to a .pth model file. This overrides the dropdown selection."
)

final_play_model_path = manual_model_path_input if manual_model_path_input else (selected_model_from_dropdown if selected_model_from_dropdown else "")
# --- END Model Selection Logic ---


num_play_episodes = st.number_input("Number of Playback Episodes", min_value=1, max_value=10, value=1, step=1)
record_play_video = st.checkbox("Record Playback Video (displays in UI)", value=True)

if st.button("Run Playback"):
    actual_model_path_for_play = final_play_model_path
    
    if not os.path.exists(actual_model_path_for_play):
        st.error(f"Model file not found at: {actual_model_path_for_play}")
    else:
        st.info(f"Initiating playback for {os.path.basename(actual_model_path_for_play)} on {env_name}...")
        
        try:
            # Update config file for play_agent.py
            temp_config = config.copy()
            # Uses agent_cfg_key defined above
            temp_config['play_agent_type'] = agent_cfg_key
            temp_config['load_model_path'] = os.path.relpath(actual_model_path_for_play, project_root)
            temp_config['episodes_to_play'] = num_play_episodes

            if record_play_video:
                temp_config['render_mode'] = "rgb_array"
                temp_config['record_video'] = True
            else:
                temp_config['render_mode'] = "human" # 'human' might not work on Streamlit Cloud
                temp_config['record_video'] = False

            # Write the temporary config settings
            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(temp_config, f, default_flow_style=False)

            play_command = [sys.executable, os.path.join(project_root, "play_agent.py")]
            
            play_video_placeholder = st.empty()
            play_output_placeholder = st.empty()
            
            with st.spinner(f'Running playback for {agent_type} on {env_name} (recording video: {record_play_video})...'):
                play_process = subprocess.Popen(
                    play_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=project_root
                )
                
                full_play_output = []
                recorded_video_path = None

                for line in play_process.stdout:
                    full_play_output.append(line)
                    play_output_placeholder.text("".join(full_play_output[-5:]))

                    # Look for the specific output line from play_agent.py
                    if "VIDEO_PATH_FOR_STREAMLIT:" in line:
                        recorded_video_path = line.split("VIDEO_PATH_FOR_STREAMLIT:")[1].strip()
                        st.success(f"Video recorded to: {recorded_video_path}")

                play_process.wait()

            if play_process.returncode == 0:
                st.success("Playback Complete!")
                if record_play_video and recorded_video_path and os.path.exists(recorded_video_path):
                    st.subheader("Recorded Playback Video")
                    video_file = open(recorded_video_path, 'rb')
                    video_bytes = video_file.read()
                    st.video(video_bytes, format='video/mp4')
                    video_file.close()
                elif record_play_video:
                    st.warning("Video recording was enabled but no .mp4 file found. Check logs for errors.")
                
                # Parse rewards from output
                final_rewards = [float(l.split("Reward = ")[1].split(",")[0]) for l in full_play_output if "Episode" in l and "Reward =" in l]
                if final_rewards:
                    st.write(f"**Playback Rewards:** {', '.join([f'{r:.2f}' for r in final_rewards])}")
                    st.write(f"**Mean Playback Reward:** {np.mean(final_rewards):.2f}")
            else:
                st.error(f"Playback Failed! Check logs above for details.")
                st.text("".join(full_play_output))

        except Exception as e:
            st.error(f"An unexpected error occurred during playback: {e}")
            st.warning("Ensure environment supports 'rgb_array' for recording or 'human' for local display.")