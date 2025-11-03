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
st.set_page_config(layout="wide") 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch 

# --- Session State Initialization (CRITICAL FIX) ---
if 'training_in_progress' not in st.session_state:
    st.session_state['training_in_progress'] = False
if 'training_pid' not in st.session_state:
    st.session_state['training_pid'] = None
if 'log_file_path' not in st.session_state:
    st.session_state['log_file_path'] = None
if 'trained_agent_name_for_logs' not in st.session_state:
    st.session_state['trained_agent_name_for_logs'] = None

# --- Configuration Loading ---
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
tab_keys = ["▶️ Play Pre-trained Model", "🧠 Train New Model", "📊 View Training Logs"]
tab_playback, tab_training, tab_logs = st.tabs(tab_keys)

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
        pre_trained_models_dir = os.path.join(project_root, 'pre_trained_models')
        available_models = []
        if os.path.exists(pre_trained_models_dir):
            for f in os.listdir(pre_trained_models_dir):
                if f.endswith('.pth'):
                    available_models.append(os.path.join(pre_trained_models_dir, f))
        
        filtered_models = []
        if playback_agent_type == "DQN":
            if playback_dqn_preset != "base_dqn":
                preset_model_name = f"best_dqn_{playback_dqn_preset}.pth"
                filtered_models = [m for m in available_models if preset_model_name in os.path.basename(m)]
            else: # base_dqn
                filtered_models = [m for m in available_models if os.path.basename(m) == "best_dqn.pth"]
        else:
            filtered_models = [m for m in available_models if os.path.basename(m) == f"best_{playback_agent_cfg_key}.pth"]

        selected_model_path = st.selectbox(
            "Select Pre-trained Model",
            options=filtered_models,
            format_func=os.path.basename,
            help="These models were found in your 'pre_trained_models' folder."
        )

    num_play_episodes = st.number_input("Number of Playback Episodes", 1, 10, 1, 1, key="play_ep_count")

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
        # All your hyperparameter widgets remain here as before
        if training_agent_type == "DQN":
            if 'dqn' not in config: config['dqn'] = {}
            dqn_conf = config['dqn']
            
            st.write("Parameters for base DQN config:")
            dqn_lr = st.number_input("Learning Rate (DQN)", value=float(dqn_conf.get('lr', 2.5e-4)), format="%e", key="dqn_lr")
            dqn_gamma = st.slider("Gamma (DQN)", value=float(dqn_conf.get('gamma', 0.99)), min_value=0.0, max_value=1.0, step=0.01, key="dqn_gamma")
            dqn_batch_size = st.number_input("Batch Size (DQN)", value=int(dqn_conf.get('batch_size', 256)), min_value=16, step=16, key="dqn_bs")
            dqn_buffer_size = st.number_input("Buffer Size (DQN)", value=int(dqn_conf.get('buffer_size', 200000)), min_value=1000, step=1000, key="dqn_buf")
            dqn_min_replay_size = st.number_input("Min Replay Size (DQN)", value=int(dqn_conf.get('min_replay_size', 20000)), min_value=100, step=100, key="dqn_min_replay")
            dqn_update_every = st.number_input("Update Every (DQN)", value=int(dqn_conf.get('update_every', 4)), min_value=1, step=1, key="dqn_update")
            dqn_tau = st.number_input("Tau (DQN)", value=float(dqn_conf.get('tau', 1e-3)), format="%e", key="dqn_tau")
            dqn_double_dqn = st.checkbox("Double DQN", value=bool(dqn_conf.get('double_dqn', True)), key="dqn_ddqn")
            dqn_clip_grad = st.number_input("Clip Grad (DQN)", value=float(dqn_conf.get('clip_grad', 0.5)), min_value=0.0, step=0.1, key="dqn_clip")
            dqn_epsilon_start = st.number_input("Epsilon Start (DQN)", value=float(dqn_conf.get('epsilon_start', 1.0)), min_value=0.0, max_value=1.0, step=0.01, key="dqn_eps_start")
            dqn_epsilon_end = st.number_input("Epsilon End (DQN)", value=float(dqn_conf.get('epsilon_end', 0.01)), min_value=0.0, max_value=1.0, step=0.001, format="%f", key="dqn_eps_end")
            dqn_epsilon_decay = st.number_input("Epsilon Decay (DQN)", value=float(dqn_conf.get('epsilon_decay', 0.997)), min_value=0.0, max_value=1.0, step=0.001, format="%f", key="dqn_eps_decay")
            dqn_target_update_every = st.number_input("Target Update Every (DQN)", value=int(dqn_conf.get('target_update_every', 1000)), min_value=1, step=100, key="dqn_target_update")
            
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

        elif training_agent_type == "A2C":
            if 'a2c' not in config: config['a2c'] = {}
            current_agent_config_dict = config['a2c']
            a2c_lr = st.number_input("Learning Rate (A2C)", value=float(current_agent_config_dict.get('lr', 7e-4)), format="%e", key="a2c_lr")
            a2c_gamma = st.slider("Gamma (A2C)", value=float(current_agent_config_dict.get('gamma', 0.99)), min_value=0.0, max_value=1.0, step=0.01, key="a2c_gamma")
            a2c_activation = st.selectbox("Activation (A2C)", ["Tanh", "ReLU"], index=0 if current_agent_config_dict.get('activation', 'Tanh') == 'Tanh' else 1, key="a2c_act")
            a2c_value_coef = st.number_input("Value Coefficient (A2C)", value=float(current_agent_config_dict.get('value_coef', 0.5)), min_value=0.0, step=0.01, key="a2c_val_coef")
            a2c_entropy_coef = st.number_input("Entropy Coefficient (A2C)", value=float(current_agent_config_dict.get('entropy_coef', 0.01)), min_value=0.0, step=0.001, format="%f", key="a2c_ent_coef")

            config['a2c']['lr'] = a2c_lr
            config['a2c']['gamma'] = a2c_gamma
            config['a2c']['activation'] = a2c_activation
            config['a2c']['value_coef'] = a2c_value_coef
            config['a2c']['entropy_coef'] = a2c_entropy_coef

        elif training_agent_type == "PPO":
            if 'ppo' not in config: config['ppo'] = {}
            current_agent_config_dict = config['ppo']
            ppo_lr = st.number_input("Learning Rate (PPO)", value=float(current_agent_config_dict.get('lr', 3e-4)), format="%e", key="ppo_lr")
            ppo_gamma = st.slider("Gamma (PPO)", value=float(current_agent_config_dict.get('gamma', 0.99)), min_value=0.0, max_value=1.0, step=0.01, key="ppo_gamma")
            ppo_n_steps = st.number_input("N Steps (PPO)", value=int(current_agent_config_dict.get('n_steps', 2048)), min_value=32, step=32, key="ppo_n_steps")
            ppo_n_epochs = st.number_input("N Epochs (PPO)", value=int(current_agent_config_dict.get('n_epochs', 10)), min_value=1, step=1, key="ppo_n_epochs")
            ppo_batch_size = st.number_input("Batch Size (PPO)", value=int(current_agent_config_dict.get('batch_size', 64)), min_value=16, step=16, key="ppo_bs")
            ppo_clip_epsilon = st.number_input("Clip Epsilon (PPO)", value=float(current_agent_config_dict.get('clip_epsilon', 0.2)), min_value=0.0, max_value=0.5, step=0.01, key="ppo_clip")
            ppo_gae_lambda = st.number_input("GAE Lambda (PPO)", value=float(current_agent_config_dict.get('gae_lambda', 0.95)), min_value=0.0, max_value=1.0, step=0.01, key="ppo_gae")
            ppo_ent_coef = st.number_input("Entropy Coefficient (PPO)", value=float(current_agent_config_dict.get('ent_coef', 0.01)), min_value=0.0, step=0.001, format="%f", key="ppo_ent")
            ppo_activation = st.selectbox("Activation (PPO)", ["ReLU", "Tanh"], index=0 if current_agent_config_dict.get('activation', 'ReLU') == 'ReLU' else 1, key="ppo_act")

            config['ppo']['lr'] = ppo_lr
            config['ppo']['gamma'] = ppo_gamma
            config['ppo']['n_steps'] = ppo_n_steps
            config['ppo']['n_epochs'] = ppo_n_epochs
            config['ppo']['batch_size'] = ppo_batch_size
            config['ppo']['clip_epsilon'] = ppo_clip_epsilon
            config['ppo']['gae_lambda'] = ppo_gae_lambda
            config['ppo']['ent_coef'] = ppo_ent_coef
            config['ppo']['activation'] = ppo_activation
    
    col1, col2 = st.columns(2)
    if col1.button("Start Training", use_container_width=True, key="start_training_btn"):
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        st.info(f"Starting {training_agent_type} training... Go to the 'View Training Logs' tab to see progress.")
        
        script_to_run = f"train_{training_agent_cfg_key}.py"
        command = [sys.executable, os.path.join(project_root, script_to_run)]
        
        current_agent_run_name = training_agent_cfg_key
        if training_agent_type == "DQN":
            st.info("DQN training will run all defined presets. Reporting for the first preset.")
            current_agent_run_name = "dqn_stability_first"
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=project_root)
        
        st.session_state['training_pid'] = process.pid
        st.session_state['trained_agent_name_for_logs'] = current_agent_run_name
        st.session_state['training_in_progress'] = True
        
        st.rerun()

    if col2.button("Stop Training", use_container_width=True, key="stop_training_btn"):
        if st.session_state.get('training_in_progress', False) and st.session_state.get('training_pid'):
            try:
                os.kill(st.session_state['training_pid'], 9) # Send SIGKILL signal
                st.session_state['training_in_progress'] = False
                st.session_state['training_pid'] = None
                st.success("Training process has been sent a stop signal.")
                time.sleep(1)
                st.rerun()
            except ProcessLookupError:
                st.warning("Training process already finished or could not be found.")
                st.session_state['training_in_progress'] = False
                st.session_state['training_pid'] = None
            except Exception as e:
                st.error(f"Failed to stop process: {e}")
        else:
            st.warning("No active training process found to stop.")

# --- TAB 3: LOGS and PLOTS ---
with tab_logs:
    st.header("Live Training Logs and Plots")
    
    if not st.session_state.get('training_in_progress', False):
        st.info("Start a new training run from the 'Train New Model' tab to see live logs and plots here.")
    else:
        st.info("Training is in progress... Click 'Refresh' to see the latest output.")

        log_placeholder = st.empty()
        plot_placeholder = st.empty()

        if st.button("Refresh"):
            # Read the entire log file up to the current point
            # For this to work, we need to redirect subprocess output to a file
            # The previous threading approach was complex, this is a placeholder
            # for a more direct file-read approach
            st.warning("Live log reading from file not yet implemented in this version.")

            # Update plot function
            def update_log_plot(agent_name):
                csv_path = os.path.join(project_root, 'results', f'{agent_name}_rollouts.csv')
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if not df.empty:
                            if 'ep_reward' in df.columns:
                                reward_col = 'ep_reward'
                            elif 'ep_rew' in df.columns:
                                reward_col = 'ep_rew'
                            else:
                                reward_col = df.columns[2]
                            
                            ma_window = 100
                            df['moving_average'] = df[reward_col].rolling(window=ma_window, min_periods=1).mean()

                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(df['episode'], df[reward_col], label='Episode reward', alpha=0.4)
                            ax.plot(df['episode'], df['moving_average'], label=f'{ma_window}-episode MA', color='tab:orange', linewidth=2)
                            ax.set_xlabel('Episode')
                            ax.set_ylabel('Reward')
                            ax.set_title(f'{agent_name.upper()} Training Rewards')
                            ax.legend()
                            ax.grid(True)
                            plot_placeholder.pyplot(fig)
                            plt.close(fig)
                    except Exception as e:
                        plot_placeholder.warning(f"Error updating plot from {csv_path}: {e}")
            
            if st.session_state.get('trained_agent_name_for_logs'):
                update_log_plot(st.session_state['trained_agent_name_for_logs'])
        
        # Check if the process is still running in the background and offer a final refresh
        if st.session_state.get('training_pid'):
            try:
                os.kill(st.session_state['training_pid'], 0)
            except ProcessLookupError:
                st.success("Training process has completed!")
                st.info("Click 'Refresh' one last time to see the final logs and plot.")
                st.session_state['training_in_progress'] = False
                st.session_state['training_pid'] = None