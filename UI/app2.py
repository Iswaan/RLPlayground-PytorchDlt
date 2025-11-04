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

# --- Session State Initialization ---
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
        if training_agent_type == "DQN":
            # Preset selection for training
            training_dqn_preset = st.selectbox(
                "DQN Preset to Train",
                ["fast_learning", "stability_first"], # Removed 'base_dqn' for training clarity
                key="train_dqn_preset"
            )
            
            # Load the correct config block based on preset selection
            if training_dqn_preset in config.get('presets', {}):
                dqn_conf = config['presets'][training_dqn_preset]['dqn']
            else:
                dqn_conf = config.get('dqn', {})

            st.write(f"Parameters for {training_dqn_preset}:")
            dqn_lr = st.number_input("Learning Rate (DQN)", value=float(dqn_conf.get('lr', 2.5e-4)), format="%e", key="dqn_lr")
            # ... and so on for all DQN parameters ...

        elif training_agent_type == "A2C":
            # ... A2C parameters ...
            pass
        elif training_agent_type == "PPO":
            # ... PPO parameters ...
            pass
        
    col1, col2 = st.columns(2)
    if col1.button("Start Training", use_container_width=True, key="start_training_btn"):
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        st.info(f"Starting {training_agent_type} training... Go to the 'View Training Logs' tab to see progress.")
        
        script_to_run = f"train_{training_agent_cfg_key}.py"
        command = [sys.executable, os.path.join(project_root, script_to_run)]
        
        log_dir = config.get('log_save_dir', 'logs_results')
        command.extend(['--save_path', log_dir])

        current_agent_run_name = training_agent_cfg_key
        if training_agent_type == "DQN":
            training_dqn_preset = st.session_state.get('train_dqn_preset', 'fast_learning')
            command.extend(["--preset_name", training_dqn_preset])
            current_agent_run_name = f"dqn_{training_dqn_preset}"
        
        log_file_path = os.path.join(project_root, "training_log.log")
        st.session_state['log_file_path'] = log_file_path
        
        with open(log_file_path, 'w') as log_file:
            process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=project_root)
        
        st.session_state['training_pid'] = process.pid
        st.session_state['trained_agent_name_for_logs'] = current_agent_run_name
        st.session_state['training_in_progress'] = True
        
        st.rerun()

    if col2.button("Stop Training", use_container_width=True, key="stop_training_btn"):
        if st.session_state.get('training_in_progress', False) and st.session_state.get('training_pid'):
            try:
                os.kill(st.session_state['training_pid'], 9)
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
            st.warning("No active training process has been started in this session.")

# --- TAB 3: LOGS and PLOTS ---
with tab_logs:
    st.header("Live Training Logs and Plots")
    
    if not st.session_state.get('training_in_progress', False):
        st.info("Start a new training run from the 'Train New Model' tab to see live logs and plots here.")
    else:
        st.info("Training is in progress... Click 'Refresh' to see the latest output.")

        log_placeholder = st.empty()
        plot_placeholder = st.empty()
        
        # Add a placeholder for the episode counter
        episode_counter_placeholder = st.empty()

        if st.button("Refresh"):
            log_file_path = st.session_state.get('log_file_path')
            if log_file_path and os.path.exists(log_file_path):
                with open(log_file_path, 'r') as f:
                    full_output = f.readlines()
                
                # Episode Counter Logic
                latest_episode = 0
                for line in reversed(full_output):
                    if "rollout/ ep:" in line:
                        try:
                            episode_str = line.split("ep:")[1].strip().split(" ")[0]
                            latest_episode = int(episode_str)
                            break
                        except (IndexError, ValueError):
                            continue
                
                if latest_episode > 0:
                    total_episodes = config.get('train_episodes', 'N/A')
                    episode_counter_placeholder.progress(latest_episode / total_episodes, text=f"Episode: {latest_episode} / {total_episodes}")
                
                log_placeholder.text("".join(full_output[-30:]))
            
            # Update plot function
            def update_log_plot(agent_name):
                log_dir = config.get('log_save_dir', 'logs_results')
                csv_path = os.path.join(project_root, log_dir, f'{agent_name}_rollouts.csv')
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if not df.empty and len(df) > 1:
                            if 'episode' in df.columns:
                                episode_col = 'episode'
                            else:
                                episode_col = df.columns[0]

                            if 'ep_reward' in df.columns:
                                reward_col = 'ep_reward'
                            elif 'ep_rew' in df.columns:
                                reward_col = 'ep_rew'
                            else:
                                reward_col = df.columns[2]
                            
                            ma_window = 100
                            df['moving_average'] = df[reward_col].rolling(window=ma_window, min_periods=1).mean()

                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(df[episode_col], df[reward_col], label='Episode reward', alpha=0.4)
                            ax.plot(df[episode_col], df['moving_average'], label=f'{ma_window}-episode MA', color='tab:orange', linewidth=2)
                            
                            from matplotlib.ticker import MaxNLocator
                            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

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
        
        if st.session_state.get('training_pid'):
            try:
                os.kill(st.session_state['training_pid'], 0)
            except ProcessLookupError:
                st.success("Training process has completed!")
                st.info("Click 'Refresh' one last time to see the final logs and plot.")
                st.session_state['training_in_progress'] = False
                st.session_state['training_pid'] = None