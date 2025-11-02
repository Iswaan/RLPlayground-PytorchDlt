import argparse
import os
import time
import glob

import gymnasium as gym
import numpy as np
import torch
import yaml

# No need to import DQNAgent here, as build_agent_from_config is not used by play_agent.py in this mode.
# If this file were used standalone for evaluation, it would need the agent imports.

def record_video_wrapper(env, env_name, agent_name, video_folder='videos', episodes_to_record=1):
    """
    Wraps an environment with gym.wrappers.RecordVideo to save gameplay to a file.
    Returns the wrapped environment and the base directory where videos will be saved.
    """
    run_id = time.strftime("%Y%m%d-%H%M%S")
    # Video folder is typically inside the results directory
    full_video_folder = os.path.join('results', video_folder, f'{agent_name}-{env_name}-{run_id}')
    os.makedirs(full_video_folder, exist_ok=True)
    wrapped_env = gym.wrappers.RecordVideo(
        env,
        video_folder=full_video_folder,
        episode_trigger=lambda episode_id: episode_id < episodes_to_record,
        disable_logger=True # Suppress gym's default video logger output
    )
    print(f"Recording enabled; videos will be saved to: {full_video_folder}")
    return wrapped_env, full_video_folder

# Removed build_agent_from_config and main() from here
# as they are not needed for play_agent.py's functionality in this context
# and would introduce unnecessary config loading complexity within utils.
# The Streamlit app and play_agent.py handle agent creation and config loading.