import time
import csv
import os
from collections import deque

import numpy as np

class RolloutLogger:
    """
    RolloutLogger prints a boxed (ASCII) per-episode summary and appends rows to CSV.

    Constructor:
      RolloutLogger(save_path='results', filename='rollouts.csv', rolling_window=100, cap=250.0)

    CSV columns:
      episode, ep_len, ep_reward, ep_mean, ep_std, fps, time_elapsed, total_timesteps, timestamp

    Usage:
      from utils.rollout_logger import RolloutLogger
      logger = RolloutLogger(save_path='results', filename='dqn_rollouts.csv', rolling_window=100, cap=250.0)
      ...
      logger.log(ep_reward_raw=ep_reward, ep_len=ep_len, timesteps_this_episode=ep_len)
    """
    def __init__(self, save_path='results', filename='rollouts.csv', rolling_window=100, cap=250.0):
        os.makedirs(save_path, exist_ok=True)
        self.file_path = os.path.join(save_path, filename)
        self.rolling_window = int(rolling_window)
        self.cap = float(cap)

        # CSV header
        header = ["episode", "ep_len", "ep_reward", "ep_mean", "ep_std", "fps", "time_elapsed", "total_timesteps", "timestamp"]
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
        else:
            # if file exists but empty, write header
            if os.path.getsize(self.file_path) == 0:
                with open(self.file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

        self.start_time = time.time()
        self.last_time = self.start_time
        self.total_timesteps = 0
        self.episode_counter = 0
        self.scores = deque(maxlen=self.rolling_window)

    def _boxed_print(self, lines, min_width=60):
        """
        Print list of text lines inside an ASCII box. Ensures readable square-ish box.
        """
        content = "  ".join(line for line in lines)
        width = max(min_width, len(content) + 4)
        print("+" + "-" * (width - 2) + "+")
        print("|" + content.center(width - 2) + "|")
        print("+" + "-" * (width - 2) + "+")

    def log(self, ep_reward_raw, ep_len, timesteps_this_episode):
        """
        Log one episode.

        Args:
          ep_reward_raw (float): raw episode reward (used for learning elsewhere)
          ep_len (int): episode length (timesteps)
          timesteps_this_episode (int): number of env steps performed in this episode (same as ep_len normally)
        """
        self.episode_counter += 1
        self.total_timesteps += int(timesteps_this_episode)

        now = time.time()
        elapsed = now - self.last_time if now - self.last_time > 1e-9 else 1e-9
        fps = timesteps_this_episode / elapsed if elapsed > 0 else 0.0
        self.last_time = now

        # update rolling window (store raw rewards)
        self.scores.append(float(ep_reward_raw))

        # rolling statistics
        ep_mean_raw = float(np.mean(self.scores)) if len(self.scores) > 0 else 0.0
        ep_std_raw = float(np.std(self.scores)) if len(self.scores) > 0 else 0.0

        # cap values for display and CSV if desired (learning should continue to use raw values)
        ep_reward = float(np.clip(ep_reward_raw, -self.cap, self.cap))
        ep_mean = float(np.clip(ep_mean_raw, -self.cap, self.cap))
        ep_std = float(ep_std_raw)

        timestamp = int(time.time())

        # --- THIS IS THE FIX ---
        # Added 'ep:{self.episode_counter}' which the Streamlit app looks for.
        lines = [
            f"rollout/ ep: {self.episode_counter:<4d} ep_len:{int(ep_len):<4d} ep_rew:{ep_reward:8.2f}",
            f"mean±std:{ep_mean:8.2f}±{ep_std:6.2f} fps:{fps:6.1f}",
            f"time_elapsed:{elapsed:6.3f}s total_timesteps:{self.total_timesteps}"
        ]
        self._boxed_print(lines, min_width=80)

        # Append to CSV (store capped reward and capped mean; std stored raw)
        with open(self.file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_counter,
                int(ep_len),
                f"{ep_reward:.6f}",
                f"{ep_mean:.6f}",
                f"{ep_std:.6f}",
                f"{fps:.6f}",
                f"{elapsed:.6f}",
                int(self.total_timesteps),
                timestamp
            ])

    def write_timestamp(self, episode, ep_len, ep_reward, ep_mean, ep_std, fps, time_elapsed, total_timesteps, timestamp=None):
        """Append a custom row to the rollouts CSV. This is used for periodic timestamp entries or final summaries.

        The row follows the CSV header used in this logger:
        [episode, ep_len, ep_reward, ep_mean, ep_std, fps, time_elapsed, total_timesteps, timestamp]
        """
        if timestamp is None:
            timestamp = int(time.time())
        # Ensure the file exists and header present (constructor already ensures this)
        with open(self.file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                int(episode),
                int(ep_len) if ep_len is not None else 0,
                f"{ep_reward:.6f}" if ep_reward is not None else "",
                f"{ep_mean:.6f}" if ep_mean is not None else "",
                f"{ep_std:.6f}" if ep_std is not None else "",
                f"{fps:.6f}" if fps is not None else "",
                f"{time_elapsed:.6f}" if time_elapsed is not None else "",
                int(total_timesteps) if total_timesteps is not None else 0,
                int(timestamp)
            ])