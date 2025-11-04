import torch
import os
import random
import numpy as np
import csv
import json
from datetime import datetime
from utils.rollout_logger import RolloutLogger

class BaseAgent:
    def __init__(self, env, network, lr=3e-4, gamma=0.99, device='cpu',
                 save_path=None, config=None, checkpoint_every=None,
                 eval_every=None, eval_episodes=5): # <--- MODIFIED: save_path=None
        self.env = env
        if device and 'cuda' in str(device) and not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to CPU.")
            device = 'cpu'
        self.device = torch.device(device)
        self.gamma = gamma
        self.lr = lr
        self.network = network.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.save_path = save_path or (config.get('save_dir') if config else 'results')
        os.makedirs(self.save_path, exist_ok=True)
        self.best_reward = -float('inf')
        self.episode_rewards = []
        self.agent_name = self.__class__.__name__.replace('Agent','').lower()
        self.rollout_logger = RolloutLogger(save_path, f'{self.agent_name}_rollouts.csv')

        self.checkpoint_every = checkpoint_every
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes

        self.config = config

        # Seeds
        self.seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(self.seed)

    def save_model(self, filename):
        torch.save(self.network.state_dict(), os.path.join(self.save_path, filename))

    # Backwards-compatible alias used by some scripts
    def save(self, filename):
        return self.save_model(filename)

    def load_model(self, filename):
        path = os.path.join(self.save_path, filename)
        if os.path.exists(path):
            self.network.load_state_dict(torch.load(path, map_location=self.device))
            self.network.eval()
        else:
            print(f"Model {filename} not found")

    # Backwards-compatible alias used by some older scripts
    def load(self, filename):
        return self.load_model(filename)

    def discount_rewards(self, rewards):
        discounted = []
        r = 0
        for reward in reversed(rewards):
            r = reward + self.gamma * r
            discounted.insert(0, r)
        return discounted

    def train_episode(self):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError

    # In base_agent.py

    def _run_deterministic_eval(self, episodes=None):
        episodes = episodes or self.eval_episodes
        env = self.env
        rewards = []
        lengths = []
        self.network.eval()
        for _ in range(episodes):
            reset_out = env.reset()
            state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done = False
            total_reward = 0.0
            steps = 0
            while not done:
                # --- START OF MODIFIED SECTION ---
                # Use the agent's act method for deterministic action
                action = self.act(state) 
                # --- END OF MODIFIED SECTION ---

                step_out = env.step(action)
                if len(step_out) == 5:
                    state, reward, terminated, truncated, _ = step_out
                    done = bool(terminated or truncated)
                elif len(step_out) == 4:
                    state, reward, done, _ = step_out
                else:
                    state, reward, done = step_out[:3]
                total_reward += reward
                steps += 1
            rewards.append(total_reward)
            lengths.append(steps)
        self.network.train()
        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        mean_len = float(np.mean(lengths))
        return {'mean_reward': mean_reward, 'std_reward': std_reward, 'mean_length': mean_len}

    def _append_eval_csv(self, eval_result, episode_number):
        os.makedirs(self.save_path, exist_ok=True)
        path = os.path.join(self.save_path, f'{self.agent_name}_eval.csv')
        fieldnames = ['timestamp', 'train_episode', 'mean_reward', 'std_reward', 'mean_length']
        exists = os.path.exists(path)
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({
                'timestamp': datetime.utcnow().isoformat(),
                'train_episode': episode_number,
                'mean_reward': eval_result['mean_reward'],
                'std_reward': eval_result['std_reward'],
                'mean_length': eval_result['mean_length']
            })

    def _save_training_summary(self):
        if len(self.episode_rewards) == 0:
            return
        arr = np.array(self.episode_rewards, dtype=np.float32)
        stats = {
            'agent': self.agent_name,
            'timestamp': datetime.utcnow().isoformat(),
            'num_episodes': int(arr.shape[0]),
            'mean_reward': float(np.mean(arr)),
            'median_reward': float(np.median(arr)),
            'std_reward': float(np.std(arr, ddof=0)),
            'var_reward': float(np.var(arr, ddof=0)),
            'min_reward': float(np.min(arr)),
            'max_reward': float(np.max(arr))
        }

        json_path = os.path.join(self.save_path, f'{self.agent_name}_training_summary.json')
        with open(json_path, 'w') as jf:
            json.dump(stats, jf, indent=2)

        csv_path = os.path.join(self.save_path, f'{self.agent_name}_training_summary.csv')
        header = ['timestamp', 'agent', 'num_episodes', 'mean_reward', 'median_reward', 'std_reward', 'var_reward', 'min_reward', 'max_reward']
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as cf:
            writer = csv.writer(cf)
            if not file_exists:
                writer.writerow(header)
            writer.writerow([stats['timestamp'], stats['agent'], stats['num_episodes'], stats['mean_reward'],
                             stats['median_reward'], stats['std_reward'], stats['var_reward'],
                             stats['min_reward'], stats['max_reward']])

    def train(self, episodes=500):
        start_time = datetime.utcnow()
        timestamp_interval = 200
        timestamps_log_path = os.path.join(self.save_path, f'{self.agent_name}_timestamps.log')

        for ep in range(1, episodes + 1):
            result = self.train_episode()
            if isinstance(result, tuple):
                reward, ep_len, timesteps = result
            else:
                reward = result
                ep_len = 0
                timesteps = 0

            if reward > self.best_reward:
                self.best_reward = reward
                self.save_model(f'best_{self.agent_name}.pth')

            self.episode_rewards.append(reward)
            self.rollout_logger.log(reward, ep_len, timesteps)

            # For a2c, ppo and dqn, print a concise per-episode summary including episode number
            if self.agent_name in ('a2c', 'ppo', 'dqn'):
                # episode number is the length of episode_rewards
                ep_no = len(self.episode_rewards)
                print(f"[TRAIN] agent={self.agent_name} ep={ep_no} reward={reward:.3f} len={ep_len}")

            if self.checkpoint_every and (ep % self.checkpoint_every == 0):
                self.save_model(f'checkpoint_ep{ep}_{self.agent_name}.pth')

            if self.eval_every and (ep % self.eval_every == 0):
                eval_result = self._run_deterministic_eval(self.eval_episodes)
                print(f"[EVAL] After episode {ep}: mean_reward={eval_result['mean_reward']:.2f}, std={eval_result['std_reward']:.2f}")
                self._append_eval_csv(eval_result, ep)

            # Periodic timestamp logging for long runs (only for a2c and ppo agents)
            if self.agent_name in ('a2c', 'ppo') and (ep % timestamp_interval == 0):
                now = datetime.utcnow()
                elapsed = (now - start_time).total_seconds()
                line = f"{now.isoformat()} | episode {ep} | elapsed_seconds {elapsed:.2f}\n"
                print(f"[TIME] {line.strip()}")
                try:
                    # also append a CSV-friendly timestamp row to the rollouts CSV
                    # we leave most numeric columns empty except episode, time_elapsed and timestamp
                    self.rollout_logger.write_timestamp(episode=ep, ep_len=ep_len, ep_reward=reward,
                                                       ep_mean=(np.mean(self.episode_rewards) if self.episode_rewards else 0.0),
                                                       ep_std=(np.std(self.episode_rewards) if self.episode_rewards else 0.0),
                                                       fps=None, time_elapsed=elapsed, total_timesteps=timesteps,
                                                       timestamp=int(now.timestamp()))
                    with open(timestamps_log_path, 'a') as tf:
                        tf.write(line)
                except Exception:
                    pass

        self._save_training_summary()
        # Print final summary (mean/std) for a2c and ppo
        if self.agent_name in ('a2c', 'ppo'):
            arr = np.array(self.episode_rewards, dtype=np.float32) if len(self.episode_rewards) else np.array([])
            mean_reward = float(np.mean(arr)) if arr.size else 0.0
            std_reward = float(np.std(arr)) if arr.size else 0.0
            now = datetime.utcnow()
            elapsed = (now - start_time).total_seconds()
            summary_line = f"FINAL SUMMARY | {now.isoformat()} | episodes {len(arr)} | mean_reward {mean_reward:.4f} | std_reward {std_reward:.4f} | elapsed_seconds {elapsed:.2f}\n"
            print(summary_line.strip())
            try:
                # also append a final summary row to rollouts CSV
                self.rollout_logger.write_timestamp(episode=len(arr), ep_len=None, ep_reward=mean_reward,
                                                   ep_mean=mean_reward, ep_std=std_reward, fps=None,
                                                   time_elapsed=elapsed, total_timesteps=None,
                                                   timestamp=int(now.timestamp()))
                with open(timestamps_log_path, 'a') as tf:
                    tf.write(summary_line)
            except Exception:
                pass

        print("Training complete.")
