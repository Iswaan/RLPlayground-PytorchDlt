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
                 eval_every=None, eval_episodes=5):
        self.env = env
        if device and 'cuda' in str(device) and not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to CPU.")
            device = 'cpu'
        self.device = torch.device(device)
        self.gamma = gamma
        self.lr = lr
        self.network = network.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        # --- MODIFIED: Handle save_path override logic ---
        self.save_path = save_path or (config.get('save_dir') if config else 'results')
        os.makedirs(self.save_path, exist_ok=True)
        # --- END MODIFIED ---

        self.best_reward = -float('inf')
        self.episode_rewards = []
        self.agent_name = self.__class__.__name__.replace('Agent','').lower()
        self.rollout_logger = RolloutLogger(self.save_path, f'{self.agent_name}_rollouts.csv') # Use self.save_path

        # Get these from the config's logging section
        logging_cfg = (config.get('logging', {}) if config else {})
        self.checkpoint_every = checkpoint_every if checkpoint_every is not None else logging_cfg.get('checkpoint_every')
        self.eval_every = eval_every if eval_every is not None else logging_cfg.get('eval_every')
        self.eval_episodes = eval_episodes if eval_episodes is not None else logging_cfg.get('eval_episodes', 5)


        self.config = config

        # Seeds
        self.seed = 42
        if config and 'seed' in config:
            self.seed = config['seed']
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(self.seed)

    def save_model(self, filename):
        torch.save(self.network.state_dict(), os.path.join(self.save_path, filename))

    def save(self, filename):
        return self.save_model(filename)

    def load_model(self, filename):
        path = filename # Assume full path is given by default for simplicity in play_agent.py
        if not os.path.isabs(path): # If not an absolute path, join with save_path
             path = os.path.join(self.save_path, filename)
        
        if not os.path.exists(path): # If still not found, try the default save_dir from config
             default_save_dir = self.config.get('save_dir', 'results') if self.config else 'results'
             path = os.path.join(default_save_dir, filename)

        if os.path.exists(path):
            self.network.load_state_dict(torch.load(path, map_location=self.device))
            self.network.eval()
        else:
            print(f"Model {filename} not found at path {path}")

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

    def _run_deterministic_eval(self, episodes=None):
        eval_episodes_count = episodes or self.eval_episodes
        env = self.env
        rewards = []
        lengths = []
        self.network.eval()
        for _ in range(eval_episodes_count):
            reset_out = env.reset()
            state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done = False
            total_reward = 0.0
            steps = 0
            while not done:
                action = self.act(state)

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
        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        std_reward = float(np.std(rewards)) if rewards else 0.0
        mean_len = float(np.mean(lengths)) if lengths else 0.0
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

            if self.checkpoint_every and (ep % self.checkpoint_every == 0):
                self.save_model(f'checkpoint_ep{ep}_{self.agent_name}.pth')

            if self.eval_every and (ep % self.eval_every == 0):
                eval_result = self._run_deterministic_eval()
                print(f"[EVAL] After episode {ep}: mean_reward={eval_result['mean_reward']:.2f}, std={eval_result['std_reward']:.2f}")
                self._append_eval_csv(eval_result, ep)

            if (ep % timestamp_interval == 0):
                now = datetime.utcnow()
                elapsed = (now - start_time).total_seconds()
                line = f"{now.isoformat()} | episode {ep} | elapsed_seconds {elapsed:.2f}\n"
                print(f"[TIME] {line.strip()}")
                try:
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
        
        arr = np.array(self.episode_rewards, dtype=np.float32) if len(self.episode_rewards) else np.array([])
        mean_reward = float(np.mean(arr)) if arr.size else 0.0
        std_reward = float(np.std(arr)) if arr.size else 0.0
        now = datetime.utcnow()
        elapsed = (now - start_time).total_seconds()
        summary_line = f"FINAL SUMMARY | {now.isoformat()} | episodes {len(arr)} | mean_reward {mean_reward:.4f} | std_reward {std_reward:.4f} | elapsed_seconds {elapsed:.2f}\n"
        print(summary_line.strip())
        try:
            self.rollout_logger.write_timestamp(episode=len(arr), ep_len=None, ep_reward=mean_reward,
                                               ep_mean=mean_reward, ep_std=std_reward, fps=None,
                                               time_elapsed=elapsed, total_timesteps=None,
                                               timestamp=int(now.timestamp()))
            with open(timestamps_log_path, 'a') as tf:
                tf.write(summary_line)
        except Exception:
            pass

        print("Training complete.")