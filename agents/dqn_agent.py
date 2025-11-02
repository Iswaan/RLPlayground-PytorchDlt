import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_agent import BaseAgent
from utils.replay_buffer import ReplayBuffer
import os # <--- ADD THIS LINE: You'll need os for os.path.join

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=None):
        """Simple MLP Q-network.

        hidden_layers: iterable of ints or None. If None, defaults to [64, 64].
        """
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 64]
        layers = []
        last_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# module-level alias for backward compatibility
DQNNetwork = QNetwork

class DQNAgent(BaseAgent):
    def __init__(self, env=None, config=None, preset_name=None, *, state_size=None, action_size=None, device='cpu', hidden_layers=None, seed=42, lr=None, batch_size=None, buffer_size=None, network=None, **kwargs):
        # Two initialization styles supported:
        # 1) DQNAgent(env=env, config=config, preset_name=...)
        # 2) DQNAgent(state_size=..., action_size=..., device=..., hidden_layers=...)
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # record provided network early so both config and config-less paths see it
        self._provided_network = network

        if config is not None:
            self.config = config
            dqn_conf = config.get('dqn', {})
            if preset_name:
                preset = config.get('presets', {}).get(preset_name, {})
                dqn_conf = {**dqn_conf, **preset.get('dqn', {})}
            # read config values with fallbacks to kwargs
            self.hidden_layers = dqn_conf.get('hidden_layers', hidden_layers or [64, 64])
            # ensure lr is a float (YAML may parse scientific notation as string in some edge cases)
            self.lr = float(dqn_conf.get('lr', lr or 2.5e-4))
            self.gamma = dqn_conf.get('gamma', 0.99)
            self.batch_size = int(dqn_conf.get('batch_size', batch_size or 64))
            self.buffer_size = int(dqn_conf.get('buffer_size', buffer_size or int(1e5)))
            self.min_replay_size = int(dqn_conf.get('min_replay_size', 1000))
            self.update_every = int(dqn_conf.get('update_every', 4))
            self.tau = float(dqn_conf.get('tau', 1e-3))
            self.double_dqn = bool(dqn_conf.get('double_dqn', True))
            self.clip_grad = dqn_conf.get('clip_grad', 0.5)
            self.epsilon = float(dqn_conf.get('epsilon_start', 1.0))
            self.epsilon_end = float(dqn_conf.get('epsilon_end', 0.01))
            self.epsilon_decay = float(dqn_conf.get('epsilon_decay', 0.997))
            self.target_update_every = dqn_conf.get('target_update_every', None)
            cfg_seed = config.get('seed', seed)
            self.replay_buffer = ReplayBuffer(max_size=self.buffer_size, seed=cfg_seed)
            # infer dims from env if provided
            if env is not None:
                state_size = int(np.prod(env.observation_space.shape))
                action_size = int(env.action_space.n)
                self.env = env
            # record provided network (defer moving to device until self.device is known)
            if network is not None:
                self._provided_network = network
        else:
            # config-less init
            self.hidden_layers = hidden_layers or [64, 64]
            self.lr = float(lr or 2.5e-4)
            self.gamma = 0.99
            self.batch_size = int(batch_size or 64)
            self.buffer_size = int(buffer_size or int(1e5))
            self.min_replay_size = int(kwargs.get('min_replay_size', 1000))
            self.update_every = int(kwargs.get('update_every', 4))
            self.tau = float(kwargs.get('tau', 1e-3))
            self.double_dqn = bool(kwargs.get('double_dqn', True))
            self.clip_grad = kwargs.get('clip_grad', 0.5)
            self.epsilon = float(kwargs.get('epsilon_start', 1.0))
            self.epsilon_end = float(kwargs.get('epsilon_end', 0.01))
            self.epsilon_decay = float(kwargs.get('epsilon_decay', 0.997))
            self.target_update_every = kwargs.get('target_update_every', None)
            self.replay_buffer = ReplayBuffer(max_size=self.buffer_size, seed=seed)
            # infer dims from env if provided in config-less init
            if env is not None:
                state_size = int(np.prod(env.observation_space.shape))
                action_size = int(env.action_space.n)
                self.env = env
            # record provided network (defer moving to device until self.device is known)
            if network is not None:
                self._provided_network = network

        # if q_net wasn't set by passing a `network` instance above, create one
        self.device = torch.device(device if (isinstance(device, str) and 'cuda' in device and torch.cuda.is_available()) else 'cpu')
        # If a network was provided (instance or class), use it now (we have self.device)
        if hasattr(self, '_provided_network') and self._provided_network is not None:
            provided = self._provided_network
            if isinstance(provided, nn.Module):
                self.q_net = provided.to(self.device)
            else:
                # assume it's a class that can be instantiated with (state_size, action_size [, hidden_layers])
                if state_size is None or action_size is None:
                    raise ValueError("DQNAgent requires state_size and action_size when a network class is provided")
                try:
                    # try the simple constructor
                    self.q_net = provided(state_size, action_size).to(self.device)
                except TypeError:
                    # fallback: pass hidden_layers if the class expects it
                    self.q_net = provided(state_size, action_size, self.hidden_layers).to(self.device)
        else:
            if not hasattr(self, 'q_net'):
                if state_size is None or action_size is None:
                    raise ValueError("DQNAgent requires state_size and action_size when env/config do not provide them")
                self.q_net = QNetwork(state_size, action_size, self.hidden_layers).to(self.device)
        # create target_net if not provided
        if not hasattr(self, 'target_net'):
            self.target_net = QNetwork(self.q_net.model[0].in_features, self.q_net.model[-1].out_features, self.hidden_layers).to(self.device)
            self.target_net.load_state_dict(self.q_net.state_dict())

        # initialize BaseAgent with q_net as the network
        save_dir = (config.get('save_dir') if config else 'results')
        logging_cfg = (config.get('logging', {}) if config else {})
        super().__init__(getattr(self, 'env', None), self.q_net, lr=self.lr, gamma=self.gamma,
                         save_path=save_dir,
                         config=self.config,
                         checkpoint_every=logging_cfg.get('checkpoint_every'),
                         eval_every=logging_cfg.get('eval_every'),
                         eval_episodes=logging_cfg.get('eval_episodes', 5))

        # --- START OF THE PART YOU NEED TO ADD/MODIFY ---
        if preset_name:
            self.agent_name = f"dqn_{preset_name}" # Override the agent_name with preset
            # Update the rollout logger's filename so it uses the new agent_name
            self.rollout_logger.file_path = os.path.join(self.save_path, f'{self.agent_name}_rollouts.csv')
            # The BaseAgent's _append_eval_csv and _save_training_summary methods
            # *will* dynamically use the updated self.agent_name, so no further changes are needed there.
        # --- END OF THE PART YOU NEED TO ADD/MODIFY ---

        self.total_steps = 0

    # alias QNetwork name to DQNNetwork for backwards compatibility with imports
    DQNNetwork = QNetwork

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        state_arr = np.array(state, dtype=np.float32).reshape(1, -1)
        state_t = torch.from_numpy(state_arr).float().to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def train_episode(self):
        state = self.env.reset()[0] if isinstance(self.env.reset(), tuple) else self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < self.config['max_timesteps_per_episode']:
            action = self.act(state)
            step_out = self.env.step(action)
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            elif len(step_out) == 4:
                next_state, reward, done, _ = step_out
            else:
                next_state, reward, done = step_out[:3]
            self.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            self.total_steps += 1
            if len(self.replay_buffer) >= self.min_replay_size and self.total_steps % self.update_every == 0:
                self.learn()
            if self.target_update_every and self.total_steps % self.target_update_every == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return total_reward, steps, self.total_steps

    def learn(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # convert numpy arrays to tensors and move to the agent device
        # using torch.from_numpy avoids an extra copy compared with torch.FloatTensor
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().unsqueeze(1).to(self.device)

        with torch.no_grad():
            if self.double_dqn:
                argmax_actions = torch.argmax(self.q_net(next_states), dim=1, keepdim=True)
                q_next = self.target_net(next_states).gather(1, argmax_actions)
            else:
                q_next = self.target_net(next_states).max(1, keepdim=True)[0]
            target = rewards + self.gamma * (1 - dones) * q_next

        q_values = self.q_net(states).gather(1, actions)
        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.clip_grad)
        self.optimizer.step()