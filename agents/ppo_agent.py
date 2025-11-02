import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_agent import BaseAgent
from torch.distributions import Categorical
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class PPONetwork(nn.Module):
    """Actor-Critic network architecture for PPO."""
    def __init__(self, input_dim, output_dim, hidden_layers=[128,128], activation=nn.ReLU):
        super(PPONetwork, self).__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
        self.feature = nn.Sequential(*layers)
        self.actor = nn.Linear(last_dim, output_dim)
        self.critic = nn.Linear(last_dim, 1)

    def forward(self, x):
        x = self.feature(x)
        return self.actor(x), self.critic(x)

class PPOAgent(BaseAgent):
    """Proximal Policy Optimization Agent."""
    def __init__(self, env, network, lr=1e-3, gamma=0.99, clip_epsilon=0.2, device='cpu', save_path='results/', config=None):
        super().__init__(env=env, network=network, lr=lr, gamma=gamma, device=device, save_path=save_path, config=config)
        self.clip_epsilon = clip_epsilon
        # --- PPO Hyperparameters from config ---
        ppo_config = (config or {}).get('ppo', {})
        self.n_steps = int(ppo_config.get('n_steps', 2048))
        self.n_epochs = int(ppo_config.get('n_epochs', 10))
        self.batch_size = int(ppo_config.get('batch_size', 64))
        self.clip_epsilon = float(ppo_config.get('clip_epsilon', getattr(self, 'clip_epsilon', 0.2)))
        self.gae_lambda = float(ppo_config.get('gae_lambda', 0.95))
        self.ent_coef = float(ppo_config.get('ent_coef', 0.01))

        # <--- FIX: Added buffers to correctly track episodic rewards ---
        # This fixes the bug where PPO reported step-reward instead of episode-reward
        self.true_episode_rewards = []
        self.current_episode_reward = 0

        # Initialize state for data collection
        try:
            reset_out = self.env.reset()
            self.current_state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        except Exception:
            self.current_state = self.env.reset()

    def act(self, state):
        """Selects a greedy action for testing/evaluation."""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.dim() == 1:
             state_tensor = state_tensor.unsqueeze(0)
             
        with torch.no_grad():
            logits, _ = self.network(state_tensor)
            
        action_prob = torch.softmax(logits.squeeze(0), dim=-1)
        action = torch.argmax(action_prob).item() 
        return action
    
    def collect_data(self):
        """Collects trajectories up to n_steps using the current policy."""
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        steps_collected = 0
        
        while steps_collected < self.n_steps:
            state_tensor = torch.tensor(self.current_state, dtype=torch.float32, device=self.device)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            with torch.no_grad():
                logits, value = self.network(state_tensor)
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            states.append(self.current_state)
            actions.append(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.squeeze(0))
            dones.append(done)
            
            self.current_state = next_state
            self.current_episode_reward += reward
            steps_collected += 1
            
            if done:
                self.true_episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0
                self.current_state, _ = self.env.reset()

        # Get final value for GAE calculation
        last_state_tensor = torch.tensor(self.current_state, dtype=torch.float32, device=self.device)
        if last_state_tensor.dim() == 1:
            last_state_tensor = last_state_tensor.unsqueeze(0)
        with torch.no_grad():
            _, last_value = self.network(last_state_tensor)
            last_value = last_value.squeeze(0).detach()

        # Calculate Advantages and Returns (using GAE)
        advantages = self.calculate_gae(rewards, values, last_value, dones)
        values_tensor = torch.stack(values).squeeze().detach()
        
        # <--- FIX: This is the main bug fix. Squeeze advantages from [2048, 1] to [2048].
        returns = advantages.squeeze(-1) + values_tensor
        
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        log_probs = torch.cat(log_probs).detach().to(self.device)

        # <--- FIX: Changed what is returned to fix the logging bug ---
        # We now return the mean episodic reward from the buffer
        mean_ep_reward = np.mean(self.true_episode_rewards) if len(self.true_episode_rewards) > 0 else -500.0
        self.true_episode_rewards = [] # Clear the buffer

        return states, actions, log_probs, advantages, returns, mean_ep_reward

    def calculate_gae(self, rewards, values, last_value, dones, lam=0.95):
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = []
        gae = 0.0
        
        # Convert values to detached numpy for calculation
        values_np = [v.cpu().detach().numpy() for v in values] + [last_value.cpu().detach().numpy()]

        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step] # 1.0 if not done, 0.0 if done
            # Ensure values are scalars
            val = values_np[step].item()
            next_val = values_np[step + 1].item()
            
            delta = rewards[step] + self.gamma * next_val * mask - val
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)

        # <--- FIX: More efficient tensor creation ---
        # This also fixes the UserWarning you saw
        return torch.tensor(advantages, dtype=torch.float32, device=self.device).unsqueeze(-1)
    
    def update_ppo(self, states, actions, old_log_probs, advantages, returns):
        """Performs multi-epoch PPO update with clipping."""
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.n_epochs):
            for state_batch, action_batch, old_log_prob_batch, adv_batch, return_batch in loader:
                
                logits, value_pred = self.network(state_batch)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(action_batch)
                
                # <--- FIX: Added Entropy calculation ---
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - old_log_prob_batch)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                
                # <--- FIX: Squeeze adv_batch to match ratio ---
                policy_loss = -torch.min(ratio * adv_batch.squeeze(-1), clipped_ratio * adv_batch.squeeze(-1)).mean()
                
                # <--- FIX: Ensure value_pred is squeezed correctly to match return_batch ---
                value_loss = F.mse_loss(value_pred.squeeze(-1), return_batch) # Use squeeze(-1)
                
                # <--- FIX: Added Entropy Bonus to the loss ---
                loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
    def train_episode(self):
        """Runs one full PPO iteration (Collect -> Update)."""
        
        # 1. Data Collection
        # <--- FIX: Correctly unpack the mean episodic reward ---
        states, actions, log_probs, advantages, returns, mean_ep_reward = self.collect_data()
        
        # 2. PPO Update (Multi-Epoch Minibatch)
        self.update_ppo(states, actions, log_probs, advantages, returns)
        
        # <--- FIX: Pass the correct episodic reward to the BaseAgent logger ---
        self.episode_rewards.append(mean_ep_reward)

        # Model Saving Logic
        if mean_ep_reward > self.best_reward:
            self.best_reward = mean_ep_reward
            self.save_model(f'best_{self.agent_name}.pth')

        ep_len = int(states.shape[0]) if hasattr(states, 'shape') else 0
        timesteps = ep_len
        # <--- FIX: Return the correct mean episodic reward ---
        return mean_ep_reward, ep_len, timesteps

