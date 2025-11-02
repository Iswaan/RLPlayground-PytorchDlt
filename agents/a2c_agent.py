import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .base_agent import BaseAgent
import numpy as np # <-- Added numpy import

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[128,128], activation=nn.Tanh):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
        self.shared = nn.Sequential(*layers)
        self.actor = nn.Linear(last_dim, output_dim)
        self.critic = nn.Linear(last_dim, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

class A2CAgent(BaseAgent):
    def __init__(self, env, network, lr=0.0007, gamma=0.99, device='cpu', save_path='results/', config=None): # <-- Added config
        # Ensure config is passed to BaseAgent for logging and hyperparams
        super().__init__(env=env, network=network, lr=lr, gamma=gamma, device=device, save_path=save_path, config=config) 

    def act(self, state):
        """Selects a greedy action for evaluation."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) # <-- Added unsqueeze
        with torch.no_grad():
             logits, _ = self.network(state_tensor)
             probs = F.softmax(logits, dim=-1)
             action = torch.argmax(probs, dim=-1).item() # Use argmax for deterministic evaluation
        return action

    # In a2c_agent.py

    def train_episode(self):
        """Collects one full episode and performs one update."""
        # --- 1. Collect Episode Trajectory ---
        try:
            reset_out = self.env.reset()
            state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        except Exception:
             state = self.env.reset() # Fallback

        done = False
        states_list = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        logits_list = []
        steps = 0
        total_reward_this_episode = 0
        
        last_truncated = False # <-- ADDED: To track reason for episode end

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device) 
            
            logits, value = self.network(state_t)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            try:
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                done = bool(terminated or truncated)
                last_truncated = bool(truncated) # <-- ADDED: Store if last step was a truncation
            except Exception: # Fallback for older gym versions
                 next_state, reward, done, _ = self.env.step(action.item())
                 last_truncated = False # <-- ADDED: Assume termination on fallback

            states_list.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value.squeeze())
            rewards.append(reward)
            logits_list.append(logits.squeeze(0))

            state = next_state
            steps += 1
            total_reward_this_episode += reward 

        # --- 2. Calculate Returns and Advantages ---
        
        # --- START OF MODIFIED SECTION ---
        R = 0.0
        if last_truncated:
            # If the episode was truncated, bootstrap from the value of the last state
            try:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, last_value = self.network(state_t)
                    R = last_value.item() # Set initial R to V(s_T+1)
            except Exception as e:
                print(f"Warning: Could not bootstrap value for truncated episode. {e}")
                R = 0.0 # Fallback
        
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        # --- END OF MODIFIED SECTION ---

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values_tensor = torch.stack(values).to(self.device)
        log_probs_tensor = torch.stack(log_probs).to(self.device).squeeze()
        logits_tensor = torch.stack(logits_list).to(self.device) 

        advantages = returns - values_tensor 

        # --- 3. Calculate Losses ---
        policy_loss = -(log_probs_tensor * advantages.detach()).mean() 
        value_loss = F.mse_loss(values_tensor, returns) 
        dist_batch = Categorical(logits=logits_tensor)
        entropy_bonus = dist_batch.entropy().mean()

        # --- 4. Get Hyperparameters from Config (with defaults) ---
        a2c_conf = (self.config or {}).get('a2c', {})
        value_coef = a2c_conf.get('value_coef', 0.5) 
        entropy_coef = a2c_conf.get('entropy_coef', 0.01)

        # --- 5. Combine Losses ---
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

        # --- 6. Update Network ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        # --- 7. Logging and Saving ---
        self.episode_rewards.append(total_reward_this_episode)

        if total_reward_this_episode > self.best_reward:
             self.best_reward = total_reward_this_episode
             self.save_model(f'best_{self.agent_name}.pth')

        return total_reward_this_episode, steps, steps