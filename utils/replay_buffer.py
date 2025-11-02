import random
import numpy as np
from collections import deque

class ReplayBuffer:
    """
    Simple numpy-backed replay buffer.

    Methods:
      - add(state, action, reward, next_state, done)
      - sample(batch_size) -> (states, actions, rewards, next_states, dones)
      - __len__()
    """
    def __init__(self, max_size=int(1e5), seed=0):
        self.buffer = deque(maxlen=int(max_size))
        random.seed(seed)
        np.random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        # store as raw Python/numpy objects; conversion to tensors is done in agent
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            raise ValueError(f"Requested batch_size ({batch_size}) > buffer size ({len(self.buffer)})")
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)