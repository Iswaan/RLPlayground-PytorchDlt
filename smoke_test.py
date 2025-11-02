import gymnasium as gym
import torch
from agents import A2CAgent, ActorCriticNetwork, PPOAgent, PPONetwork, DQNAgent, DQNNetwork

print('Running smoke tests for agents...')

# Try preferred env versions (gym vs gymnasium registries may have v3 or v2).
env_candidates = ['LunarLander-v3', 'LunarLander-v2']
env = None
last_exc = None
for candidate in env_candidates:
	try:
		env = gym.make(candidate)
		print(f"Using environment: {candidate}")
		break
	except Exception as e:
		last_exc = e
if env is None:
	raise last_exc
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# A2C
net_a2c = ActorCriticNetwork(obs_dim, action_dim)
agent_a2c = A2CAgent(env, net_a2c)
print('A2C act:', agent_a2c.act(env.reset()[0]))

# PPO
net_ppo = PPONetwork(obs_dim, action_dim)
agent_ppo = PPOAgent(env, net_ppo)
print('PPO act:', agent_ppo.act(env.reset()[0]))

# DQN
# Option A: provide a network instance
net_dqn = DQNNetwork(obs_dim, action_dim)
agent_dqn = DQNAgent(env=env, network=net_dqn)
print('DQN act:', agent_dqn.act(env.reset()[0]))

print('Smoke tests completed successfully.')
