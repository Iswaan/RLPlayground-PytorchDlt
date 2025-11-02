import torch
import json
from agents.a2c_agent import ActorCriticNetwork, A2CAgent
from agents.ppo_agent import PPONetwork, PPOAgent
from agents.dqn_agent import DQNNetwork, DQNAgent
import gym

# ---- Load best agent info ----
with open('models/best_agent.json', 'r') as f:
    best_info = json.load(f)

agent_type = best_info['name']
model_file = best_info['model_file']

print(f"Playing best agent: {agent_type.upper()} ({model_file})")

# ---- Environment ----
env_name = 'LunarLander-v3'
env = gym.make(env_name)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# ---- Initialize agent ----
if agent_type == 'A2C':
    network = ActorCriticNetwork(input_dim, output_dim)
    agent = A2CAgent(env, network)
elif agent_type == 'PPO':
    network = PPONetwork(input_dim, output_dim)
    agent = PPOAgent(env, network)
elif agent_type == 'DQN':
    network = DQNNetwork(input_dim, output_dim)
    agent = DQNAgent(env, network)
else:
    raise ValueError("Unknown agent type")

# ---- Load model ----
agent.load_model(model_file)

# ---- Play ----
state = env.reset()
done = False
while not done:
    env.render()
    action = agent.act(state)
    state, reward, done, _ = env.step(action)
env.close()
