import gym
import torch
import yaml
import os
from agents.dqn_agent import DQNAgent

def get_latest_best_model(save_dir):
    # Look for all best models saved by presets
    best_models = []
    for fname in os.listdir(save_dir):
        if fname.startswith("best_") and fname.endswith(".pth"):
            path = os.path.join(save_dir, fname)
            best_models.append((os.path.getmtime(path), path))
    if not best_models:
        raise FileNotFoundError("No best model found in results folder.")
    # Pick the most recently saved best model
    best_models.sort(reverse=True)
    return best_models[0][1]

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    env = gym.make(config['env_name'])
    env.seed(config['seed'])

    # Create agent (preset name doesn't matter since we will load the best model)
    agent = DQNAgent(env, config)

    best_model_path = get_latest_best_model(config['save_dir'])
    print(f"Loading best model: {best_model_path}")
    agent.load_model(best_model_path)

    for ep in range(5):
        state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            action = agent.act(state)
            step_out = env.step(action)
            if len(step_out) == 5:
                state, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            elif len(step_out) == 4:
                state, reward, done, _ = step_out
            else:
                state, reward, done = step_out[:3]
            total_reward += reward
        print(f"Episode {ep+1} reward: {total_reward}")

    env.close()
