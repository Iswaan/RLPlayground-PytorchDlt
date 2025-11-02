import argparse
import gymnasium as gym
import torch
import yaml
from agents.dqn_agent import DQNAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable fast debug mode (fewer episodes, smaller batches)')
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # If debug flag is set, override a few config values for fast runs
    if args.debug:
        print("[DEBUG] Debug mode enabled: overriding config for faster runs")
        config['train_episodes'] = 10
        cfg_dqn = config.get('dqn', {})
        cfg_dqn['batch_size'] = 32
        cfg_dqn['buffer_size'] = 5000
        cfg_dqn['min_replay_size'] = 1000
        cfg_dqn['update_every'] = 8
        config['dqn'] = cfg_dqn

    env_name = config.get('env_name', 'LunarLander-v3')
    # try a few env versions (v3 preferred, fall back to v2)
    env_candidates = [env_name]
    if env_name.endswith('-v3'):
        env_candidates.append(env_name.replace('-v3', '-v2'))
    elif env_name.endswith('-v2'):
        env_candidates.insert(0, env_name.replace('-v2', '-v3'))

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
    # gym/gymnasium compatibility: prefer reset(seed=...) instead of env.seed()
    try:
        env.reset(seed=config.get('seed', 42))
    except Exception:
        try:
            env.seed(config.get('seed', 42))
        except Exception:
            pass

    presets = config.get('presets', {})
    for preset_name in presets.keys():
        print(f"===== Training preset: {preset_name} =====")
        agent = DQNAgent(env, config, preset_name=preset_name)
        agent.train(episodes=config['train_episodes'])
        print(f"===== Finished training preset: {preset_name} =====\n")

if __name__ == "__main__":
    main()
