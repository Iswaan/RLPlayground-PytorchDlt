import argparse
import gymnasium as gym
import torch
import yaml
import os
from agents.dqn_agent import DQNAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable fast debug mode (fewer episodes, smaller batches)')
    parser.add_argument('--preset_name', type=str, default=None, help='Train a specific DQN preset, otherwise all presets will be trained.')
    parser.add_argument('--save_path', type=str, default=None, help='Directory to save results.')
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

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
    env_candidates = [env_name]
    if env_name.endswith('-v3'):
        env_candidates.append(env_name.replace('-v3', '-v2'))
    elif env_name.endswith('-v2'):
        env_candidates.insert(0, env_name.replace('-v2', '-v3'))

    # Environment parameter handling
    env_kwargs = {}
    if "LunarLander" in env_name:
        env_params = config.get('environment_params', {}).get(env_name, {})
        if 'gravity' in env_params:
            env_kwargs['gravity'] = env_params['gravity']
        if 'enable_wind' in env_params and env_params['enable_wind']:
            env_kwargs['enable_wind'] = True
            env_kwargs['wind_power'] = env_params.get('wind_power', 0.0)

    env = None
    last_exc = None
    for candidate in env_candidates:
        try:
            env = gym.make(candidate, **env_kwargs) # Pass env_kwargs here
            print(f"Using environment: {candidate}")
            break
        except Exception as e:
            last_exc = e
    if env is None:
        raise last_exc
    
    try:
        env.reset(seed=config.get('seed', 42))
    except Exception:
        try:
            env.seed(config.get('seed', 42))
        except Exception:
            pass

    presets = config.get('presets', {})
    
    # Logic to train specific preset or all presets
    presets_to_train = {}
    if args.preset_name:
        if args.preset_name == "dqn": # "dqn" is the implicit base config
             presets_to_train["dqn"] = config.get('dqn', {})
        elif args.preset_name in presets:
            presets_to_train = {args.preset_name: presets[args.preset_name]}
        else:
            print(f"Error: Preset '{args.preset_name}' not found in config.yaml.")
            return
    else:
        # If no preset specified, train the base 'dqn' config and then all named presets
        if 'dqn' in config:
            presets_to_train["dqn"] = config.get('dqn', {}) # Use a different key, "base_dqn", to avoid conflict if 'dqn' is also a preset name
        presets_to_train.update(presets)

    sorted_preset_names = sorted(presets_to_train.keys())

    for preset_name_in_loop in sorted_preset_names:
        print(f"===== Training preset: {preset_name_in_loop} =====")
        # Pass save_path to the agent constructor. This requires modifying the agent __init__
        agent = DQNAgent(env, config, preset_name=preset_name_in_loop, save_path=args.save_path)
        agent.train(episodes=config['train_episodes'])
        print(f"===== Finished training preset: {preset_name_in_loop} =====\n")

if __name__ == "__main__":
    main()