#!/usr/bin/env python3
"""
record_and_run.py

Usage example:
  python record_and_run.py --model results/final_model_dqn.pth --env LunarLander-v3 --episodes 2

This script:
 - loads config.yaml (for building the agent)
 - constructs the agent (agents.dqn_agent.DQNAgent)
 - loads the checkpoint from --model (if given)
 - creates the env with render_mode='rgb_array'
 - wraps with RecordVideo and records `episodes` episodes
 - prints the folder where the mp4s are saved and lists the mp4 files
"""
import argparse
import os
import time
import glob

import gymnasium as gym
import numpy as np
import torch
import yaml

from agents.dqn_agent import DQNAgent

def record_video_wrapper(env, env_name, agent_name, video_folder='results/videos', episodes_to_record=1):
    run_id = time.strftime("%Y%m%d-%H%M%S")
    video_path = os.path.join(video_folder, f'{agent_name}-{env_name}-{run_id}')
    os.makedirs(video_path, exist_ok=True)
    wrapped_env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_path,
        episode_trigger=lambda episode_id: episode_id < episodes_to_record,
        disable_logger=True
    )
    print(f"Recording enabled; videos will be saved to: {video_path}")
    return wrapped_env, video_path

def build_agent_from_config(state_size, action_size, device="cpu"):
    # load config.yaml
    cfg = {}
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f) or {}
    dqn_cfg = cfg.get("dqn", {})
    hidden_layers = dqn_cfg.get("hidden_layers", [128,128])
    lr = dqn_cfg.get("lr", 2.5e-4)
    batch_size = dqn_cfg.get("batch_size", 256)
    buffer_size = int(dqn_cfg.get("buffer_size", 2e5))
    gamma = dqn_cfg.get("gamma", 0.99)
    tau = dqn_cfg.get("tau", 1e-3)
    update_every = dqn_cfg.get("update_every", 4)
    min_replay_size = int(dqn_cfg.get("min_replay_size", 1000))
    double_dqn = bool(dqn_cfg.get("double_dqn", True))
    clip_grad = dqn_cfg.get("clip_grad", 0.5)
    target_update_every = dqn_cfg.get("target_update_every", None)

    agent = DQNAgent(state_size=state_size,
                     action_size=action_size,
                     device=device,
                     hidden_layers=hidden_layers,
                     seed=cfg.get("seed", 42),
                     lr=lr,
                     batch_size=batch_size,
                     buffer_size=buffer_size,
                     gamma=gamma,
                     tau=tau,
                     update_every=update_every,
                     min_replay_size=min_replay_size,
                     double_dqn=double_dqn,
                     clip_grad=clip_grad,
                     target_update_every=target_update_every)
    return agent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=False, default=None, help="Path to model checkpoint to load")
    p.add_argument("--env", default="LunarLander-v3", help="Gym env id")
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to record")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    env_name = args.env
    episodes = int(args.episodes)
    device = args.device

    # create env with rgb array rendering mode
    env = gym.make(env_name, render_mode="rgb_array")
    obs_space = env.observation_space
    act_space = env.action_space
    state_size = obs_space.shape[0]
    action_size = act_space.n

    # build agent
    agent = build_agent_from_config(state_size, action_size, device=device)

    # load model if provided
    if args.model:
        if os.path.exists(args.model):
            try:
                agent.load(args.model)
                print(f"Loaded model: {args.model}")
            except Exception as e:
                print(f"Warning: failed to load model {args.model}: {e}")
        else:
            print(f"Warning: model path does not exist: {args.model}")

    # wrap env to record first `episodes` episodes
    wrapped_env, video_path = record_video_wrapper(env, env_name, agent_name="dqn_agent", video_folder="results/videos", episodes_to_record=episodes)

    # run deterministic episodes (epsilon=0)
    for ep in range(episodes):
        reset_out = wrapped_env.reset()
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done = False
        steps = 0
        while not done:
            # get deterministic action; support different signatures
            try:
                action = agent.act(state, epsilon=0.0)
            except TypeError:
                action = agent.act(state)
            action_val = int(action.item()) if hasattr(action, "item") else int(action)
            step_out = wrapped_env.step(action_val)
            if len(step_out) == 5:
                state, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            elif len(step_out) == 4:
                state, reward, done, info = step_out
            else:
                state = step_out[0]
            steps += 1
            if steps > 10000:
                break

    wrapped_env.close()

    mp4s = sorted(glob.glob(os.path.join(video_path, "*.mp4")))
    if len(mp4s) == 0:
        print("No mp4 files were created. Check that the env supports render_mode='rgb_array' and that episodes finished.")
    else:
        print("Saved videos:")
        for pth in mp4s:
            print("  ", pth)

if __name__ == "__main__":
    main()