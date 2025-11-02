#!/usr/bin/env python3
"""
evaluate_and_plot_dqn.py

Load a checkpoint and run deterministic greedy evaluation episodes.
Appends result to results/eval_rollouts.csv and optionally records an mp4.
"""
import argparse
import os
import csv
import time
import yaml
from datetime import datetime

import gymnasium as gym
import torch
import numpy as np
import imageio

from agents.dqn_agent import DQNAgent

def load_config(path="config.yaml"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def run_eval(model_path, env_name="LunarLander-v3", episodes=5, device="cpu", record=False, out="results/eval_video.mp4"):
    cfg = load_config()
    dqn_cfg = cfg.get("dqn", {})
    hidden_layers = dqn_cfg.get("hidden_layers", [128,128])

    env = gym.make(env_name, render_mode='rgb_array' if record else None)
    state_shape = env.observation_space.shape
    state_size = state_shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size=state_size,
                     action_size=action_size,
                     device=device,
                     hidden_layers=hidden_layers)
    agent.load(model_path)

    frames = []
    rewards = []
    lengths = []

    for ep in range(1, episodes + 1):
        reset_out = env.reset()
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done = False
        total = 0.0
        steps = 0
        while not done:
            action = agent.act(state, epsilon=0.0)  # greedy
            step_out = env.step(action)
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            elif len(step_out) == 4:
                next_state, reward, done, _ = step_out
            else:
                next_state, reward, done = step_out[:3]

            total += reward
            steps += 1
            state = next_state
            if record:
                try:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                except Exception:
                    pass
        rewards.append(total)
        lengths.append(steps)
        print(f"Eval {ep}: reward={total:.2f}, steps={steps}")

    env.close()

    if record and len(frames) > 0:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        imageio.mimsave(out, frames, fps=30)
        print(f"Saved video to {out}")

    mean = float(np.mean(rewards))
    std = float(np.std(rewards))
    timestamp = datetime.utcnow().isoformat()

    os.makedirs("results", exist_ok=True)
    eval_csv = os.path.join("results", "eval_rollouts.csv")
    header = ["timestamp", "model", "eval_episodes", "mean_reward", "std_reward", "mean_length"]
    exists = os.path.exists(eval_csv)
    with open(eval_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow([timestamp, os.path.basename(model_path), episodes, f"{mean:.6f}", f"{std:.6f}", float(np.mean(lengths))])

    print(f"Eval result: mean={mean:.2f} std={std:.2f}")
    return mean, std

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--record", action="store_true")
    p.add_argument("--out", type=str, default="results/eval_video.mp4")
    args = p.parse_args()

    cfg = load_config()
    env_name = cfg.get("env_name", "LunarLander-v3")
    run_eval(args.model, env_name=env_name, episodes=args.episodes, device=args.device, record=args.record, out=args.out)