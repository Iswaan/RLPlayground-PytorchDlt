import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_rollouts(csv_path='results/dqn_rollouts.csv', save_fig_path='results/rewards_plot.png', ma_window=100):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run training first to generate logs.")

    df = pd.read_csv(csv_path)
    # Expect columns: episode, ep_len, ep_reward, fps, time_elapsed, total_timesteps
    if 'ep_reward' in df.columns:
        reward_col = 'ep_reward'
    elif 'ep_rew' in df.columns:
        reward_col = 'ep_rew'
    else:
        # try common names
        reward_col = 'ep_reward' if 'ep_reward' in df.columns else df.columns[2]

    rewards = df[reward_col].astype(float)
    episodes = df['episode'] if 'episode' in df.columns else range(1, len(rewards) + 1)
    ma = rewards.rolling(window=ma_window, min_periods=1).mean()

    plt.figure(figsize=(10,6))
    plt.plot(episodes, rewards, label='Episode reward', alpha=0.4)
    plt.plot(episodes, ma, label=f'{ma_window}-episode MA', color='tab:orange', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training rewards')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
    plt.savefig(save_fig_path, bbox_inches='tight')
    print(f"Saved reward plot to {save_fig_path}")
    plt.show()

if __name__ == "__main__":
    plot_rollouts()