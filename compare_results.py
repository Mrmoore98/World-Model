import matplotlib.pyplot as plt
from core.utils import load_progress
import seaborn as sns
import numpy as np

ppo_progress = load_progress("./data/cCarRacing-v0_PPO_12-01_21-39")
fig, ax =plt.subplots(figsize=(20,20))

sns.lineplot(
    data=ppo_progress,
    x="total_steps",
    # y="training_episode_reward/episode_reward_mean",
    y="learning_stats/entropy",
    ax = ax,
    legend='brief',
    label='1',
    estimator= np.mean
)

ppo_progress1 = load_progress("./data/cCarRacing-v0_PPO_12-10_11-29")
sns.lineplot(
    data=ppo_progress1,
    x="total_steps",
    # y="training_episode_reward/episode_reward_mean",
    y="learning_stats/entropy",
    ax= ax,
    legend='brief',
    label='2'
)

ax.set_title("PPO training reward in cCarRacing-v0")
ax.set_ylabel("Mean Rewards")
ax.set_xlabel("Sampled Steps")
plt.legend()
plt.show()
