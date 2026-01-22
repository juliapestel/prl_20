"""
Common plotting utilities for dam operation experiments.
Used to ensure consistent visualisation and fair comparison.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_cumulative_profit(cum_rewards, out_dir, filename, title):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(cum_rewards)
    plt.xlabel("Time (hours)")
    plt.ylabel("Cumulative profit (EUR)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


def plot_dam_level(dam_levels, out_dir, filename, title):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(dam_levels)
    plt.xlabel("Time (hours)")
    plt.ylabel("Dam level (m3)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


def plot_action_vs_price(prices, actions, out_dir, filename, title):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.scatter(prices, actions, alpha=0.3)
    plt.yticks([-1, 0, 1], ["Produce", "Idle", "Pump"])
    plt.xlabel("Price (EUR/MWh)")
    plt.ylabel("Action")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


def plot_mean_action_by_hour(actions, out_dir, filename, title):
    os.makedirs(out_dir, exist_ok=True)

    hours = np.arange(len(actions)) % 24
    mean_action_by_hour = [
        actions[hours == h].mean()
        for h in range(24)
    ]

    plt.figure(figsize=(10, 4))
    plt.bar(range(24), mean_action_by_hour)
    plt.xlabel("Hour of day")
    plt.ylabel("Mean action")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()
