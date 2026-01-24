"""
Common plotting utilities for dam operation experiments.
Used to ensure consistent visualisation and fair comparison.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})


def plot_cumulative_profit(cum_rewards, out_dir, filename, title):
    os.makedirs(out_dir, exist_ok=True)

    # downsample for readability
    step = max(len(cum_rewards) // 500, 1)
    x = np.arange(0, len(cum_rewards), step)
    y = cum_rewards[::step]

    plt.figure(figsize=(6, 3))
    plt.plot(x, y, linewidth=2)
    plt.xlabel("Time (hours)")
    plt.ylabel("Cumulative profit (EUR)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()



def plot_dam_level(dam_levels, out_dir, filename, title):
    os.makedirs(out_dir, exist_ok=True)

    step = max(len(dam_levels) // 500, 1)
    x = np.arange(0, len(dam_levels), step)
    y = dam_levels[::step]

    plt.figure(figsize=(6, 3))
    plt.plot(x, y, linewidth=2)
    plt.xlabel("Time (hours)")
    plt.ylabel("Dam level (m$^3$)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()



def plot_action_vs_price(prices, actions, out_dir, filename, title):
    os.makedirs(out_dir, exist_ok=True)

    jitter = np.random.uniform(-0.05, 0.05, size=len(actions))

    plt.figure(figsize=(6, 3))
    plt.scatter(
        prices,
        actions + jitter,
        s=8,
        alpha=0.2
    )
    plt.yticks([-1, 0, 1], ["Produce", "Idle", "Pump"])
    plt.xlabel("Price (EUR/MWh)")
    plt.ylabel("Action")
    plt.title(title)
    plt.ylim(-1.5, 1.5)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()



def plot_mean_action_by_hour(actions, out_dir, filename, title):
    os.makedirs(out_dir, exist_ok=True)

    hours = np.arange(len(actions)) % 24
    mean_action_by_hour = [
        actions[hours == h].mean()
        for h in range(24)
    ]

    plt.figure(figsize=(6, 3))
    plt.bar(range(24), mean_action_by_hour, width=0.8)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Hour of day")
    plt.ylabel("Mean action")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()



# qlearning plots
def plot_q_value_heatmap(
    Q,
    fixed_hour,
    fixed_weekday,
    out_dir,
    filename,
    title,
):
    """
    Heatmap of V(s) = max_a Q(s,a)
    for fixed (hour, weekday), over (volume_bin, price_bin).
    """

    os.makedirs(out_dir, exist_ok=True)

    volume_bins = sorted({s[0] for s in Q.keys()})
    price_bins = sorted({s[1] for s in Q.keys()})

    heatmap = np.full((len(volume_bins), len(price_bins)), np.nan)

    for (v, p, h, w), q_vals in Q.items():
        if h == fixed_hour and w == fixed_weekday:
            i = volume_bins.index(v)
            j = price_bins.index(p)
            heatmap[i, j] = np.max(q_vals)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        heatmap,
        origin="lower",
        aspect="auto"
    )
    plt.colorbar(im, label="V(s) = max Q(s,a)")
    plt.xlabel("Price bin")
    plt.ylabel("Volume bin")
    plt.title(title)
    plt.xticks(range(len(price_bins)), price_bins)
    plt.yticks(range(len(volume_bins)), volume_bins)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()



def plot_policy_heatmap(
    Q,
    fixed_hour,
    fixed_weekday,
    out_dir,
    filename,
    title,
):
    os.makedirs(out_dir, exist_ok=True)

    volume_bins = sorted({s[0] for s in Q.keys()})
    price_bins = sorted({s[1] for s in Q.keys()})

    heatmap = np.full((len(volume_bins), len(price_bins)), np.nan)

    for (v, p, h, w), q_vals in Q.items():
        if h == fixed_hour and w == fixed_weekday:
            i = volume_bins.index(v)
            j = price_bins.index(p)
            heatmap[i, j] = np.argmax(q_vals)

    # discrete colormap for actions
    cmap = ListedColormap(["tab:blue", "tab:gray", "tab:green"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm
    )
    # X-axis: price bins (semantic labels)
    plt.xticks(
        ticks=range(len(price_bins)),
        labels=["Very low", "Low", "High", "Very high"]
    )

    # Y-axis: volume bins (semantic labels)
    plt.yticks(
        ticks=range(len(volume_bins)),
        labels=["Empty", "Low", "Medium", "High", "Full"]
    )

    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Release", "Hold", "Pump"])

    plt.xlabel("Electricity price regime")
    plt.ylabel("Reservoir storage regime")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()


def plot_state_visitation_heatmap(
    visited_states,
    fixed_hour,
    fixed_weekday,
    out_dir,
    filename,
    title,
):
    """
    Heatmap of state visitation counts
    for fixed (hour, weekday), over (volume_bin, price_bin).
    """

    os.makedirs(out_dir, exist_ok=True)

    volume_bins = sorted({s[0] for s in visited_states})
    price_bins = sorted({s[1] for s in visited_states})

    heatmap = np.zeros((len(volume_bins), len(price_bins)))

    for (v, p, h, w) in visited_states:
        if h == fixed_hour and w == fixed_weekday:
            i = volume_bins.index(v)
            j = price_bins.index(p)
            heatmap[i, j] += 1

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        heatmap,
        origin="lower",
        aspect="auto"
    )
    plt.colorbar(im, label="Visit count")
    plt.xlabel("Price bin")
    plt.ylabel("Volume bin")
    plt.title(title)
    plt.xticks(range(len(price_bins)), price_bins)
    plt.yticks(range(len(volume_bins)), volume_bins)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()
