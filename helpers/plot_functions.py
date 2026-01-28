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


# =========================
# Helper functions
# =========================

def _make_price_labels(price_bins):
    """
    Generate readable labels from price quantiles.
    """  
    labels = []

    labels.append(f"<{price_bins[0]:.0f}")

    for i in range(len(price_bins) - 1):
        labels.append(
            f"{price_bins[i]:.0f}-{price_bins[i+1]:.0f}"
        )

    labels.append(f">{price_bins[-1]:.0f}")

    return labels


def _make_volume_labels(n_vol):
    """
    Generate percentage-based volume labels.
    """
    return [
        f"{int(100*i/n_vol)}–{int(100*(i+1)/n_vol)}%"
        for i in range(n_vol)
    ]


def _setup_heatmap_axes(
    price_bins,
    volume_bins,
    price_quantiles,
    xlabel,
    ylabel,
    title,
):
    """
    Apply consistent axes formatting to heatmaps.
    """

    price_labels = _make_price_labels(price_quantiles)
    volume_labels = _make_volume_labels(len(volume_bins))

    plt.xticks(range(len(price_bins)), price_labels)
    plt.yticks(range(len(volume_bins)), volume_labels)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


# =========================
# Q-value heatmap
# =========================

def plot_q_value_heatmap(
    Q,
    PRICE_BINS,
    out_dir,
    filename,
    title,
):
    """
    Heatmap of V(s)=max_a Q(s,a), averaged over time.
    """

    os.makedirs(out_dir, exist_ok=True)

    from collections import defaultdict

    V_sum = defaultdict(float)
    counts = defaultdict(int)

    # Aggregate values
    for (v, p, h, w), q_vals in Q.items():

        key = (v, p)

        V_sum[key] += np.max(q_vals)
        counts[key] += 1

    volume_bins = sorted({k[0] for k in V_sum})
    price_bins = sorted({k[1] for k in V_sum})

    heatmap = np.full(
        (len(volume_bins), len(price_bins)),
        np.nan
    )

    for (v, p), val in V_sum.items():

        i = volume_bins.index(v)
        j = price_bins.index(p)

        heatmap[i, j] = val / counts[(v, p)]

    # Plot
    plt.figure(figsize=(6, 5))

    im = plt.imshow(heatmap, origin="lower", aspect="auto")
    plt.colorbar(im, label="Average V(s)")

    _setup_heatmap_axes(
        price_bins,
        volume_bins,
        PRICE_BINS,
        xlabel="Electricity price (EUR/MWh)",
        ylabel="Reservoir level (%)",
        title=title,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()


# =========================
# Policy heatmap
# =========================

def plot_policy_heatmap(
    Q,
    PRICE_BINS,
    out_dir,
    filename,
    title,
):
    """
    Heatmap of dominant action per (volume, price),
    averaged over time.
    """

    os.makedirs(out_dir, exist_ok=True)

    from collections import defaultdict

    action_counts = defaultdict(lambda: np.zeros(3))

    # Count best actions
    for (v, p, h, w), q_vals in Q.items():

        key = (v, p)

        best_a = np.argmax(q_vals)

        action_counts[key][best_a] += 1

    volume_bins = sorted({k[0] for k in action_counts})
    price_bins = sorted({k[1] for k in action_counts})

    heatmap = np.full(
        (len(volume_bins), len(price_bins)),
        np.nan
    )

    for (v, p), acts in action_counts.items():

        i = volume_bins.index(v)
        j = price_bins.index(p)

        heatmap[i, j] = np.argmax(acts)

    cmap = ListedColormap(["tab:blue", "tab:gray", "tab:green"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    # Plot
    plt.figure(figsize=(6, 5))

    im = plt.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )

    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Release", "Hold", "Pump"])

    _setup_heatmap_axes(
        price_bins,
        volume_bins,
        PRICE_BINS,
        xlabel="Electricity price (EUR/MWh)",
        ylabel="Reservoir level (%)",
        title=title,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()


# =========================
# State visitation heatmap
# =========================

def plot_state_visitation_heatmap(
    visited_states,
    PRICE_BINS,
    out_dir,
    filename,
    title,
):
    """
    Heatmap of visitation counts over (volume, price),
    aggregated over time.
    """

    os.makedirs(out_dir, exist_ok=True)

    from collections import defaultdict

    counts = defaultdict(int)

    for (v, p, h, w) in visited_states:

        key = (v, p)
        counts[key] += 1

    volume_bins = sorted({k[0] for k in counts})
    price_bins = sorted({k[1] for k in counts})

    heatmap = np.zeros(
        (len(volume_bins), len(price_bins))
    )

    for (v, p), c in counts.items():

        i = volume_bins.index(v)
        j = price_bins.index(p)

        heatmap[i, j] = c

    # Plot
    plt.figure(figsize=(6, 5))

    im = plt.imshow(heatmap, origin="lower", aspect="auto")
    plt.colorbar(im, label="Visit count")

    _setup_heatmap_axes(
        price_bins,
        volume_bins,
        PRICE_BINS,
        xlabel="Electricity price (EUR/MWh)",
        ylabel="Reservoir level (%)",
        title=title,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()

