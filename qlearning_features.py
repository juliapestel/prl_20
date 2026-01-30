import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TestEnv import HydroElectric_Test
from agents.agent_qlearning import QLearningPolicy
from helpers.obs_utils import parse_observation
from helpers.eval_utils import evaluate_policy
from collections import defaultdict
from helpers.plot_functions import (
    plot_cumulative_profit,
    plot_dam_level,
    plot_action_vs_price,
    plot_mean_action_by_hour,
    plot_q_value_heatmap,
    plot_policy_heatmap,
    plot_state_visitation_heatmap
)


from collections import defaultdict
from matplotlib.colors import ListedColormap, BoundaryNorm

def _price_labels_extreme():
    return ["Low", "Medium", "High"]


def _volume_labels(n_bins):
    return [
        f"{int(100*i/n_bins)}–{int(100*(i+1)/n_bins)}%"
        for i in range(n_bins)
    ]


def _setup_axes(volume_bins, price_bins, title):

    plt.xticks(
        range(len(price_bins)),
        _price_labels_extreme()
    )

    plt.yticks(
        range(len(volume_bins)),
        _volume_labels(len(volume_bins))
    )

    plt.xlabel("Electricity price category")
    plt.ylabel("Reservoir level (%)")
    plt.title(title)


def plot_state_visitation_heatmap_features(
    visited_states,
    price_bins,
    out_dir,
    filename,
    title,
):

    os.makedirs(out_dir, exist_ok=True)

    counts = defaultdict(int)

    # Discretize observations 
    for obs in visited_states:

        v, p, h, w = discretize_observation(obs, price_bins)

        counts[(v, p)] += 1


    volume_bins = sorted({k[0] for k in counts})
    price_bins_plot = sorted({k[1] for k in counts})

    heatmap = np.zeros((len(volume_bins), len(price_bins_plot)))

    for (v, p), c in counts.items():

        i = volume_bins.index(v)
        j = price_bins_plot.index(p)

        heatmap[i, j] = c

    plt.figure(figsize=(6, 5))

    im = plt.imshow(
        heatmap,
        origin="lower",
        aspect="auto"
    )

    plt.colorbar(im, label="Visit count")

    _setup_axes(volume_bins, price_bins_plot, title)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()



def plot_value_heatmap_features(
    Q,
    out_dir,
    filename,
    title,
):

    os.makedirs(out_dir, exist_ok=True)

    V_sum = defaultdict(float)
    counts = defaultdict(int)

    # Marginalize over time
    for (v, p, h, w), q_vals in Q.items():

        V = np.max(q_vals)

        key = (v, p)

        V_sum[key] += V
        counts[key] += 1

    volume_bins = sorted({k[0] for k in V_sum})
    price_bins_plot = sorted({k[1] for k in V_sum})

    heatmap = np.full(
        (len(volume_bins), len(price_bins_plot)),
        np.nan
    )

    for (v, p), val in V_sum.items():

        i = volume_bins.index(v)
        j = price_bins_plot.index(p)

        heatmap[i, j] = val / counts[(v, p)]

    plt.figure(figsize=(6, 5))

    im = plt.imshow(
        heatmap,
        origin="lower",
        aspect="auto"
    )

    plt.colorbar(im, label="Average V(s)")

    _setup_axes(volume_bins, price_bins_plot, title)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()


def plot_policy_heatmap_features(
    Q,
    out_dir,
    filename,
    title,
):

    os.makedirs(out_dir, exist_ok=True)

    action_counts = defaultdict(lambda: np.zeros(3))

    for (v, p, h, w), q_vals in Q.items():

        best_a = np.argmax(q_vals)

        action_counts[(v, p)][best_a] += 1

    volume_bins = sorted({k[0] for k in action_counts})
    price_bins_plot = sorted({k[1] for k in action_counts})

    heatmap = np.full(
        (len(volume_bins), len(price_bins_plot)),
        np.nan
    )

    for (v, p), acts in action_counts.items():

        i = volume_bins.index(v)
        j = price_bins_plot.index(p)

        heatmap[i, j] = np.argmax(acts)

    cmap = ListedColormap([
        "tab:blue",   # Produce
        "tab:gray",   # Idle
        "tab:green",  # Pump
    ])

    norm = BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5],
        cmap.N
    )

    plt.figure(figsize=(6, 5))

    im = plt.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )

    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Produce", "Idle", "Pump"])

    _setup_axes(volume_bins, price_bins_plot, title)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()



# extractor = FeatureExtractorCont(max_volume=MAX_VOLUME)
def compute_price_bins(prices, n_bins=5):
    """
    Compute price bins based on quantiles of the given prices.
    Returns an array of bin edges.
    """
    return np.quantile(prices, np.linspace(0, 1, n_bins + 1)[1:-1])

QT_DIR = "qtables"
os.makedirs(QT_DIR, exist_ok=True)


alg_name = "qlearning_features"
img_root = "img"
IMG_DIR = os.path.join(img_root, alg_name)

MAX_VOLUME = 100_000  # m3

N_EPISODES = 150
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.97

ACTIONS = {
    0: -1.0,
    1:  0.0,
    2:  1.0
}
N_ACTIONS = len(ACTIONS)

train = pd.read_excel("train.xlsx").rename(columns={"PRICES": "Date"})
train["Date"] = pd.to_datetime(train["Date"])

HOUR_COLS = [f"Hour {h:02d}" for h in range(1, 25)]

train_long = train.melt(
    id_vars=["Date"],
    value_vars=HOUR_COLS,
    var_name="Hour",
    value_name="Price"
)

# price bins from training distribution
PRICE_BINS = np.quantile(train_long["Price"],
                         [0.15, 0.3, 0.5, 0.7, 0.85])

# extractor = FeatureExtractor(max_volume=MAX_VOLUME)


def discretize_observation(observation, price_bins=None):
    obs = parse_observation(observation)

    if price_bins is None:
        raise ValueError("price_bins cannot be None. Generate them from your dataset!")

    # Volume bins
    volume_bin = int(np.clip(obs["volume"] / MAX_VOLUME * 8, 0, 7))

    # Price bins
    price_bin = int(np.digitize(obs["price"], price_bins))
    price_extreme = 0 if price_bin <= 1 else 2 if price_bin >= 4 else 1

    # Hour group
    hour_group = (obs["hour"] - 1) * 4 // 24

    # Weekday
    weekday_bin = obs["weekday"]

    return (volume_bin, price_extreme, hour_group, weekday_bin)




def make_agent(train=False):

    train_df = pd.read_excel("train.xlsx").rename(columns={"PRICES": "Date"})
    train_df["Date"] = pd.to_datetime(train_df["Date"])
    HOUR_COLS = [f"Hour {h:02d}" for h in range(1, 25)]
    train_long = train_df.melt(
        id_vars=["Date"],
        value_vars=HOUR_COLS,
        var_name="Hour",
        value_name="Price"
    )


    price_bins = compute_price_bins(train_long["Price"].values, n_bins=5)

    agent = QLearningPolicy(
        discretize_fn=lambda obs: discretize_observation(obs, price_bins=price_bins),
        actions=ACTIONS,
        n_actions=N_ACTIONS,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        n_episodes=N_EPISODES,
        env_class=HydroElectric_Test,
        train_path="train.xlsx",
    )

    MODEL_PATH = os.path.join(QT_DIR, "qtable_features.npy")


    if train or not os.path.exists(MODEL_PATH):
        print("[Feature Q] Training model...")
        agent.train()
        np.save(MODEL_PATH, dict(agent.Q))
        print(f"[Feature Q] Saved model to {MODEL_PATH}")
    else:
        print(f"[Feature Q] Loading model from {MODEL_PATH}")
        agent.Q.update(np.load(MODEL_PATH, allow_pickle=True).item())

    agent.epsilon = 0.0
    agent.price_bins = price_bins  
    return agent



def linearize_qtable(agent, visited_states, price_bins):
    """
    Convert feature-based Q table to a pseudo Q-table for plotting.
    """
    Q_plot = defaultdict(lambda: np.zeros(len(ACTIONS)))

    counts = defaultdict(int)

    for obs in visited_states:
        # Discrete state volgens dezelfde logica als tijdens training
        state = discretize_observation(obs, price_bins)
        q_vals = agent.Q[state]
        Q_plot[state] += q_vals
        counts[state] += 1

    for state in Q_plot:
        Q_plot[state] /= counts[state]

    return dict(Q_plot)

# agent en price bins ophalen
policy = make_agent(train=False)
price_bins = policy.price_bins


# validate
env = HydroElectric_Test(path_to_test_data="validate.xlsx")
results = evaluate_policy(env, policy)

profit = results["cum_rewards"][-1]

# build pseudo Q-table
Q_plot = linearize_qtable(policy, results["visited_states"], price_bins)
 

alg_name = "qlearning_features"
IMG_DIR = os.path.join(os.path.dirname(__file__), "img", alg_name)
os.makedirs(IMG_DIR, exist_ok=True)

# plots
plot_cumulative_profit(results["cum_rewards"], IMG_DIR, "ftr_cumulative_profit.png",
                           "Linear Q-learning: cumulative profit (validation)")
    

plot_dam_level(results["dam_levels"], IMG_DIR, "ftr_dam_level.png",
                   "Linear Q-learning: dam level over time")
plot_action_vs_price(results["prices"], results["actions"], IMG_DIR, "ftr_action_vs_price.png",
                         "Linear Q-learning: action vs price")
plot_mean_action_by_hour(results["actions"], IMG_DIR, "ftr_mean_action_by_hour.png",
                             "Linear Q-learning: mean action by hour")




plot_state_visitation_heatmap_features(
    results["visited_states"],
    price_bins,
    IMG_DIR,
    "ftr_state_visitation.png",
    "Feature Q-learning: State visitation"
)


plot_value_heatmap_features(
    Q_plot,
    IMG_DIR,
    "ftr_value_heatmap.png",
    "Feature Q-learning: Value function"
)


plot_policy_heatmap_features(
    Q_plot,
    IMG_DIR,
    "ftr_policy_heatmap.png",
    "Feature Q-learning: Policy"
)


plot_cumulative_profit(
    results["cum_rewards"],
    IMG_DIR,
    "ft_cumulative_profit.png",
    "Feature Q: cumulative profit"
)

plot_dam_level(
    results["dam_levels"],
    IMG_DIR,
    "ft_dam_level.png",
    "Feature Q: dam level"
)

plot_action_vs_price(
    results["prices"],
    results["actions"],
    IMG_DIR,
    "ft_action_vs_price.png",
    "Feature Q: action vs price"
)

plot_mean_action_by_hour(
    results["actions"],
    IMG_DIR,
    "ft_mean_action_by_hour.png",
    "Feature Q: mean action"
)



def load_agent():
    return make_agent(train=True)
