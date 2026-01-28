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
EPSILON_END = 0.01
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
    # --- Laad training data ---
    train_df = pd.read_excel("train.xlsx").rename(columns={"PRICES": "Date"})
    train_df["Date"] = pd.to_datetime(train_df["Date"])
    HOUR_COLS = [f"Hour {h:02d}" for h in range(1, 25)]
    train_long = train_df.melt(
        id_vars=["Date"],
        value_vars=HOUR_COLS,
        var_name="Hour",
        value_name="Price"
    )

    # --- Dynamische price bins ---
    price_bins = compute_price_bins(train_long["Price"].values, n_bins=5)

    # --- Maak agent ---
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

    # --- Train of load model ---
    if train or not os.path.exists(MODEL_PATH):
        print("[Feature Q] Training model...")
        agent.train()
        np.save(MODEL_PATH, dict(agent.Q))
        print(f"[Feature Q] Saved model to {MODEL_PATH}")
    else:
        print(f"[Feature Q] Loading model from {MODEL_PATH}")
        agent.Q.update(np.load(MODEL_PATH, allow_pickle=True).item())

    agent.epsilon = 0.0
    return agent, price_bins


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
policy, price_bins = make_agent(train=False)

# validate
env = HydroElectric_Test(path_to_test_data="validate.xlsx")
results = evaluate_policy(env, policy)

profit = results["cum_rewards"][-1]

# build pseudo Q-table
Q_plot = linearize_qtable(policy, results["visited_states"], price_bins)

# zorg dat map bestaat
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


def load_agent():
    """
    Entry point for graders.
    Trains and returns a Q-learning agent.
    """
    return make_agent(train=True)

