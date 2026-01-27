# linearq_features.py

import os
import numpy as np
import pandas as pd

from TestEnv import HydroElectric_Test
from agents.agent_linearq import LinearQPolicy
from helpers.feature_extractor_cont import FeatureExtractorCont
from helpers.obs_utils import parse_observation
from helpers.eval_utils import evaluate_policy
from helpers.plot_functions import (
    plot_cumulative_profit,
    plot_dam_level,
    plot_action_vs_price,
    plot_mean_action_by_hour,
    plot_q_value_heatmap,
    plot_policy_heatmap,
)

# paths
QT_DIR = "qtables"
os.makedirs(QT_DIR, exist_ok=True)

alg_name = "linearq"
IMG_DIR = os.path.join("img", alg_name)
os.makedirs(IMG_DIR, exist_ok=True)

# config
MAX_VOLUME = 100_000
N_VOL_BINS = 8

ACTIONS = {0: -1.0, 1: 0.0, 2: 1.0}

# =====================
# LOAD TRAIN DATA
# =====================
train = pd.read_excel("train.xlsx").rename(columns={"PRICES": "Date"})
train["Date"] = pd.to_datetime(train["Date"])

HOUR_COLS = [f"Hour {h:02d}" for h in range(1, 25)]

train_long = train.melt(
    id_vars=["Date"],
    value_vars=HOUR_COLS,
    var_name="Hour",
    value_name="Price"
)

PRICE_BINS = np.quantile(
    train_long["Price"],
    [0.15, 0.3, 0.5, 0.7, 0.85]
)

# feature extractor
extractor = FeatureExtractorCont(max_volume=MAX_VOLUME)

# =====================
# DISCRETIZE (for plotting)
# =====================
def discretize_for_plot(observation):

    obs = parse_observation(observation)

    v = obs["volume"] / MAX_VOLUME
    volume_bin = int(np.clip(v * N_VOL_BINS, 0, N_VOL_BINS - 1))

    price_bin = int(np.digitize(obs["price"], PRICE_BINS))

    hour_bin = obs["hour"] - 1
    weekday_bin = obs["weekday"]

    return (volume_bin, price_bin, hour_bin, weekday_bin)

# linear doesnt have qtable so we do this 
def linear_to_qtable(agent, visited_states, discretize_fn):

    from collections import defaultdict

    Q = defaultdict(lambda: np.zeros(agent.n_actions))
    counts = defaultdict(int)

    # reset feature memory
    if hasattr(agent.feature_fn, "reset"):
        agent.feature_fn.reset()

    for obs in visited_states:

        s = discretize_fn(obs)

        phi = agent.feature_fn(obs)
        q_vals = agent.q_values(phi)

        Q[s] += q_vals
        counts[s] += 1

    # average
    for s in Q:
        Q[s] /= counts[s]

    return dict(Q)



def make_agent(train=False):

    agent = LinearQPolicy(
        feature_fn=extractor,
        actions=ACTIONS,
        alpha=2e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.97,
        n_episodes=200,
        env_class=HydroElectric_Test,
        train_path="train.xlsx",
    )

    MODEL_PATH = os.path.join(QT_DIR, "qtable_linear.npy")

    if train or not os.path.exists(MODEL_PATH):

        print("[Linear Q] Training model...")
        agent.train()

        np.save(MODEL_PATH, agent.W)
        print(f"[Linear Q] Saved to {MODEL_PATH}")

    else:

        print(f"[Linear Q] Loading model from {MODEL_PATH}")
        agent.W = np.load(MODEL_PATH).astype(np.float32)

    # reset feature history
    if hasattr(extractor, "reset"):
        extractor.reset()

    # greedy for eval
    agent.epsilon = 0.0

    return agent

# =====================
# MAIN
# =====================

if __name__ == "__main__":

    policy = make_agent(train=False)

    # validate
    env = HydroElectric_Test(path_to_test_data="validate.xlsx")
    results = evaluate_policy(env, policy)

    profit = results["cum_rewards"][-1]
    print(f"Linear-Q validation profit: {profit:.2f} EUR")

    # build pseudo Q-table
    Q_plot = linear_to_qtable(
        policy,
        results["visited_states"],
        discretize_for_plot
    )

    # plots
    plot_cumulative_profit(
        results["cum_rewards"],
        IMG_DIR,
        "linear_cumulative_profit.png",
        "Linear Q-learning: cumulative profit (validation)"
    )

    plot_dam_level(
        results["dam_levels"],
        IMG_DIR,
        "linear_dam_level.png",
        "Linear Q-learning: dam level over time"
    )

    plot_action_vs_price(
        results["prices"],
        results["actions"],
        IMG_DIR,
        "linear_action_vs_price.png",
        "Linear Q-learning: action vs price"
    )

    plot_mean_action_by_hour(
        results["actions"],
        IMG_DIR,
        "linear_mean_action_by_hour.png",
        "Linear Q-learning: mean action by hour"
    )

    plot_q_value_heatmap(
        Q_plot,
        PRICE_BINS,
        IMG_DIR,
        "linear_q_value_heatmap.png",
        "Linear Q-learning: value heatmap"
    )

    plot_policy_heatmap(
        Q_plot,
        PRICE_BINS,
        IMG_DIR,
        "linear_policy_heatmap.png",
        "Linear Q-learning: policy heatmap"
    )


def load_agent():
    return make_agent(train=False)
