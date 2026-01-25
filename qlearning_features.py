import os
import numpy as np
import pandas as pd

from TestEnv import HydroElectric_Test
from agents.agent_qlearning import QLearningPolicy
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

alg_name = "qlearning_features"
img_root = "img"
IMG_DIR = os.path.join(img_root, alg_name)

MAX_VOLUME = 100_000  # m3

N_EPISODES = 50
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.95

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
PRICE_BINS = np.quantile(train_long["Price"], [0.25, 0.5, 0.75])

def discretize_observation(observation):
    """
    Feature-engineered discretisation using ONLY available env features.
    Fully compatible with existing plots & env.
    """
    obs = parse_observation(observation)

  
    v_ratio = obs["volume"] / MAX_VOLUME
    if v_ratio < 0.1:
        volume_bin = 0
    elif v_ratio < 0.3:
        volume_bin = 1
    elif v_ratio < 0.7:
        volume_bin = 2
    elif v_ratio < 0.9:
        volume_bin = 3
    else:
        volume_bin = 4


    price_bin = int(np.digitize(obs["price"], PRICE_BINS))

    if price_bin == 0:
        price_extreme = 0      # very low
    elif price_bin == 3:
        price_extreme = 2      # very high
    else:
        price_extreme = 1      # mid


    h = obs["hour"]
    if h <= 6:
        hour_group = 0         # night
    elif h <= 12:
        hour_group = 1         # morning
    elif h <= 18:
        hour_group = 2         # afternoon
    else:
        hour_group = 3         # evening

    weekday_bin = obs["weekday"]
    hour_bin = obs["hour"] - 1   # keep for plots

    return (
        volume_bin,
        price_bin,
        price_extreme,
        hour_group,
        hour_bin,
        weekday_bin,
    )

# kan hetezelfde als in q_learning.py,  maar dan wel die andere discretized observations 
def make_agent():
    """
    Create and train a tabular Q-learning agent
    using feature-engineered discretisation.
    """
    agent = QLearningPolicy(
        discretize_fn=discretize_observation,
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

    agent.train()
    return agent

def reduce_Q_for_plotting(Q):
    """
    Reduce high-dimensional Q-table to 4D (v, p, h, w)
    by averaging over the extra feature dimensions.
    """
    from collections import defaultdict

    Q_reduced = defaultdict(lambda: np.zeros(len(ACTIONS)))
    counts = defaultdict(int)

    for state, q_vals in Q.items():
        v, p, _, _, h, w = state   # negeer extra features
        key = (v, p, h, w)

        Q_reduced[key] += q_vals
        counts[key] += 1

    for key in Q_reduced:
        Q_reduced[key] /= counts[key]

    return dict(Q_reduced)

# precies hetzelfde als normale tabular qlearning maar dan andere nam voor plotjes
if __name__ == "__main__":

    os.makedirs(IMG_DIR, exist_ok=True)

    # train agent
    policy = make_agent()

    # validate
    env = HydroElectric_Test(path_to_test_data="validate.xlsx")
    results = evaluate_policy(env, policy)

    total_profit = results["cum_rewards"][-1]
    print(f"Validation total profit ({alg_name}): {total_profit:.2f} EUR")

    # plots
    plot_cumulative_profit(
        results["cum_rewards"],
        IMG_DIR,
        "featql_cumulative_profit.png",
        "Q-learning: cumulative profit (validation)"
    )

    plot_dam_level(
        results["dam_levels"],
        IMG_DIR,
        "featql_dam_level.png",
        "Q-learning: dam level over time"
    )

    plot_action_vs_price(
        results["prices"],
        results["actions"],
        IMG_DIR,
        "featql_action_vs_price.png",
        "Q-learning: action vs price"
    )

    plot_mean_action_by_hour(
        results["actions"],
        IMG_DIR,
        "featql_mean_action_by_hour.png",
        "Q-learning: mean action by hour"
    )
    Q_plot = reduce_Q_for_plotting(policy.Q)

    plot_q_value_heatmap(
        Q_plot,
        fixed_hour=12,
        fixed_weekday=0,
        out_dir=IMG_DIR,
        filename="featql_q_value_heatmap.png",
        title="Q-learning: value heatmap (hour=12, weekday=Mon)"
    )

    plot_policy_heatmap(
        Q_plot,
        fixed_hour=12,
        fixed_weekday=0,
        out_dir=IMG_DIR,
        filename="featql_policy_heatmap.png",
        title="Q-learning: policy heatmap (hour=12, weekday=Mon)"
    )

def load_agent():
    """
    Entry point for graders.
    Trains and returns a Q-learning agent.
    """
    return make_agent()

