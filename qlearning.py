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
    plot_state_visitation_heatmap
)

QT_DIR = "qtables"
os.makedirs(QT_DIR, exist_ok=True)


# =====================
# CONFIG
# =====================
alg_name = "qlearning"
img_root = "img"
IMG_DIR = os.path.join(img_root, alg_name)

MAX_VOLUME = 100_000  # m3

# Q-learning hyperparameters
N_EPISODES = 30
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.97

# Discrete actions (mapped to env actions)
ACTIONS = {
    0: -1.0,   # release
    1:  0.0,   # hold
    2:  1.0    # pump
}
N_ACTIONS = len(ACTIONS)

# =====================
# LOAD TRAINING DATA (for discretisation only)
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

# price bins from training distribution
PRICE_BINS = np.quantile(train_long["Price"],
                         [0.15, 0.3, 0.5, 0.7, 0.85])


# =====================
# DISCRETISATION (BASELINE)
# =====================
def discretize_observation(observation):
    """
    Convert continuous observation into a discrete state.
    """
    obs = parse_observation(observation)

    N_VOL_BINS = 8
    volume_bin = int(
        np.clip(obs["volume"] / MAX_VOLUME * N_VOL_BINS, 0, N_VOL_BINS-1)
    )

    price_bin = int(np.digitize(obs["price"], PRICE_BINS))
    hour_bin = obs["hour"] - 1          # env gives 1–24
    weekday_bin = obs["weekday"]

    return (volume_bin, price_bin, hour_bin, weekday_bin)

# =====================
# AGENT FACTORY
# =====================
def make_agent(train=False):
    """
    Create, train (if needed), and return a tabular Q-learning agent.
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

    MODEL_PATH = os.path.join(QT_DIR, "qtable_tabular.npy")


    # Auto-train if needed
    if train or not os.path.exists(MODEL_PATH):

        print("[Tabular Q] Training model...")
        agent.train()

        np.save(MODEL_PATH, dict(agent.Q))
        print(f"[Tabular Q] Saved model to {MODEL_PATH}")

    else:

        print(f"[Tabular Q] Loading model from {MODEL_PATH}")
        agent.Q.update(
            np.load(MODEL_PATH, allow_pickle=True).item()
        )

    # No exploration during evaluation
    agent.epsilon = 0.0

    return agent


# =====================
# MAIN (validation + plots)
# =====================
if __name__ == "__main__":

    os.makedirs(IMG_DIR, exist_ok=True)

    # train agent
    policy = make_agent(train=False)

    # validate
    env = HydroElectric_Test(path_to_test_data="validate.xlsx")
    results = evaluate_policy(env, policy)

    total_profit = results["cum_rewards"][-1]
    print(f"Validation total profit ({alg_name}): {total_profit:.2f} EUR")

    # plots
    plot_cumulative_profit(
        results["cum_rewards"],
        IMG_DIR,
        "ql_cumulative_profit.png",
        "Q-learning: cumulative profit (validation)"
    )

    plot_dam_level(
        results["dam_levels"],
        IMG_DIR,
        "ql_dam_level.png",
        "Q-learning: dam level over time"
    )

    plot_action_vs_price(
        results["prices"],
        results["actions"],
        IMG_DIR,
        "ql_action_vs_price.png",
        "Q-learning: action vs price"
    )

    plot_mean_action_by_hour(
        results["actions"],
        IMG_DIR,
        "ql_mean_action_by_hour.png",
        "Q-learning: mean action by hour"
    )

    plot_q_value_heatmap(
        policy.Q,
        PRICE_BINS,
        out_dir=IMG_DIR,
        filename="ql_q_value_heatmap.png",
        title="Q-learning: value heatmap (averaged over time)"
    )


    plot_policy_heatmap(
        policy.Q,
        PRICE_BINS,
        out_dir=IMG_DIR,
        filename="ql_policy_heatmap.png",
        title="Q-learning: policy heatmap (averaged over time)"
    )
    
    plot_state_visitation_heatmap(
        results["visited_states"],
        PRICE_BINS,
        out_dir=IMG_DIR,
        filename="ql_state_visits.png",
        title="Q-learning: state visitation"
    )



def load_agent():
    """
    Entry point for graders.
    Trains and returns a Q-learning agent.
    """
    return make_agent(train=False)
