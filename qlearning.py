import os
import numpy as np
import pandas as pd
from collections import defaultdict

from TestEnv import HydroElectric_Test
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

# =====================
# CONFIG
# =====================
alg_name = "qlearning"
img_root = "img"
IMG_DIR = os.path.join(img_root, alg_name)

MAX_VOLUME = 100_000  # m3

# Q-learning hyperparameters
N_EPISODES = 20
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.95

# Discrete actions (mapped to env actions)
ACTIONS = {
    0: -1.0,   # release
    1:  0.0,   # hold
    2:  1.0    # pump
}
N_ACTIONS = len(ACTIONS)


# LOAD TRAINING DATA (for discretization only)
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

# DISCRETIZATION
def discretize_observation(observation):
    """
    Convert continuous observation into a discrete state.
    """
    obs = parse_observation(observation)

    volume_bin = int(np.clip(obs["volume"] / MAX_VOLUME * 5, 0, 4))
    price_bin = int(np.digitize(obs["price"], PRICE_BINS))
    hour_bin = obs["hour"] - 1          # env gives 1–24
    weekday_bin = obs["weekday"]

    return (volume_bin, price_bin, hour_bin, weekday_bin)


# Q-LEARNING AGENT
class QLearningPolicy:
    """
    Tabular Q-learning agent.
    Compatible with TestEnv and evaluate_policy.
    """

    def __init__(self):
        self.Q = defaultdict(lambda: np.zeros(N_ACTIONS))

    def train(self, path_to_train_data):
        epsilon = EPSILON_START

        for episode in range(N_EPISODES):

            env = HydroElectric_Test(path_to_train_data)
            obs = env.observation()
            done = False
            total_reward = 0.0

            while not done:
                state = discretize_observation(obs)

                # ε-greedy action selection
                if np.random.rand() < epsilon:
                    action_idx = np.random.randint(N_ACTIONS)
                else:
                    action_idx = np.argmax(self.Q[state])

                action = ACTIONS[action_idx]

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                next_state = discretize_observation(next_obs)

                # Q-learning update
                self.Q[state][action_idx] += ALPHA * (
                    reward
                    + GAMMA * np.max(self.Q[next_state])
                    - self.Q[state][action_idx]
                )

                obs = next_obs
                total_reward += reward

            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

            print(
                f"Episode {episode + 1}/{N_EPISODES} | "
                f"reward={total_reward:.2f} | epsilon={epsilon:.3f}"
            )


    def act(self, observation):
        """
        Deterministic policy used during evaluation.
        """
        state = discretize_observation(observation)
        action_idx = np.argmax(self.Q[state])
        return ACTIONS[action_idx]


def make_agent():
    """
    Factory function for consistency with baseline.
    """
    agent = QLearningPolicy()
    agent.train("train.xlsx")
    return agent


# MAIN (validation + plots)
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
        fixed_hour=12,
        fixed_weekday=0,
        out_dir=IMG_DIR,
        filename="ql_q_value_heatmap.png",
        title="Q-learning: value heatmap (hour=12, weekday=Mon)"
    )

    plot_policy_heatmap(
        policy.Q,
        fixed_hour=12,
        fixed_weekday=0,
        out_dir=IMG_DIR,
        filename="ql_policy_heatmap.png",
        title="Q-learning: policy heatmap (hour=12, weekday=Mon)"
    )
