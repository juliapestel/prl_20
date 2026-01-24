import os
import numpy as np
import pandas as pd

from TestEnv import HydroElectric_Test
from helpers.obs_utils import parse_observation
from helpers.eval_utils import evaluate_policy
from helpers.plot_functions import (
    plot_cumulative_profit,
    plot_dam_level,
    plot_action_vs_price,
    plot_mean_action_by_hour
)


alg_name = "baseline"
img_root = "img"
IMG_DIR = os.path.join(img_root, alg_name)

UPPER_MARGIN = 1.08
LOWER_MARGIN = 0.92

MAX_VOLUME = 100000  # m3

# load training data
train = pd.read_excel("train.xlsx").rename(columns={"PRICES": "Date"})
train["Date"] = pd.to_datetime(train["Date"])

HOUR_COLS = [f"Hour {h:02d}" for h in range(1, 25)]

# precompute expected prices
train_long = train.melt(
    id_vars=["Date"],
    value_vars=HOUR_COLS,
    var_name="Hour",
    value_name="Price"
)

train_long["Hour"] = train_long["Hour"].str.extract(r"(\d+)").astype(int) - 1
train_long["Weekday"] = train_long["Date"].dt.weekday
train_long["DayOfYear"] = train_long["Date"].dt.dayofyear

weekday_hour_mean = (
    train_long
    .groupby(["Weekday", "Hour"])["Price"]
    .mean()
    .unstack()
)

dayofyear_mean = (
    train_long
    .groupby("DayOfYear")["Price"]
    .mean()
)


class BaselinePolicy:
    """
    Rule-based baseline using expected price + deadband.
    Compatible with TestEnv.
    """

    def __init__(self, weekday_hour_mean, dayofyear_mean):
        self.weekday_hour_mean = weekday_hour_mean
        self.dayofyear_mean = dayofyear_mean

    def act(self, observation):
        """
        observation =
        [volume, price, hour, weekday, dayofyear, month, year]
        """

        obs = parse_observation(observation)

        price = obs["price"]
        hour = obs["hour"]
        weekday = obs["weekday"]
        dayofyear = obs["dayofyear"]
        storage_frac = obs["volume"] / MAX_VOLUME

        exp_wh = self.weekday_hour_mean.loc[weekday, hour]
        exp_doy = self.dayofyear_mean.loc[dayofyear]
        expected_price = 0.5 * exp_wh + 0.5 * exp_doy

        if price > expected_price * UPPER_MARGIN and storage_frac > 0.1:
            return -1.0
        elif price < expected_price * LOWER_MARGIN and storage_frac < 0.9:
            return +1.0
        else:
            return 0.0


def make_agent():
    """
    Factory function to create the baseline agent.
    Used by main.py.
    """
    return BaselinePolicy(weekday_hour_mean, dayofyear_mean)



if __name__ == "__main__":

    os.makedirs(IMG_DIR, exist_ok=True)

    # initialise environment
    env = HydroElectric_Test(path_to_test_data="validate.xlsx")
    policy = BaselinePolicy(weekday_hour_mean, dayofyear_mean)

    results = evaluate_policy(env, policy)

    total_profit = results["cum_rewards"][-1]
    print(f"Validation total profit ({alg_name}): {total_profit:.2f} EUR")

    # plotten
    plot_cumulative_profit(
        results["cum_rewards"],
        IMG_DIR,
        "cumulative_profit.png",
        "Baseline: cumulative profit (validation)"
    )

    plot_dam_level(
        results["dam_levels"],
        IMG_DIR,
        "dam_level.png",
        "Baseline: dam level over time"
    )

    plot_action_vs_price(
        results["prices"],
        results["actions"],
        IMG_DIR,
        "action_vs_price.png",
        "Baseline: action vs price"
    )

    plot_mean_action_by_hour(
        results["actions"],
        IMG_DIR,
        "mean_action_by_hour.png",
        "Baseline: mean action by hour"
    )
