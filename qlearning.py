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


ALGO_NAME = "qlearning"
IMG_ROOT = "img"
IMG_DIR = os.path.join(IMG_ROOT, ALGO_NAME)

UPPER_MARGIN = 1.08
LOWER_MARGIN = 0.92

MAX_VOLUME = 100_000  # m3

# load training data
train = pd.read_excel("train.xlsx").rename(columns={"PRICES": "Date"})
train["Date"] = pd.to_datetime(train["Date"])

HOUR_COLS = [f"Hour {h:02d}" for h in range(1, 25)]

# =====================
# PRECOMPUTE EXPECTED PRICES (OFFLINE)
# =====================
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