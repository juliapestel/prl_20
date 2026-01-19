import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from TestEnv import HydroElectric_Test
import gymnasium as gym
import os


# =====================================================
# 1. Load training data and compute price thresholds
# =====================================================

train = pd.read_excel("train.xlsx")

# Convert to long format
train_long = train.melt(
    id_vars=["PRICES"],
    var_name="Hour",
    value_name="price"
)

train_long["hour"] = train_long["Hour"].str.extract(r"(\d+)").astype(int)
train_long["weekday"] = train_long["PRICES"].dt.weekday  # Monday = 0
train_long = train_long.dropna()

# Compute percentiles per (weekday, hour)
thresholds = (
    train_long
    .groupby(["weekday", "hour"])["price"]
    .quantile([0.10, 0.90])
    .unstack()
    .reset_index()
)

thresholds.columns = ["weekday", "hour", "p10", "p90"]

# Convert to dictionary for fast lookup
threshold_dict = {}
for _, row in thresholds.iterrows():
    threshold_dict[(row.weekday, row.hour)] = (row.p10, row.p90)

# =====================================================
# 2. Run baseline on validation environment
# =====================================================

env = HydroElectric_Test(path_to_test_data="validate.xlsx")

observation = env.observation()
total_reward = 0
cumulative_reward = []
reservoir_levels = []
prices = []

for _ in range(730 * 24 - 1):

    dam_level, price, hour, weekday, *_ = observation
    hour = int(hour)
    weekday = int(weekday)

    p10, p90 = threshold_dict.get(
        (weekday, hour),
        (None, None)
    )

    # Default: do nothing
    action = np.array([0.0])

    if p10 is not None:
        if price < p10:
            action = np.array([1.0])     # pump
        elif price > p90:
            action = np.array([-1.0])    # generate

    next_obs, reward, terminated, truncated, _ = env.step(action)

    total_reward += reward
    cumulative_reward.append(total_reward)
    reservoir_levels.append(dam_level)
    prices.append(price)

    observation = next_obs

    if terminated or truncated:
        break

# =====================================================
# 3. Plot results
# =====================================================

# =====================================================
# 3. Plot results (save to img/)
# =====================================================

# Cumulative reward plot
plt.figure()
plt.plot(cumulative_reward)
plt.xlabel("Time (hours)")
plt.ylabel("Cumulative reward (€)")
plt.title("Heuristic percentile baseline")
plt.tight_layout()
plt.savefig("img/cumulative_reward.png", dpi=300)
plt.close()

# Reservoir level plot
plt.figure()
plt.plot(reservoir_levels)
plt.xlabel("Time (hours)")
plt.ylabel("Reservoir level (m³)")
plt.title("Reservoir level over time")
plt.tight_layout()
plt.savefig("img/reservoir_level.png", dpi=300)
plt.close()

print("Total reward:", total_reward)

