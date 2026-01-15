import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================
IMG_DIR = "img"
os.makedirs(IMG_DIR, exist_ok=True)

START_RESERVOIR = 50_000  # m3
MAX_VOLUME = 100_000     # m3
G = 9.81
H = 30

UPPER_MARGIN = 1.08   # deadband
LOWER_MARGIN = 0.92

HOUR_COLS = [f"Hour {h:02d}" for h in range(1, 25)]

# ============================================================
# LOAD DATA
# ============================================================
train = pd.read_excel("train.xlsx").rename(columns={"PRICES": "Date"})
val   = pd.read_excel("validate.xlsx").rename(columns={"PRICES": "Date"})

train["Date"] = pd.to_datetime(train["Date"])
val["Date"]   = pd.to_datetime(val["Date"])

# ============================================================
# PHYSICS HELPERS
# ============================================================
def potential_energy(volume_m3, g=G, h=H):
    return volume_m3 * 1000 * g * h

def calculate_per_hour(g=G, h=H):
    flow_m3_per_hour = 5 * 3600
    energy_per_m3 = 1000 * g * h
    hydro_energy_per_hour = flow_m3_per_hour * energy_per_m3
    return hydro_energy_per_hour * 0.9, hydro_energy_per_hour / 0.8

def j_to_mwh(j):
    return j / 3.6e9

MAX_ENERGY = potential_energy(MAX_VOLUME)
MAX_PROD_J, MAX_PUMP_J = calculate_per_hour()

# ============================================================
# TRAINING STATISTICS (TRAIN ONLY)
# ============================================================
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

dayofyear_mean = train_long.groupby("DayOfYear")["Price"].mean()

# ============================================================
# BASELINE WITH FIXES (DEADBAND + STORAGE AWARE)
# ============================================================
def run_baseline_with_logging(df):
    reservoir_energy = potential_energy(START_RESERVOIR)
    profit = 0.0

    profit_trace = []
    reservoir_trace = []
    price_trace = []
    action_trace = []
    time_trace = []

    t = 0

    for _, row in df.iterrows():
        weekday = row["Date"].weekday()
        dayofyear = row["Date"].dayofyear

        for hour in range(24):
            price = row[f"Hour {hour+1:02d}"]

            exp_wh = weekday_hour_mean.loc[weekday, hour]
            exp_doy = dayofyear_mean.loc[dayofyear]
            expected_price = 0.5 * exp_wh + 0.5 * exp_doy

            storage_frac = reservoir_energy / MAX_ENERGY
            pump_allowed = storage_frac < 0.9
            produce_allowed = storage_frac > 0.1

            action = 0

            if (
                price > expected_price * UPPER_MARGIN
                and produce_allowed
                and reservoir_energy > 0
            ):
                energy_used = min(MAX_PROD_J, reservoir_energy)
                reservoir_energy -= energy_used
                profit += j_to_mwh(energy_used) * price
                action = 1

            elif (
                price < expected_price * LOWER_MARGIN
                and pump_allowed
                and reservoir_energy < MAX_ENERGY
            ):
                energy_added = min(MAX_PUMP_J, MAX_ENERGY - reservoir_energy)
                reservoir_energy += energy_added * 0.8
                profit -= j_to_mwh(energy_added) * price
                action = -1

            profit_trace.append(profit)
            reservoir_trace.append(j_to_mwh(reservoir_energy))
            price_trace.append(price)
            action_trace.append(action)
            time_trace.append(t)
            t += 1

    return {
        "profit": profit,
        "profit_trace": profit_trace,
        "reservoir_trace": reservoir_trace,
        "price_trace": price_trace,
        "action_trace": action_trace,
        "time": time_trace,
    }

# ============================================================
# RUN ON VALIDATION ONLY
# ============================================================
val_results = run_baseline_with_logging(val)
print(f"Validation profit: {val_results['profit']:.2f} EUR")

# ============================================================
# PLOTS (VALIDATION)
# ============================================================

# 1) Cumulative profit
plt.figure(figsize=(10, 4))
plt.plot(val_results["time"], val_results["profit_trace"])
plt.xlabel("Time (hours)")
plt.ylabel("Cumulative profit (EUR)")
plt.title("Validation: cumulative profit over time")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "val_cumulative_profit.png"))
plt.close()

# 2) Reservoir energy
plt.figure(figsize=(10, 4))
plt.plot(val_results["time"], val_results["reservoir_trace"])
plt.xlabel("Time (hours)")
plt.ylabel("Stored energy (MWh)")
plt.title("Validation: reservoir energy over time")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "val_reservoir_level.png"))
plt.close()

# 3) Action vs price (clipped for readability)
prices = np.array(val_results["price_trace"])
actions = np.array(val_results["action_trace"])
price_clip = np.percentile(prices, 99)
prices = np.clip(prices, None, price_clip)

plt.figure(figsize=(10, 4))
plt.scatter(prices, actions, alpha=0.3)
plt.yticks([-1, 0, 1], ["Pump", "Idle", "Produce"])
plt.xlabel("Electricity price (EUR/MWh)")
plt.ylabel("Action")
plt.title("Validation: action vs price (clipped)")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "val_action_vs_price.png"))
plt.close()

# 4) Mean action by hour
hours = np.array(val_results["time"]) % 24
mean_action_by_hour = [actions[hours == h].mean() for h in range(24)]

plt.figure(figsize=(10, 4))
plt.bar(range(24), mean_action_by_hour)
plt.xlabel("Hour of day")
plt.ylabel("Mean action")
plt.title("Validation: average action by hour")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "val_mean_action_by_hour.png"))
plt.close()
