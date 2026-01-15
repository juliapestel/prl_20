import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Configuration
# ===============================
IMG_DIR = "img"
os.makedirs(IMG_DIR, exist_ok=True)

START_RESERVOIR = 50_000  # m3
MAX_VOLUME = 100_000     # m3
G = 9.81
H = 30

HOUR_COLS = [f"Hour {h:02d}" for h in range(1, 25)]

# ===============================
# Load data
# ===============================
train = pd.read_excel("train.xlsx")
val   = pd.read_excel("validate.xlsx")

# Rename and parse date column
train = train.rename(columns={"PRICES": "Date"})
val   = val.rename(columns={"PRICES": "Date"})

train["Date"] = pd.to_datetime(train["Date"])
val["Date"]   = pd.to_datetime(val["Date"])

# ===============================
# Physics helpers
# ===============================
def potential_energy(volume_m3, g=G, h=H):
    return volume_m3 * 1000 * g * h  # Joule

def calculate_per_hour(g=G, h=H):
    flow_m3_per_hour = 5 * 3600
    energy_per_m3 = 1000 * g * h
    hydro_energy_per_hour = flow_m3_per_hour * energy_per_m3

    production = hydro_energy_per_hour * 0.9
    pumping = hydro_energy_per_hour / 0.8
    return production, pumping

def j_to_mwh(j):
    return j / 3.6e9

MAX_ENERGY = potential_energy(MAX_VOLUME)
MAX_PROD_J, MAX_PUMP_J = calculate_per_hour()

# ===============================
# Reshape to long format (TRAIN ONLY)
# ===============================
train_long = train.melt(
    id_vars=["Date"],
    value_vars=HOUR_COLS,
    var_name="Hour",
    value_name="Price"
)

train_long["Hour"] = train_long["Hour"].str.extract(r"(\d+)").astype(int) - 1
train_long["Weekday"] = train_long["Date"].dt.weekday
train_long["DayOfYear"] = train_long["Date"].dt.dayofyear

# ===============================
# Hourly mean by weekday
# ===============================
weekday_hour_mean = (
    train_long
    .groupby(["Weekday", "Hour"])["Price"]
    .mean()
    .unstack()
)

plt.figure(figsize=(12, 6))
weekday_hour_mean.T.plot(ax=plt.gca())
plt.xlabel("Hour of day")
plt.ylabel("Mean price (EUR/MWh)")
plt.title("Hourly mean price by weekday (train)")
plt.legend(
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    title="Weekday"
)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "hourly_mean_by_weekday.png"))
plt.close()

# ===============================
# Mean price by day of year
# ===============================
dayofyear_mean = (
    train_long
    .groupby("DayOfYear")["Price"]
    .mean()
)

plt.figure(figsize=(12, 4))
plt.plot(dayofyear_mean.index, dayofyear_mean.values)
plt.xlabel("Day of year")
plt.ylabel("Mean price (EUR/MWh)")
plt.title("Mean price by day of year (train)")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "mean_price_day_of_year.png"))
plt.close()

# ===============================
# Heatmap weekday x hour
# ===============================
plt.figure(figsize=(10, 5))
sns.heatmap(
    weekday_hour_mean,
    cmap="viridis",
    xticklabels=range(24),
    yticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
)
plt.xlabel("Hour of day")
plt.ylabel("Weekday")
plt.title("Mean price heatmap (weekday x hour, train)")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "heatmap_weekday_hour.png"))
plt.close()

# ===============================
# Expected price tables
# ===============================
expected_price_wh = weekday_hour_mean
expected_price_doy = dayofyear_mean

# ===============================
# Modified baseline policy
# ===============================
def run_baseline(df):
    reservoir_energy = potential_energy(START_RESERVOIR)
    profit = 0.0

    for _, row in df.iterrows():
        date = row["Date"]
        weekday = date.weekday()
        dayofyear = date.dayofyear

        for hour in range(24):
            price = row[f"Hour {hour+1:02d}"]

            exp_wh = expected_price_wh.loc[weekday, hour]
            exp_doy = expected_price_doy.loc[dayofyear]

            expected_price = 0.5 * exp_wh + 0.5 * exp_doy

            if price > expected_price and reservoir_energy > 0:
                energy_used = min(MAX_PROD_J, reservoir_energy)
                reservoir_energy -= energy_used
                profit += j_to_mwh(energy_used) * price

            elif price < expected_price and reservoir_energy < MAX_ENERGY:
                energy_added = min(MAX_PUMP_J, MAX_ENERGY - reservoir_energy)
                reservoir_energy += energy_added * 0.8
                profit -= j_to_mwh(energy_added) * price

    return profit

# ===============================
# Run evaluation
# ===============================
train_profit = run_baseline(train)
val_profit   = run_baseline(val)

print(f"Train profit: {train_profit:.2f} EUR")
print(f"Validation profit: {val_profit:.2f} EUR")


# ============================================================
# VALIDATION EVALUATION + PLOTS (VALIDATION SET ONLY)
# ============================================================

def run_baseline_with_logging(df):
    reservoir_energy = potential_energy(START_RESERVOIR)
    profit = 0.0

    profit_trace = []
    reservoir_trace = []
    price_trace = []
    action_trace = []   # -1 = pump, 0 = idle, +1 = produce
    time_trace = []

    t = 0

    for _, row in df.iterrows():
        date = row["Date"]
        weekday = date.weekday()
        dayofyear = date.dayofyear

        for hour in range(24):
            price = row[f"Hour {hour+1:02d}"]

            exp_wh = expected_price_wh.loc[weekday, hour]
            exp_doy = expected_price_doy.loc[dayofyear]
            expected_price = 0.5 * exp_wh + 0.5 * exp_doy

            action = 0

            if price > expected_price and reservoir_energy > 0:
                energy_used = min(MAX_PROD_J, reservoir_energy)
                reservoir_energy -= energy_used
                profit += j_to_mwh(energy_used) * price
                action = 1

            elif price < expected_price and reservoir_energy < MAX_ENERGY:
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
# RUN ON VALIDATION SET
# ============================================================
val_results = run_baseline_with_logging(val)

print(f"Validation profit: {val_results['profit']:.2f} EUR")


# ============================================================
# PLOT 1: CUMULATIVE PROFIT
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(val_results["time"], val_results["profit_trace"])
plt.xlabel("Time (hours)")
plt.ylabel("Cumulative profit (EUR)")
plt.title("Validation: cumulative profit over time")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "val_cumulative_profit.png"))
plt.close()


# ============================================================
# PLOT 2: RESERVOIR ENERGY
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(val_results["time"], val_results["reservoir_trace"])
plt.xlabel("Time (hours)")
plt.ylabel("Stored energy (MWh)")
plt.title("Validation: reservoir energy over time")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "val_reservoir_level.png"))
plt.close()


# ============================================================
# PLOT 3: ACTION VS PRICE
# ============================================================
plt.figure(figsize=(10, 4))
plt.scatter(
    val_results["price_trace"],
    val_results["action_trace"],
    alpha=0.3
)
plt.yticks([-1, 0, 1], ["Pump", "Idle", "Produce"])
plt.xlabel("Electricity price (EUR/MWh)")
plt.ylabel("Action")
plt.title("Validation: action as function of price")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "val_action_vs_price.png"))
plt.close()


# ============================================================
# PLOT 4: MEAN ACTION BY HOUR
# ============================================================
actions = np.array(val_results["action_trace"])
hours = np.array(val_results["time"]) % 24

mean_action_by_hour = [
    actions[hours == h].mean() for h in range(24)
]

plt.figure(figsize=(10, 4))
plt.bar(range(24), mean_action_by_hour)
plt.xlabel("Hour of day")
plt.ylabel("Mean action")
plt.title("Validation: average action by hour")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "val_mean_action_by_hour.png"))
plt.close()
