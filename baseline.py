import pandas as pd
import numpy as np
import os
import matplotlib as plt

df = pd.read_excel("train.xlsx")

# start reservoir in m3 (half full)
startreservoir = 50_000

def potential_energy(volume_m3, g=9.81, h=30):
    m_kg = volume_m3 * 1000
    u = m_kg * g * h
    return u  # joule

# reservoir limits
max_volume = 100_000
max_energy = potential_energy(max_volume)
reservoir_energy = potential_energy(startreservoir)

# energy per hour limits (turbine + pump)
def calculate_per_hour(g=9.81, h=30):
    flow_m3_per_hour = 5 * 3600
    energy_per_m3 = 1000 * g * h
    hydro_energy_per_hour = flow_m3_per_hour * energy_per_m3

    production = hydro_energy_per_hour * 0.9   # turbine efficiency
    pumping = hydro_energy_per_hour / 0.8      # pump efficiency

    return production, pumping

max_prod_j, max_pump_j = calculate_per_hour()

def j_to_mwh(j):
    return j / 3.6e9

# baseline control parameters
lookback_days = 30
low_q = 0.3
high_q = 0.7

profit = 0.0
price_history = []

reservoir_trace = []
profit_trace = []
price_trace = []
time_trace = []
t = 0


for day in range(len(df)):
    for hour in range(1, 25):
        price = df.loc[day, f"Hour {hour:02d}"]
        price_history.append(price)

        # wait until enough history is available
        if len(price_history) < 24 * lookback_days:
            continue

        recent_prices = price_history[-24 * lookback_days:]
        low_thr = np.quantile(recent_prices, low_q)
        high_thr = np.quantile(recent_prices, high_q)

        # produce electricity
        if price > high_thr and reservoir_energy > 0:
            energy_used = min(max_prod_j, reservoir_energy)
            reservoir_energy -= energy_used
            profit += j_to_mwh(energy_used) * price

        # pump water
        elif price < low_thr and reservoir_energy < max_energy:
            energy_added = min(max_pump_j, max_energy - reservoir_energy)
            reservoir_energy += energy_added * 0.8
            profit -= j_to_mwh(energy_added) * price

        reservoir_trace.append(j_to_mwh(reservoir_energy))
        profit_trace.append(profit)
        price_trace.append(price)
        time_trace.append(t)
        t += 1


        # else: do nothing

print(f"final reservoir level: {j_to_mwh(reservoir_energy):.2f} mwh")
print(f"total profit: €{profit:.2f}")


base_dir = "results/results_baseline/img"
os.makedirs(base_dir, exist_ok=True)


plt.figure(figsize=(10, 4))
plt.plot(time_trace, reservoir_trace)
plt.xlabel("Time (hours)")
plt.ylabel("Reservoir energy (MWh)")
plt.title("Reservoir Energy Over Time – Rule-Based Baseline")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{base_dir}/reservoir_energy.png")
plt.close()


plt.figure(figsize=(10, 4))
plt.plot(time_trace, price_trace)
plt.xlabel("Time (hours)")
plt.ylabel("Electricity price (€/MWh)")
plt.title("Electricity Price Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{base_dir}/price.png")
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(time_trace, profit_trace)
plt.xlabel("Time (hours)")
plt.ylabel("Cumulative profit (€)")
plt.title("Cumulative Profit – Rule-Based Baseline")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{base_dir}/profit.png")
plt.close()

