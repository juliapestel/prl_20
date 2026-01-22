# Reinforcement Learning for Hydroelectric Dam Operation

This repository contains the implementation for a reinforcement learning project on the optimal control of a pumped-storage hydroelectric dam under stochastic electricity prices.

The objective is to decide, at each hour, whether to pump water, generate electricity, or remain idle, using only information available at the current time step.

---

## Repository Structure

```
├── TestEnv.py                # Fixed validation environment (DO NOT MODIFY)
├── main.py                   # Main evaluation entry point
│
├── baseline_env.py            # Baseline algorithm
├── qlearning_env.py           # (Optional) Q-learning algorithm
│
├── helpers/
│   ├── __init__.py
│   ├── obs_utils.py           # Observation parsing and normalization
│   ├── eval_utils.py          # Shared evaluation loop
│   └── plot_functions.py      # Shared plotting utilities
│
├── train.xlsx                 # Training electricity price data
├── validate.xlsx              # Validation electricity price data
│
└── img/
    ├── baseline/              # Baseline figures
    └── qlearning/             # RL figures
```
---

## Environment

The environment is implemented in `TestEnv.py` and follows the Gymnasium API.

- `TestEnv.py` must NOT be modified.
- Observations have the form:
  [volume, price, hour_of_day, day_of_week, day_of_year, month, year]
- Actions are continuous values in the range [-1, 1]:
  - -1: generate electricity at maximum flow
  - +1: pump water at maximum flow
  -  0: no action

All physics, constraints, and reward calculations are handled by the environment.

---

## How to Run (Validation)

The project is evaluated using the standardized validation setup.

Run:

python3 main.py --excel_file validate.xlsx

This command:
- loads the validation data,
- runs the selected agent,
- prints the total cumulative reward,
- plots the cumulative reward over time.

The structure of `main.py` follows the example provided by the course staff.

---

## Algorithms

### Baseline Algorithm

The baseline is implemented in `baseline_env.py` and follows a rule-based control strategy. The policy operates according to the following principles:

- Expected electricity prices are estimated from the training data.
- A deadband strategy is applied to decide when to pump water or generate electricity.
- All physical and operational constraints imposed by the environment are respected.

The baseline can be run standalone for development and visualization purposes using:

python baseline_env.py

This execution will:

- Evaluate the baseline on the validation dataset.
- Save all generated figures to `img/baseline/`.

### Tabular Q-Learning

---

## Shared Utilities

Reusable code is placed in the `helpers/` directory:

- `obs_utils.py`: observation parsing and normalization
- `eval_utils.py`: shared evaluation loop
- `plot_functions.py`: shared plotting utilities

All algorithms use the same helpers to ensure fair comparison.

---

## Data

- `train.xlsx`: training electricity price data
- `validate.xlsx`: validation electricity price data

---

## Notes

- No future price information is used at decision time.
- Agents interact with the environment only via observations and actions.
- The code is compatible with the standardized validation environment.
