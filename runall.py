# runall.py

import os
import numpy as np
import matplotlib.pyplot as plt

from TestEnv import HydroElectric_Test
from helpers.eval_utils import evaluate_policy

# Import your agents
from baseline import make_agent as make_baseline
from qlearning import make_agent as make_qlearning
from qlearning_features import make_agent as make_feat_qlearning
from linearq_features import make_agent as make_linear



# -----------------------
# CONFIG
# -----------------------

VALIDATION_FILE = "validate.xlsx"
OUT_DIR = "img/comparison"
SEED = 42

np.random.seed(SEED)


# -----------------------
# RUNNER
# -----------------------

def run_agent(name, make_agent_fn, train=False):
    """
    Train (optional) + evaluate one agent.
    """

    print(f"\nRunning {name}...")

    # init env
    env = HydroElectric_Test(path_to_test_data=VALIDATION_FILE)

    # build agent
    if train:
        agent = make_agent_fn(train=True)
    else:
        agent = make_agent_fn()

    # evaluate
    results = evaluate_policy(env, agent)

    cum_rewards = np.array(results["cum_rewards"])
    total_profit = cum_rewards[-1]

    print(f"{name} validation profit: {total_profit:.2f} EUR")

    return {
        "name": name,
        "cum_rewards": cum_rewards,
        "profit": total_profit,
    }


# -----------------------
# MAIN
# -----------------------

def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = []

    # -------------------
    # Baseline 
    # -------------------
    baseline_res = run_agent(
        "Baseline",
        make_baseline,
        train=False
    )
    all_results.append(baseline_res)

    # -------------------
    # Tabular Q-learning
    # -------------------
    ql_res = run_agent(
        "Tabular Q-learning",
        make_qlearning,
        train=False   # trains internally
    )
    all_results.append(ql_res)

    # -------------------
    # Feature Q-learning
    # -------------------
    feat_res = run_agent(
        "Feature Q-learning",
        make_feat_qlearning,
        train=False  # loads trained model
    )
    all_results.append(feat_res)

    # -------------------
    # Linear Q-learning
    # -------------------
    linear_res = run_agent(
        "Linear Q-learning",
        make_linear,
        train=False   # loads trained weights
    )
    all_results.append(linear_res)


    # -------------------
    # Plot comparison
    # -------------------

    plt.figure(figsize=(8, 4))

    for res in all_results:
        plt.plot(
            res["cum_rewards"],
            label=res["name"],
            linewidth=2
        )

    plt.xlabel("Time (hours)")
    plt.ylabel("Cumulative profit (EUR)")
    plt.title("Algorithm Comparison (Validation)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "comparison_cumulative_profit.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"\nSaved comparison plot to: {out_path}")

    # -------------------
    # Print table
    # -------------------

    print("\n=== FINAL COMPARISON ===\n")

    print(f"{'Algorithm':<20} | {'Profit (EUR)':>12}")
    print("-" * 36)

    for res in all_results:
        print(f"{res['name']:<20} | {res['profit']:>12.2f}")

    print()


if __name__ == "__main__":
    main()
