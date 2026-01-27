import numpy as np

def evaluate_policy(env, policy):

    obs = env.observation()

    # Reset stateful components if present
    if hasattr(policy, "discretize") and hasattr(policy.discretize, "reset"):
        policy.discretize.reset()

    if hasattr(policy, "feature_fn") and hasattr(policy.feature_fn, "reset"):
        policy.feature_fn.reset()

    rewards = []
    dam_levels = []
    prices = []
    actions = []
    visited_states = []

    done = False

    while not done:

        action = policy.act(obs)

        obs, reward, terminated, truncated, _ = env.step(action)

        rewards.append(reward)
        dam_levels.append(obs[0])
        prices.append(obs[1])
        actions.append(action)
        visited_states.append(obs)

        done = terminated or truncated

    return {
        "rewards": np.array(rewards),
        "cum_rewards": np.cumsum(rewards),
        "dam_levels": np.array(dam_levels),
        "prices": np.array(prices),
        "actions": np.array(actions),
        "visited_states": visited_states,
    }
