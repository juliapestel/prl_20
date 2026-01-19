import numpy as np

def evaluate_policy(env, policy):
    obs = env.observation()

    rewards = []
    dam_levels = []
    prices = []
    actions = []

    done = False

    while not done:
        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)

        rewards.append(reward)
        dam_levels.append(obs[0])
        prices.append(obs[1])
        actions.append(action)

        done = terminated or truncated

    return {
        "rewards": np.array(rewards),
        "cum_rewards": np.cumsum(rewards),
        "dam_levels": np.array(dam_levels),
        "prices": np.array(prices),
        "actions": np.array(actions),
    }
