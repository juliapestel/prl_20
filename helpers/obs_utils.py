"""
Utilities for safely handling observations from TestEnv.
Ensures compatibility across baselines and RL agents.
"""

def parse_observation(observation):
    """
    Input:
        observation = [volume, price, hour, weekday, dayofyear, month, year]

    Output:
        dict with safe, normalized fields
    """

    return {
        "volume": observation[0],
        "price": observation[1],
        "hour": (int(observation[2]) - 1) % 24,
        "weekday": int(observation[3]) % 7,
        "dayofyear": min(max(int(observation[4]), 1), 365),
        "month": min(max(int(observation[5]), 1), 12),
        # year intentionally ignored
    }
