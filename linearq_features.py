# linearq_features.py
from TestEnv import HydroElectric_Test
from agents.agent_linearq import LinearQPolicy
from helpers.feature_extractor_cont import FeatureExtractorCont

MAX_VOLUME = 100_000
ACTIONS = {0: -1.0, 1: 0.0, 2: 1.0}

extractor = FeatureExtractorCont(max_volume=MAX_VOLUME)

def make_agent(train=False):
    agent = LinearQPolicy(
        feature_fn=extractor,
        actions=ACTIONS,
        alpha=2e-4,          # smaller alpha because we added more features
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.97,
        n_episodes=200,
        env_class=HydroElectric_Test,
        train_path="train.xlsx",
        reward_scale=1.0,
    )

    if train:
        agent.train()

    # greedy for validation
    agent.epsilon = 0.0
    return agent

def load_agent():
    # grader entry point
    return make_agent(train=True)
