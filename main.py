from TestEnv import HydroElectric_Test
import argparse
import matplotlib.pyplot as plt

from baseline import make_agent
# later you can switch to:
# from qlearning_env import make_agent

parser = argparse.ArgumentParser()
parser.add_argument('--excel_file', type=str, default='validate.xlsx')
args = parser.parse_args()

env = HydroElectric_Test(path_to_test_data=args.excel_file)
agent = make_agent()

total_reward = []
cumulative_reward = []

observation = env.observation()

for _ in range(730 * 24 - 1):
    action = agent.act(observation)
    observation, reward, terminated, truncated, _ = env.step(action)

    total_reward.append(reward)
    cumulative_reward.append(sum(total_reward))

    if terminated or truncated:
        break

print("Total reward:", sum(total_reward))

plt.plot(cumulative_reward)
plt.xlabel("Time (hours)")
plt.ylabel("Cumulative reward")
plt.show()
