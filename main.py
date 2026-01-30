from TestEnv import HydroElectric_Test
import argparse
import matplotlib.pyplot as plt


# from qlearning_features import make_agent
from linearq_features import make_agent
# from qlearning import make_agent


parser = argparse.ArgumentParser()
parser.add_argument(
    '--excel_file',
    type=str,
    default='validate.xlsx'
)
args = parser.parse_args()


env = HydroElectric_Test(path_to_test_data=args.excel_file)


# Load agent 
agent = make_agent(train=False)

total_reward = []
cumulative_reward = []

observation = env.observation()


for i in range(730 * 24 - 1):


    action = agent.act(observation)

    next_observation, reward, terminated, truncated, info = env.step(action)

    total_reward.append(reward)
    cumulative_reward.append(sum(total_reward))

    done = terminated or truncated

    observation = next_observation

    if done:

        print('Total reward: ', sum(total_reward))

        plt.plot(cumulative_reward)
        plt.xlabel('Time (Hours)')
        plt.ylabel('Cumulative reward')
        plt.title('Validation performance')

        plt.show()
