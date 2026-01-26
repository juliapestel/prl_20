import numpy as np
from collections import defaultdict

class QLearningPolicy:
    """
    Generic tabular Q-learning agent.
    State representation is provided via a discretisation function.
    """

    def __init__(
        self,
        discretize_fn,
        actions,
        n_actions,
        alpha,
        gamma,
        epsilon_start,
        epsilon_end,
        epsilon_decay,
        n_episodes,
        env_class,
        train_path
    ):
        self.discretize = discretize_fn
        self.actions = actions
        self.n_actions = n_actions
        self.Q = defaultdict(lambda: np.zeros(n_actions))

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.n_episodes = n_episodes

        self.env_class = env_class
        self.train_path = train_path

    def train(self):
        epsilon = self.epsilon_start

        for episode in range(self.n_episodes):

            env = self.env_class(self.train_path)
            obs = env.observation()
            if hasattr(self.discretize, "reset"):
                self.discretize.reset()
            done = False
            total_reward = 0.0

            while not done:
                state = self.discretize(obs)

                if np.random.rand() < epsilon:
                    action_idx = np.random.randint(self.n_actions)
                else:
                    q = self.Q[state]

                    # tie-break: if all equal (common when unseen), don't default to argmax->0
                    if np.allclose(q, q[0]):
                        action_idx = 1  # safe default = "do nothing" (assumes actions [-1,0,1])
                    else:
                        action_idx = int(np.argmax(q))

                action = self.actions[action_idx]

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated


                next_state = self.discretize(next_obs)

                self.Q[state][action_idx] += self.alpha * (
                    reward
                    + self.gamma * np.max(self.Q[next_state])
                    - self.Q[state][action_idx]
                )

                obs = next_obs
                total_reward += reward

            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)

            print(
                f"Episode {episode + 1}/{self.n_episodes} | "
                f"reward={total_reward:.2f} | epsilon={epsilon:.3f}"
            )

    def act(self, observation):
        state = self.discretize(observation)
        q = self.Q[state]

        # If all Q-values are equal (common for unseen states), avoid always picking index 0
        if np.allclose(q, q[0]):
            action_idx = 1  # safe default = "do nothing" (assumes actions = [-1, 0, 1])
        else:
            action_idx = int(np.argmax(q))

        return self.actions[action_idx]

