# agents/agent_linearq.py
import numpy as np

class LinearQPolicy:
    """
    Linear function approximation Q-learning:
      Q(s,a) = W[a] dot phi(s)

    Stability:
      - bounded features (handled by extractor)
      - small alpha
      - safe greedy selection with NaN/Inf guard
      - bounded episode length to avoid TestEnv out-of-bounds
    """
    def __init__(
        self,
        feature_fn,
        actions,
        alpha=2e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.97,
        n_episodes=200,
        env_class=None,
        train_path="train.xlsx",
        reward_scale=1.0,
    ):
        self.feature_fn = feature_fn

        self.actions = actions
        self.action_list = [actions[i] for i in sorted(actions.keys())]
        self.n_actions = len(self.action_list)

        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        self.n_episodes = int(n_episodes)
        self.env_class = env_class
        self.train_path = train_path

        self.reward_scale = float(reward_scale)

        self.W = None  # (n_actions, n_features)

    def _ensure_W(self, n_features):
        if self.W is None:
            self.W = np.zeros((self.n_actions, n_features), dtype=np.float32)

    def q_values(self, phi):
        self._ensure_W(phi.shape[0])
        return self.W @ phi

    @staticmethod
    def _finite(x):
        return np.all(np.isfinite(x))

    def _safe_greedy_action_index(self, q):
        if not self._finite(q):
            return 1  # safe default = 0.0 action (assumes [-1,0,1])

        max_q = np.max(q)
        if not np.isfinite(max_q):
            return 1

        best = np.flatnonzero(q == max_q)
        if best.size == 0:
            return 1
        return int(np.random.choice(best))

    def act(self, observation):
        phi = self.feature_fn(observation)
        q = self.q_values(phi)

        if np.random.rand() < self.epsilon:
            a_idx = np.random.randint(self.n_actions)
        else:
            a_idx = self._safe_greedy_action_index(q)

        return self.action_list[a_idx]

    def train(self):
        for ep in range(1, self.n_episodes + 1):
            env = self.env_class(path_to_test_data=self.train_path)

            # avoid TestEnv last-step index error
            n_days = env.price_values.shape[0]
            max_steps = n_days * 24 - 1

            if hasattr(self.feature_fn, "reset"):
                self.feature_fn.reset()

            obs = env.observation()
            ep_reward = 0.0

            for _ in range(max_steps):
                phi = self.feature_fn(obs)
                q = self.q_values(phi)

                # epsilon-greedy
                if np.random.rand() < self.epsilon:
                    a_idx = np.random.randint(self.n_actions)
                else:
                    a_idx = self._safe_greedy_action_index(q)

                action = self.action_list[a_idx]
                next_obs, reward, terminated, truncated, _ = env.step(action)

                ep_reward += float(reward)
                r = float(reward) / self.reward_scale
                done = terminated or truncated

                # target
                if done:
                    target = r
                else:
                    phi2 = self.feature_fn(next_obs)
                    q2 = self.q_values(phi2)
                    if not self._finite(q2):
                        target = r
                    else:
                        target = r + self.gamma * float(np.max(q2))

                # update
                q_sa = float(q[a_idx]) if np.isfinite(q[a_idx]) else 0.0
                td_error = target - q_sa

                if np.isfinite(td_error) and self._finite(phi):
                    self.W[a_idx] += self.alpha * td_error * phi

                obs = next_obs
                if done:
                    break

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            print(f"Episode {ep}/{self.n_episodes} | reward={ep_reward:.2f} | epsilon={self.epsilon:.3f}")
