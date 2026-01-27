# agents/agent_linearq.py
import numpy as np


class LinearQPolicy:
    """
    Linear function approximation Q-learning:
      Q(s,a) = W[a] dot phi(s)

    Notes:
      - feature_fn can be stateful (rolling windows). During training we must compute
        features exactly once per observation and carry phi_next -> phi to avoid
        corrupting the feature history.
      - epsilon-greedy exploration with per-episode decay.
      - guards against NaN/Inf Q-values.
      - bounded episode length to avoid TestEnv out-of-bounds.
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
        n_episodes=50,
        env_class=None,
        train_path="train.xlsx",
        reward_scale=1.0,
        l2=1e-4
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
        self.l2 = float(l2)
        self.reward_scale = float(reward_scale)

        self.W = None  # (n_actions, n_features)

    def _ensure_W(self, n_features: int) -> None:
        if self.W is None:
            self.W = np.zeros((self.n_actions, n_features), dtype=np.float32)

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        self._ensure_W(int(phi.shape[0]))
        return self.W @ phi

    @staticmethod
    def _finite(x: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(x)))

    def _safe_greedy_action_index(self, q: np.ndarray) -> int:
        # Safe default = index 1 (assumes actions correspond to [-1, 0, 1])
        if not self._finite(q):
            return 1

        max_q = np.max(q)
        if not np.isfinite(max_q):
            return 1

        best = np.flatnonzero(q == max_q)
        if best.size == 0:
            return 1
        return int(np.random.choice(best))

    def act(self, observation):
        """
        Action selection for evaluation / inference.
        Note: If feature_fn is stateful, calling act() sequentially through an episode is fine.
        Avoid calling act() multiple times on the same observation without resetting feature_fn.
        """
        phi = self.feature_fn(observation)
        q = self.q_values(phi)

        if np.random.rand() < self.epsilon:
            a_idx = np.random.randint(self.n_actions)
        else:
            a_idx = self._safe_greedy_action_index(q)

        return self.action_list[a_idx]

    def train(self):
        for _ep in range(1, self.n_episodes + 1):
            env = self.env_class(path_to_test_data=self.train_path)

            # avoid TestEnv last-step index error
            n_days = env.price_values.shape[0]
            max_steps = n_days * 24 - 1

            # reset feature state ONCE per episode (if supported)
            if hasattr(self.feature_fn, "reset"):
                self.feature_fn.reset()

            # first observation + features (compute ONCE)
            obs = env.observation()
            phi = self.feature_fn(obs)

            # accumulate episode reward for logging
            ep_reward = 0.0

            for _t in range(max_steps):
                q = self.q_values(phi)

                # epsilon-greedy selection
                if np.random.rand() < self.epsilon:
                    a_idx = np.random.randint(self.n_actions)
                else:
                    a_idx = self._safe_greedy_action_index(q)

                action = self.action_list[a_idx]

                next_obs, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)

                # next features (compute ONCE)
                phi_next = self.feature_fn(next_obs)

                # TD target
                r = float(reward) / self.reward_scale
                ep_reward += r

                if done:
                    target = r
                else:
                    q_next = self.q_values(phi_next)
                    target = r + self.gamma * float(np.max(q_next))

                # update
                td_error = target - float(q[a_idx])
                self._ensure_W(int(phi.shape[0]))
                # L2 regularisatie / weight decay
                self.W[a_idx] += self.alpha * ((td_error * phi) - self.l2 * self.W[a_idx])

 
                # carry forward WITHOUT recomputing for the same obs
                obs = next_obs
                phi = phi_next

                if done:
                    break

            # print episode summary
            print(f"Episode {_ep:4d} | reward: {ep_reward: .4f} | epsilon: {self.epsilon: .3f}")

            # epsilon decay per episode
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return self
