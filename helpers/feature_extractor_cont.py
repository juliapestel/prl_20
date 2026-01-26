# helpers/feature_extractor_cont.py
import numpy as np
from collections import deque
from helpers.obs_utils import parse_observation

class FeatureExtractorCont:
    """
    Stateful continuous feature extractor for linear function approximation.
    Uses ONLY past prices (no leakage) via rolling windows.

    Features (all bounded):
      1) bias
      2) v (volume ratio)
      3) v_low  (near-empty indicator, 0..1)
      4) v_high (near-full indicator, 0..1)
      5-6) hour sin/cos
      7-8) weekday sin/cos
      9) z-score (rolling 24h)
      10) p_rank (rolling percentile rank, centered)
      11) trend (short mean - long mean, normalized)
      12) interaction z*v
      13) delta_1h (clipped)
    """

    def __init__(
        self,
        max_volume=100_000,
        w_short=24,          # z-score window
        w_rank=72,           # percentile rank window
        w_trend_short=12,    # trend short window
        w_trend_long=72,     # trend long window
        w_long=168,          # buffer length (must cover largest window)
        z_clip=5.0,
        d1_clip=200.0,
        inter_clip=5.0,
    ):
        self.max_volume = max_volume
        self.w_short = int(w_short)
        self.w_rank = int(w_rank)
        self.w_trend_short = int(w_trend_short)
        self.w_trend_long = int(w_trend_long)
        self.w_long = int(max(w_long, w_trend_long, w_rank, w_short))
        self.z_clip = float(z_clip)
        self.d1_clip = float(d1_clip)
        self.inter_clip = float(inter_clip)
        self.reset()

    def reset(self):
        self.prices = deque(maxlen=self.w_long)
        self.prev_price = None

    @staticmethod
    def _cyc(x, period):
        ang = 2.0 * np.pi * (x / period)
        return float(np.sin(ang)), float(np.cos(ang))

    @staticmethod
    def _safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    def __call__(self, observation):
        obs = parse_observation(observation)

        # ---- volume features ----
        v = float(obs["volume"]) / float(self.max_volume)
        v = float(np.clip(v, 0.0, 1.0))

        # near empty/full indicators (0..1)
        v_low = float(np.clip((0.10 - v) / 0.10, 0.0, 1.0))
        v_high = float(np.clip((v - 0.90) / 0.10, 0.0, 1.0))

        # ---- cyc time ----
        h_sin, h_cos = self._cyc(obs["hour"], 24)
        w_sin, w_cos = self._cyc(obs["weekday"], 7)

        # ---- price context ----
        price = float(obs["price"])

        # delta 1h (clipped)
        delta_1h = 0.0 if self.prev_price is None else (price - self.prev_price)
        delta_1h = float(np.clip(delta_1h, -self.d1_clip, self.d1_clip))

        # rolling z-score (short window)
        hist = list(self.prices)
        short = hist[-self.w_short:] if len(hist) > 0 else []
        if len(short) >= 5:
            mu = float(np.mean(short))
            sd = float(np.std(short)) + 1e-6
            z = (price - mu) / sd
        else:
            z = 0.0
        z = float(np.clip(z, -self.z_clip, self.z_clip))

        # rolling percentile rank (centered)
        rank_hist = hist[-self.w_rank:] if len(hist) > 0 else []
        if len(rank_hist) >= 10:
            arr = np.asarray(rank_hist, dtype=np.float32)
            p_rank = float(np.mean(arr <= price))  # 0..1
            p_rank = float(np.clip(p_rank - 0.5, -0.5, 0.5))  # center
        else:
            p_rank = 0.0

        # trend: short mean - long mean, normalized
        t_short = hist[-self.w_trend_short:] if len(hist) > 0 else []
        t_long = hist[-self.w_trend_long:] if len(hist) > 0 else []
        mu_s = self._safe_mean(t_short)
        mu_l = self._safe_mean(t_long)
        denom = abs(mu_l) + 1e-6
        trend = (mu_s - mu_l) / denom
        trend = float(np.clip(trend, -1.0, 1.0))

        # interaction
        z_x_v = float(np.clip(z * v, -self.inter_clip, self.inter_clip))

        # update history AFTER computing features (no leakage)
        self.prices.append(price)
        self.prev_price = price

        # ---- assemble feature vector (bounded) ----
        phi = np.array(
            [
                1.0,
                v, v_low, v_high,
                h_sin, h_cos,
                w_sin, w_cos,
                z, p_rank, trend,
                z_x_v,
                delta_1h,
            ],
            dtype=np.float32,
        )

        # final safety clip (keeps linear learning stable)
        phi = np.clip(phi, -5.0, 5.0)
        return phi
