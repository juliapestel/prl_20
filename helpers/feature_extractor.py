# helpers/feature_extractor.py
import numpy as np
from collections import deque
from helpers.obs_utils import parse_observation

class FeatureExtractor:
    """
    Stateful extractor: houdt prijs-history bij zodat je rolling stats kan maken
    zonder future leakage.
    """
    def __init__(self, max_volume=100_000, w_short=24, w_long=168):
        self.max_volume = max_volume
        self.w_short = w_short
        self.w_long = w_long
        self.reset()

    def reset(self):
        self.prices = deque(maxlen=self.w_long)  # store last 168h
        self.prev_price = None

    # ---------- helpers ----------
    @staticmethod
    def _cyc(x, period):
        ang = 2.0 * np.pi * (x / period)
        return np.sin(ang), np.cos(ang)

    @staticmethod
    def _bin(x, edges):
        # returns 0..len(edges)
        return int(np.digitize(x, edges))

    # ---------- main ----------
    def __call__(self, observation):
        obs = parse_observation(observation)

        # ---- reservoir (keep simple) ----
        v_ratio = obs["volume"] / self.max_volume
        vol_bin = self._bin(v_ratio, [0.1, 0.3, 0.7, 0.9])  # 0..4

        # ---- cyclical time encoding ----
        h_sin, h_cos = self._cyc(obs["hour"], 24)
        w_sin, w_cos = self._cyc(obs["weekday"], 7)
        d_sin, d_cos = self._cyc(obs["dayofyear"], 365)

        # discretize sin/cos to keep tabular state manageable
        # (edges chosen so bins are stable; you can tune #bins later)
        h_sin_bin = self._bin(h_sin, [-0.5, 0.0, 0.5])   # 0..3
        h_cos_bin = self._bin(h_cos, [-0.5, 0.0, 0.5])   # 0..3
        w_sin_bin = self._bin(w_sin, [-0.5, 0.0, 0.5])   # 0..3
        w_cos_bin = self._bin(w_cos, [-0.5, 0.0, 0.5])   # 0..3
        d_sin_bin = self._bin(d_sin, [-0.5, 0.0, 0.5])   # 0..3
        d_cos_bin = self._bin(d_cos, [-0.5, 0.0, 0.5])   # 0..3

        # ---- price context (online, no leakage) ----
        price = float(obs["price"])

        # momentum
        delta_1h = 0.0 if self.prev_price is None else (price - self.prev_price)
        price_24h_ago = self.prices[-24] if len(self.prices) >= 24 else None
        delta_24h = 0.0 if price_24h_ago is None else (price - price_24h_ago)

        # rolling mean/std (short window)
        short = list(self.prices)[-self.w_short:] if len(self.prices) > 0 else []
        if len(short) >= 5:
            mu = float(np.mean(short))
            sd = float(np.std(short)) + 1e-6
            z = (price - mu) / sd
        else:
            z = 0.0

        # discretize price context
        z_bin = self._bin(z, [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])        # 0..6
        d1_bin = self._bin(delta_1h, [-20, -5, -1, 1, 5, 20])          # 0..6 (tune)
        d24_bin = self._bin(delta_24h, [-40, -10, -2, 2, 10, 40])      # 0..6 (tune)

        # update history AFTER computing features (no leakage)
        self.prices.append(price)
        self.prev_price = price

        # return discrete state
        return (
    vol_bin,
    h_sin_bin, h_cos_bin,
    w_sin_bin, w_cos_bin,
    z_bin
)


