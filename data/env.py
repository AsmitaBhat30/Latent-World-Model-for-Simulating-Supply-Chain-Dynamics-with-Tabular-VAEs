"""Multi-echelon inventory MDP used as the *real* env for data collection
and for ground-truth comparisons during agent evaluation.

The imagined env that RL agents actually train on is the latent world model
(`models/world_model.py`); this env is the source of "real" rollouts used
both to fit the world model and to score it.
"""
from __future__ import annotations

import math
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_BASE = gym.Env
except Exception:   # pragma: no cover - fallback for environments without gymnasium
    class _FakeBox:
        def __init__(self, low, high, shape, dtype):
            self.low = low; self.high = high; self.shape = shape; self.dtype = dtype
    class _FakeSpaces:
        @staticmethod
        def Box(low, high, shape, dtype):  # noqa: N802
            return _FakeBox(low, high, shape, dtype)
    spaces = _FakeSpaces()
    class _FakeEnv:
        metadata = {"render_modes": []}
    _GYM_BASE = _FakeEnv


class MultiEchelonInventoryEnv(_GYM_BASE):
    """A small but non-trivial inventory MDP.

    Observation
    -----------
    A vector of size ``2 + len(features)`` containing
    ``[inventory, in_transit, ...exogenous_features]``.  Exogenous features
    are sampled from the same generative process used in the synthetic dataset
    so the encoder distributions match.

    Action
    ------
    A continuous reorder quantity in ``[0, 1]``; the env rescales it to a
    realistic order size.

    Reward
    ------
    ``- holding_cost * on_hand - stockout_cost * unmet_demand``.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_len: int = 32,
        holding_cost: float = 0.5,
        stockout_cost: float = 4.0,
        lead_time_mean: float = 3.0,
        lead_time_std: float = 1.0,
        max_order: float = 200.0,
        seed: int = 0,
    ):
        super().__init__()
        self.episode_len = episode_len
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.lead_time_mean = lead_time_mean
        self.lead_time_std = lead_time_std
        self.max_order = max_order
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self._reset_internals()

    def _reset_internals(self):
        self.t = 0
        self.inventory = 50.0
        self.in_transit = 0.0
        self.pending: list[tuple[int, float]] = []   # (arrival_t, qty)
        # SKU latent parameters (re-sampled per episode).
        self.base = float(self.rng.uniform(20, 120))
        self.season_amp = float(self.rng.uniform(0.1, 0.6))
        self.season_phase = float(self.rng.uniform(0, 2 * math.pi))
        self.price = float(self.rng.uniform(10, 200))

    def reset(self, *, seed: int | None = None, options=None):  # type: ignore[override]
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_internals()
        return self._obs(0.0, 0.0), {}

    def _expected_demand(self):
        return self.base * (1 + self.season_amp * math.sin(2 * math.pi * self.t / 30 + self.season_phase))

    def _obs(self, demand: float, last_reward: float):
        d_expected = self._expected_demand()
        ship_real = float(self.rng.normal(3.0, 1.0))
        ship_sched = 3.0
        return np.array(
            [
                self.inventory,
                self.in_transit,
                d_expected,
                demand,
                self.price,
                ship_real,
                ship_sched,
                last_reward,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        action = float(np.asarray(action).reshape(-1)[0])
        order_qty = float(np.clip(action, 0.0, 1.0)) * self.max_order
        # Schedule arrival with stochastic lead time.
        lead = max(1, int(round(self.rng.normal(self.lead_time_mean, self.lead_time_std))))
        self.pending.append((self.t + lead, order_qty))
        self.in_transit = sum(q for (_, q) in self.pending)

        # Realised demand.
        demand_mean = max(1.0, self._expected_demand())
        demand = float(self.rng.poisson(demand_mean))

        # Receive arrivals.
        arrived = sum(q for (a, q) in self.pending if a <= self.t)
        self.pending = [(a, q) for (a, q) in self.pending if a > self.t]
        self.inventory += arrived

        # Sell.
        sold = min(self.inventory, demand)
        unmet = demand - sold
        self.inventory -= sold

        reward = -(self.holding_cost * self.inventory + self.stockout_cost * unmet)
        self.t += 1
        terminated = self.t >= self.episode_len
        return self._obs(demand, reward), float(reward), terminated, False, {"demand": demand, "unmet": unmet}
