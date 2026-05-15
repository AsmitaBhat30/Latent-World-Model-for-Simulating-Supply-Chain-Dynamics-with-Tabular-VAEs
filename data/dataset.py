"""Tabular dataset utilities for the DataCo Smart Supply Chain dataset.

If the raw DataCo CSV is available locally, we use it directly: column types are
inferred, categoricals are integer-encoded, numericals are standardized, and we
construct per-SKU/per-warehouse trajectories suitable for sequence VAEs and the
RSSM world model.

If the CSV is not present, we transparently fall back to a synthetic generator
that emits the same `FeatureSchema` so all downstream code (VAEs, world model,
RL agents, UI) is dataset-agnostic.
"""
from __future__ import annotations

import os
import math
import json
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ----------------------------- Feature schema ----------------------------- #

@dataclass
class FeatureSchema:
    """Describes the tabular features in a dataset.

    The schema is the *single source of truth* for the rest of the pipeline:
    encoders read `categoricals` to build embedding tables and `numericals`
    to size the numerical projection; decoders read it to build per-feature
    output heads (softmax for categoricals, Gaussian for numericals).
    """
    numericals: List[str] = field(default_factory=list)
    categoricals: List[str] = field(default_factory=list)
    cat_cardinalities: Dict[str, int] = field(default_factory=dict)
    # Mean/std for numericals, applied at preprocessing time.
    num_means: Dict[str, float] = field(default_factory=dict)
    num_stds: Dict[str, float] = field(default_factory=dict)
    # Action / reward / continuation fields (set when constructing trajectories)
    action_dim: int = 1
    reward_field: str = "reward"
    done_field: str = "done"

    @property
    def n_num(self) -> int:  # noqa: D401 - simple property
        return len(self.numericals)

    @property
    def n_cat(self) -> int:
        return len(self.categoricals)

    def total_feature_tokens(self) -> int:
        """Number of FT-Transformer tokens (one per feature + CLS)."""
        return self.n_num + self.n_cat + 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# ----------------------------- DataCo loader ----------------------------- #

# DataCo column groups that we care about. Column names follow the public CSV
# header on Kaggle (`DataCoSupplyChainDataset.csv`). Anything missing is dropped
# automatically and the schema adapts.
DATACO_NUMERICS = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Profit Ratio",
    "Product Price",
    "Sales",
]
DATACO_CATEGORICALS = [
    "Type",
    "Delivery Status",
    "Late_delivery_risk",
    "Category Name",
    "Department Name",
    "Customer Segment",
    "Market",
    "Order Region",
    "Shipping Mode",
]
DATACO_TIME_FIELD = "order date (DateOrders)"
DATACO_GROUP_FIELDS = ["Product Card Id", "Order Region"]  # per-SKU/region series


def _load_dataco_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    # Keep only the fields we use; drop rows missing the time field.
    cols = [DATACO_TIME_FIELD] + DATACO_GROUP_FIELDS + DATACO_NUMERICS + DATACO_CATEGORICALS
    cols = [c for c in cols if c in df.columns]
    df = df[cols].dropna(subset=[DATACO_TIME_FIELD]).copy()
    df[DATACO_TIME_FIELD] = pd.to_datetime(df[DATACO_TIME_FIELD], errors="coerce")
    df = df.dropna(subset=[DATACO_TIME_FIELD])
    return df


def _make_dataco_trajectories(
    df: pd.DataFrame,
    episode_len: int,
    holding_cost: float,
    stockout_cost: float,
) -> Tuple[List[Dict[str, np.ndarray]], FeatureSchema]:
    """Aggregate DataCo records into per-(SKU, region) daily trajectories and
    synthesise inventory state + a coarse reward signal so we can run RL on top.

    We treat ``Order Item Quantity`` as exogenous demand and roll a simple
    inventory equation::

        inv_{t+1} = max(0, inv_t + reorder_t - demand_t)

    where ``reorder_t`` is a *recorded* action proxy: the rolling demand mean
    (the historical "policy" we want to improve upon).  The agent will later
    re-learn this in the imagined env.
    """
    numericals = [c for c in DATACO_NUMERICS if c in df.columns]
    categoricals = [c for c in DATACO_CATEGORICALS if c in df.columns]

    # Integer-encode categoricals globally for a stable cardinality.
    cat_cardinalities = {}
    for c in categoricals:
        codes, uniques = pd.factorize(df[c].astype("string").fillna("__nan__"))
        df[c] = codes
        cat_cardinalities[c] = int(len(uniques))

    # Aggregate to daily granularity per (Product, Region).
    df["__day__"] = df[DATACO_TIME_FIELD].dt.floor("D")
    group_keys = [c for c in DATACO_GROUP_FIELDS if c in df.columns] + ["__day__"]
    agg = {c: "mean" for c in numericals}
    for c in categoricals:
        agg[c] = "first"
    daily = df.groupby(group_keys, sort=True).agg(agg).reset_index()

    # Standardise numericals globally.
    num_means = {c: float(daily[c].mean()) for c in numericals}
    num_stds = {c: float(daily[c].std() + 1e-6) for c in numericals}
    for c in numericals:
        daily[c] = (daily[c] - num_means[c]) / num_stds[c]

    # Build trajectories grouped on (Product, Region).
    traj_keys = [c for c in DATACO_GROUP_FIELDS if c in df.columns]
    trajectories: List[Dict[str, np.ndarray]] = []

    # Demand proxy: original (unscaled) order item quantity, recovered from the
    # standardised column for the reward function.
    qty_col = "Order Item Quantity" if "Order Item Quantity" in numericals else None

    for _, sub in daily.groupby(traj_keys, sort=False):
        sub = sub.sort_values("__day__")
        if len(sub) < episode_len + 1:
            continue
        # Sliding windows -> several episodes per group.
        for start in range(0, len(sub) - episode_len, episode_len):
            window = sub.iloc[start : start + episode_len].reset_index(drop=True)
            num_arr = window[numericals].to_numpy(dtype=np.float32) if numericals else np.zeros((episode_len, 0), np.float32)
            cat_arr = window[categoricals].to_numpy(dtype=np.int64) if categoricals else np.zeros((episode_len, 0), np.int64)

            if qty_col is not None:
                demand = window[qty_col].to_numpy(dtype=np.float32) * num_stds[qty_col] + num_means[qty_col]
            else:
                demand = np.ones(episode_len, dtype=np.float32)
            demand = np.clip(demand, 0.0, None)

            # Heuristic action: rolling mean of demand (a baseline (s,S)-style policy).
            action = np.convolve(demand, np.ones(3) / 3.0, mode="same").astype(np.float32)

            # Inventory roll-out (purely deterministic given action + demand).
            inv = np.zeros(episode_len, dtype=np.float32)
            inv[0] = demand.mean()
            for t in range(1, episode_len):
                inv[t] = max(0.0, inv[t - 1] + action[t - 1] - demand[t - 1])
            stockouts = np.clip(demand - (inv + action), 0.0, None)
            reward = -(holding_cost * inv + stockout_cost * stockouts)

            done = np.zeros(episode_len, dtype=np.float32)
            done[-1] = 1.0
            trajectories.append(
                dict(
                    num=num_arr,
                    cat=cat_arr,
                    action=action[:, None],   # (T, 1)
                    reward=reward,
                    done=done,
                    inv=inv,
                    demand=demand,
                )
            )

    schema = FeatureSchema(
        numericals=numericals,
        categoricals=categoricals,
        cat_cardinalities=cat_cardinalities,
        num_means=num_means,
        num_stds=num_stds,
        action_dim=1,
    )
    return trajectories, schema


# ----------------------------- Synthetic fallback ----------------------------- #

def _make_synthetic_trajectories(
    n_skus: int,
    n_warehouses: int,
    episode_len: int,
    n_episodes: int,
    holding_cost: float,
    stockout_cost: float,
    seed: int = 0,
) -> Tuple[List[Dict[str, np.ndarray]], FeatureSchema]:
    """Generate trajectories whose schema mirrors the DataCo loader.

    The generative process is a multi-SKU, multi-warehouse seasonal demand model
    with stochastic lead times -- complex enough to make the VAE non-trivial,
    structured enough that we can audit it visually.
    """
    rng = np.random.default_rng(seed)

    # Schema mirrors the DataCo subset that downstream models expect.
    numericals = [
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Order Item Quantity",
        "Order Item Discount Rate",
        "Order Item Product Price",
        "Order Item Profit Ratio",
        "Product Price",
        "Sales",
    ]
    categoricals = ["Type", "Delivery Status", "Category Name", "Department Name", "Market", "Shipping Mode"]
    cat_cardinalities = {"Type": 4, "Delivery Status": 4, "Category Name": 10, "Department Name": 5,
                         "Market": 4, "Shipping Mode": 4}

    # Per-SKU latent parameters that drive demand.
    base = rng.uniform(20, 120, size=n_skus)
    season_amp = rng.uniform(0.1, 0.6, size=n_skus)
    season_phase = rng.uniform(0, 2 * math.pi, size=n_skus)
    sku_categories = rng.integers(0, cat_cardinalities["Category Name"], size=n_skus)
    sku_departments = rng.integers(0, cat_cardinalities["Department Name"], size=n_skus)
    sku_prices = rng.uniform(10, 200, size=n_skus)

    trajs: List[Dict[str, np.ndarray]] = []
    for _ in range(n_episodes):
        sku = rng.integers(0, n_skus)
        wh = rng.integers(0, n_warehouses)
        t0 = rng.integers(0, 1000)
        t = np.arange(t0, t0 + episode_len)

        demand_mean = base[sku] * (1 + season_amp[sku] * np.sin(2 * math.pi * t / 30 + season_phase[sku]))
        demand = rng.poisson(np.clip(demand_mean, 1, None)).astype(np.float32)

        ship_real = rng.normal(3.0, 1.0, size=episode_len).clip(0, None).astype(np.float32)
        ship_sched = rng.normal(3.0, 0.3, size=episode_len).clip(0, None).astype(np.float32)
        discount = rng.beta(2, 8, size=episode_len).astype(np.float32)
        price = np.full(episode_len, sku_prices[sku], dtype=np.float32)
        profit = (price * (1 - discount) - price * 0.6).astype(np.float32) / (price + 1e-6)
        sales = (demand * price * (1 - discount)).astype(np.float32)

        # Categoricals (mostly SKU-static, with a stochastic shipping mode).
        type_ = np.full(episode_len, rng.integers(0, cat_cardinalities["Type"]), dtype=np.int64)
        delivery_status = rng.integers(0, cat_cardinalities["Delivery Status"], size=episode_len, dtype=np.int64)
        category = np.full(episode_len, sku_categories[sku], dtype=np.int64)
        department = np.full(episode_len, sku_departments[sku], dtype=np.int64)
        market = np.full(episode_len, wh % cat_cardinalities["Market"], dtype=np.int64)
        ship_mode = rng.integers(0, cat_cardinalities["Shipping Mode"], size=episode_len, dtype=np.int64)

        # Baseline action policy: forecast-following with safety stock.
        sigma = max(1.0, float(demand.std()))
        action = np.clip(demand_mean + 0.7 * sigma, 0, None).astype(np.float32)

        inv = np.zeros(episode_len, dtype=np.float32)
        inv[0] = float(demand.mean())
        for k in range(1, episode_len):
            inv[k] = max(0.0, inv[k - 1] + action[k - 1] - demand[k - 1])
        stockouts = np.clip(demand - (inv + action), 0.0, None)
        reward = -(holding_cost * inv + stockout_cost * stockouts).astype(np.float32)

        num_arr = np.stack([ship_real, ship_sched, demand, discount, price, profit, price, sales], axis=1)
        cat_arr = np.stack([type_, delivery_status, category, department, market, ship_mode], axis=1)

        done = np.zeros(episode_len, dtype=np.float32)
        done[-1] = 1.0
        trajs.append(
            dict(
                num=num_arr,
                cat=cat_arr,
                action=action[:, None],
                reward=reward,
                done=done,
                inv=inv,
                demand=demand,
            )
        )

    # Standardise numericals after generation so the schema matches DataCo's.
    stacked = np.concatenate([t["num"] for t in trajs], axis=0)
    num_means = {c: float(stacked[:, i].mean()) for i, c in enumerate(numericals)}
    num_stds = {c: float(stacked[:, i].std() + 1e-6) for i, c in enumerate(numericals)}
    means = np.array([num_means[c] for c in numericals], dtype=np.float32)
    stds = np.array([num_stds[c] for c in numericals], dtype=np.float32)
    for t in trajs:
        t["num"] = (t["num"] - means) / stds

    schema = FeatureSchema(
        numericals=numericals,
        categoricals=categoricals,
        cat_cardinalities=cat_cardinalities,
        num_means=num_means,
        num_stds=num_stds,
        action_dim=1,
    )
    return trajs, schema


# ----------------------------- Torch dataset ----------------------------- #

class TabularSupplyChainDataset(Dataset):
    """Wraps a list of trajectory dicts as a torch Dataset.

    Each item is a contiguous window of ``window`` steps.  The VAE consumes
    `num` + `cat`; the world model additionally consumes `action`, `reward`,
    and `done`.
    """

    def __init__(self, trajectories: List[Dict[str, np.ndarray]], window: int):
        self.window = window
        # Pre-slice into all valid windows.
        self.index: List[Tuple[int, int]] = []
        self.trajs = trajectories
        for ti, t in enumerate(trajectories):
            T = t["num"].shape[0]
            for start in range(0, T - window + 1):
                self.index.append((ti, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ti, start = self.index[i]
        t = self.trajs[ti]
        sl = slice(start, start + self.window)
        return dict(
            num=torch.from_numpy(t["num"][sl]),
            cat=torch.from_numpy(t["cat"][sl]),
            action=torch.from_numpy(t["action"][sl]),
            reward=torch.from_numpy(t["reward"][sl]),
            done=torch.from_numpy(t["done"][sl]),
            inv=torch.from_numpy(t["inv"][sl]),
            demand=torch.from_numpy(t["demand"][sl]),
        )


# ----------------------------- Top-level entry point ----------------------------- #

def load_dataco_or_synthetic(cfg) -> Tuple[List[Dict[str, np.ndarray]], FeatureSchema]:
    """Returns (trajectories, schema).  Uses DataCo if the CSV is present,
    otherwise generates synthetic data with the same schema."""
    path = Path(cfg["data"]["dataco_path"])
    if cfg["data"]["dataset"] == "dataco" and path.exists():
        df = _load_dataco_csv(str(path))
        return _make_dataco_trajectories(
            df,
            episode_len=cfg["data"]["synthetic"]["episode_len"],
            holding_cost=cfg["env"]["holding_cost"],
            stockout_cost=cfg["env"]["stockout_cost"],
        )
    return _make_synthetic_trajectories(
        n_skus=cfg["data"]["synthetic"]["n_skus"],
        n_warehouses=cfg["data"]["synthetic"]["n_warehouses"],
        episode_len=cfg["data"]["synthetic"]["episode_len"],
        n_episodes=cfg["data"]["synthetic"]["n_episodes"],
        holding_cost=cfg["env"]["holding_cost"],
        stockout_cost=cfg["env"]["stockout_cost"],
        seed=cfg.get("seed", 0),
    )
