"""Shared helpers for the training entry points."""
from __future__ import annotations
import os
import yaml
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)
    # Resolve dataco path relative to project root.
    if not os.path.isabs(cfg["data"]["dataco_path"]):
        cfg["data"]["dataco_path"] = str(ROOT / cfg["data"]["dataco_path"])
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        cfg["device"] = "cpu"
    return cfg


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kl_warmup(step: int, total: int) -> float:
    return min(1.0, step / max(1, total))
