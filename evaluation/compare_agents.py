"""Roll out Dreamer, SAC, and PPO on the real MultiEchelonInventoryEnv and
compare the average return per episode.

Outputs ``artifacts/agent_comparison.json``.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data import MultiEchelonInventoryEnv, load_dataco_or_synthetic
from models import RSSM, LatentWorldModel
from agents import DreamerV3Agent, SACAgent, PPOAgent
from training.train_vae import build_vae
from training.common import load_config


def _rollout(env: MultiEchelonInventoryEnv, agent, wm, schema, device, n_episodes: int = 20, horizon: int = 32):
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_return = 0.0
        # Seed the WM state from a flat embedding -- we use a zero "tabular row"
        # to bootstrap since the real env's observation space doesn't match the
        # tabular schema directly. The agent still consumes RSSM features.
        num = torch.zeros(1, schema.n_num, device=device)
        cat = torch.zeros(1, schema.n_cat, dtype=torch.long, device=device)
        action = torch.tensor([[0.5]], device=device)
        with torch.no_grad():
            embed = wm.embed(num, cat)
            state = wm.rssm.initial(1, device)
            state, _, _ = wm.rssm.step(state, action, embed=embed)
        for t in range(horizon):
            feat = torch.cat([state["h"], state["z"].flatten(-2)], dim=-1)
            with torch.no_grad():
                a = agent.act(feat, deterministic=True)
            obs, r, term, trunc, _ = env.step(a.cpu().numpy()[0])
            ep_return += r
            if term or trunc:
                break
            with torch.no_grad():
                state, _, _ = wm.rssm.step(state, a, embed=None)
        returns.append(ep_return)
    return float(np.mean(returns)), float(np.std(returns))


def main():
    cfg = load_config()
    device = cfg["device"]
    trajs, schema = load_dataco_or_synthetic(cfg)
    vae = build_vae("vqvae", schema, cfg)
    rs = cfg["world_model"]["rssm"]
    rssm = RSSM(
        embed_dim=cfg["vae"]["shared"]["d_token"], action_dim=schema.action_dim,
        deterministic_dim=rs["deterministic_dim"], n_cats=rs["n_categoricals"], n_classes=32,
        hidden=rs["hidden"],
    )
    wm = LatentWorldModel(vae, rssm, latent_dim=256, action_dim=schema.action_dim).to(device)
    root = Path(__file__).resolve().parents[1]
    wm_ckpt = root / "checkpoints" / "world_model_vqvae.pt"
    if wm_ckpt.exists():
        wm.load_state_dict(torch.load(wm_ckpt, map_location=device)["model"])

    env = MultiEchelonInventoryEnv(episode_len=cfg["env"]["horizon"],
                                   holding_cost=cfg["env"]["holding_cost"],
                                   stockout_cost=cfg["env"]["stockout_cost"])
    results = {}
    agent_classes = dict(dreamer=DreamerV3Agent, sac=SACAgent, ppo=PPOAgent)
    for name, cls in agent_classes.items():
        agent = cls(wm, cfg["agents"][name]).to(device)
        ckpt = root / "checkpoints" / f"agent_{name}.pt"
        if ckpt.exists():
            agent.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        mean, std = _rollout(env, agent, wm, schema, device, n_episodes=20, horizon=cfg["env"]["horizon"])
        results[name] = dict(mean=mean, std=std)
        print(name, results[name])
    (root / "artifacts").mkdir(exist_ok=True)
    with open(root / "artifacts" / "agent_comparison.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
