"""Train Dreamer-V3 / SAC / PPO on the latent world model and log returns.

Usage:
    python training/train_agents.py --agent dreamer
    python training/train_agents.py --agent sac
    python training/train_agents.py --agent ppo
"""
from __future__ import annotations
import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from data import load_dataco_or_synthetic, TabularSupplyChainDataset
from models import RSSM, LatentWorldModel
from agents import DreamerV3Agent, SACAgent, PPOAgent
from training.train_vae import build_vae
from training.common import load_config, set_seed


def _build_world_model(args, schema, cfg, device):
    vae = build_vae(args.vae, schema, cfg)
    embed_dim = cfg["vae"]["shared"]["d_token"]
    rs = cfg["world_model"]["rssm"]
    rssm = RSSM(
        embed_dim=embed_dim, action_dim=schema.action_dim,
        deterministic_dim=rs["deterministic_dim"],
        n_cats=rs["n_categoricals"], n_classes=32, hidden=rs["hidden"],
    )
    wm = LatentWorldModel(vae, rssm, latent_dim=256, action_dim=schema.action_dim,
                          reward_bins=cfg["world_model"]["reward_bins"]).to(device)
    ckpt = Path(__file__).resolve().parents[1] / "checkpoints" / f"world_model_{args.vae}.pt"
    if ckpt.exists():
        wm.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        print(f"[agent] loaded WM from {ckpt}")
    return wm


def _seed_states(wm, batch, device):
    """Run the WM posterior on a short observed prefix and return the final state."""
    with torch.no_grad():
        num = batch["num"].to(device); cat = batch["cat"].to(device)
        actions = batch["action"].to(device)
        B, T = num.shape[:2]
        flat_e = wm.embed(num.reshape(B * T, *num.shape[2:]), cat.reshape(B * T, *cat.shape[2:]))
        embeds = flat_e.view(B, T, -1)
        out = wm.rssm.observe(embeds, actions)
        return dict(h=out["h"][:, -1], z=out["z"][:, -1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["dreamer", "sac", "ppo"], required=True)
    ap.add_argument("--vae", choices=["hierarchical", "vqvae"], default="vqvae")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = cfg["device"]

    trajs, schema = load_dataco_or_synthetic(cfg)
    dataset = TabularSupplyChainDataset(trajs, window=8)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    wm = _build_world_model(args, schema, cfg, device)
    wm.eval()

    if args.agent == "dreamer":
        agent = DreamerV3Agent(wm, cfg["agents"]["dreamer"]).to(device)
    elif args.agent == "sac":
        agent = SACAgent(wm, cfg["agents"]["sac"]).to(device)
    else:
        agent = PPOAgent(wm, cfg["agents"]["ppo"]).to(device)

    steps = args.steps or cfg["training"]["agent_steps"]
    log_every = cfg["training"]["log_every"]
    it = iter(loader)
    history = []
    print(f"[agent/{args.agent}] training for {steps} steps")
    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        init_state = _seed_states(wm, batch, device)

        if args.agent == "dreamer":
            info = agent.update(init_state)
        elif args.agent == "sac":
            agent.collect(init_state, horizon=15)
            info = agent.update(batch_size=256)
        else:
            info = agent.update(init_state, horizon=cfg["env"]["horizon"])

        if step % log_every == 0 or step == 1:
            history.append({"step": step, **{k: float(v) for k, v in info.items() if isinstance(v, (int, float))}})
            print(f"step {step:>6}  " + "  ".join(f"{k} {v}" for k, v in info.items()))

    out_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    torch.save({"model": agent.state_dict()}, out_dir / f"agent_{args.agent}.pt")
    print(f"[agent/{args.agent}] done ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
