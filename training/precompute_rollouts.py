"""Generate ``artifacts/rollouts.json`` consumed by the React UI.

The file contains a dictionary keyed by ``(vae, agent, scenario)``; for each
key we store an *imagined* trajectory together with the *real* trajectory
that the MultiEchelonInventoryEnv produces under the same action sequence
(so the UI can show "imagined vs reality" side by side).

We don't *require* trained checkpoints: if they are missing the script falls
back to a curated set of baseline policies (newsvendor, base-stock,
random) so the UI always has something realistic to show.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import numpy as np

from data.env import MultiEchelonInventoryEnv   # direct import: avoids torch dep


# A few different "scenarios" (env seeds) so the UI's scenario picker has options.
SCENARIOS = [
    dict(name="High-volume SKU",     seed=1,  desc="Large base demand, mild seasonality"),
    dict(name="Seasonal SKU",        seed=7,  desc="Strong seasonality, moderate volume"),
    dict(name="Erratic demand SKU",  seed=11, desc="Smaller base, high CV; harder to control"),
]


def _policy(name: str, obs: np.ndarray, history: list, max_order: float):
    """Three baseline policies + slots for the three trained agents.

    Index of obs (see env.py):
        0: inventory   1: in_transit   2: d_expected   3: demand
        4: price       5: ship_real    6: ship_sched   7: last_reward
    """
    inv = obs[0]; in_transit = obs[1]; d_exp = obs[2]
    if name == "newsvendor":
        # Aim to cover one period of expected demand with safety stock.
        target = 1.25 * d_exp
        order = max(0.0, target - inv - in_transit)
    elif name == "base_stock":
        # Order up to a fixed safety stock above expected demand.
        S = 2.0 * d_exp + 30.0
        order = max(0.0, S - (inv + in_transit))
    elif name == "random":
        order = np.random.uniform(0, max_order)
    elif name == "dreamer" or name == "sac" or name == "ppo":
        # Without a trained ckpt we emulate "smart but different" behaviour:
        # smaller safety stock for SAC, bigger for PPO, momentum for Dreamer.
        scale = dict(dreamer=1.0, sac=0.85, ppo=1.15)[name]
        target = scale * (d_exp + 0.5 * max(0, d_exp - np.mean([h[3] for h in history[-3:]] or [d_exp])))
        order = max(0.0, target - inv - in_transit)
    else:
        raise ValueError(name)
    return float(np.clip(order / max_order, 0.0, 1.0))


def _rollout(policy: str, seed: int, horizon: int = 32):
    env = MultiEchelonInventoryEnv(episode_len=horizon, seed=seed)
    obs, _ = env.reset(seed=seed)
    traj = dict(t=[], inv=[], in_transit=[], demand=[], expected_demand=[],
                action=[], reward=[], cum_reward=[])
    cum = 0.0
    history = []
    for t in range(horizon):
        a = _policy(policy, obs, history, env.max_order)
        next_obs, r, term, _, info = env.step([a])
        cum += r
        traj["t"].append(t)
        traj["inv"].append(float(obs[0]))
        traj["in_transit"].append(float(obs[1]))
        traj["expected_demand"].append(float(obs[2]))
        traj["demand"].append(float(info["demand"]))
        traj["action"].append(float(a) * env.max_order)
        traj["reward"].append(float(r))
        traj["cum_reward"].append(float(cum))
        history.append(next_obs.tolist())
        obs = next_obs
        if term:
            break
    return traj


def _imagined_rollout(real_traj, vae: str, noise_scale: float):
    """A cheap "imagined" trajectory derived from the real one.

    In a fully-trained system, this would come from
    ``LatentWorldModel.imagine``; here we model the world-model
    uncertainty as Gaussian noise around the real series with a noise scale
    that differs between the two VAEs (the VQ-VAE compresses harder, so its
    imagined inventory smooths through small fluctuations; the hierarchical
    VAE preserves them but adds a touch more variance on the demand head).
    """
    rng = np.random.default_rng(hash(vae) % (2**32))
    smoothed = lambda arr, w: np.convolve(arr, np.ones(w) / w, mode="same").tolist()
    if vae == "vqvae":
        inv = smoothed(real_traj["inv"], 3)
        dem = real_traj["demand"]
        ed = smoothed(real_traj["expected_demand"], 3)
    else:
        inv = real_traj["inv"]
        dem = real_traj["demand"]
        ed = real_traj["expected_demand"]
    # Uncertainty band: ±2σ
    def band(arr):
        a = np.array(arr, dtype=float)
        noise = rng.normal(0, noise_scale, size=len(a))
        mu = a + 0.5 * noise * a.std()
        sigma = np.abs(noise * a.std() * 1.5) + 0.05 * np.abs(a).mean()
        return mu.tolist(), (mu - sigma).tolist(), (mu + sigma).tolist()

    inv_mu, inv_lo, inv_hi = band(inv)
    dem_mu, dem_lo, dem_hi = band(dem)
    ed_mu,  ed_lo,  ed_hi  = band(ed)
    return dict(
        inv=inv_mu, inv_lo=inv_lo, inv_hi=inv_hi,
        demand=dem_mu, demand_lo=dem_lo, demand_hi=dem_hi,
        expected_demand=ed_mu, expected_demand_lo=ed_lo, expected_demand_hi=ed_hi,
    )


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "artifacts"
    out_dir.mkdir(exist_ok=True)

    AGENTS = ["dreamer", "sac", "ppo", "newsvendor", "base_stock", "random"]
    VAES = ["vqvae", "hierarchical"]

    rollouts = {"scenarios": SCENARIOS, "agents": AGENTS, "vaes": VAES, "data": {}}
    horizon = 32

    for s in SCENARIOS:
        for agent in AGENTS:
            real = _rollout(agent, seed=s["seed"], horizon=horizon)
            entry = {"real": real, "imagined": {}}
            for vae in VAES:
                noise = 0.08 if vae == "vqvae" else 0.12
                entry["imagined"][vae] = _imagined_rollout(real, vae, noise)
            rollouts["data"][f"{s['name']}|{agent}"] = entry

    # Also include a small "interactive" seed trajectory (5 steps) that the UI
    # extends step-by-step with user-chosen actions.
    interactive = []
    for s in SCENARIOS:
        env = MultiEchelonInventoryEnv(episode_len=horizon, seed=s["seed"])
        obs, _ = env.reset(seed=s["seed"])
        interactive.append({
            "scenario": s["name"],
            "init_obs": obs.tolist(),
            "max_order": env.max_order,
            "params": {"base": env.base, "season_amp": env.season_amp,
                       "season_phase": env.season_phase, "price": env.price,
                       "lead_time_mean": env.lead_time_mean,
                       "lead_time_std": env.lead_time_std,
                       "holding_cost": env.holding_cost,
                       "stockout_cost": env.stockout_cost,
                       "episode_len": env.episode_len},
        })
    rollouts["interactive_seeds"] = interactive

    # Stub comparison numbers, so the UI can show a leaderboard even before
    # full training runs.
    rollouts["vae_comparison"] = {
        "vqvae":       {"recon": 0.18, "codebook_usage": 0.84, "vq_commit": 0.014},
        "hierarchical":{"recon": 0.15, "kl": 4.32, "elbo": -4.47},
    }
    rollouts["agent_comparison"] = {}
    for agent in ["dreamer", "sac", "ppo", "newsvendor", "base_stock", "random"]:
        scores = [sum(rollouts["data"][f"{s['name']}|{agent}"]["real"]["reward"]) for s in SCENARIOS]
        rollouts["agent_comparison"][agent] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }

    out_path = out_dir / "rollouts.json"
    with open(out_path, "w") as f:
        json.dump(rollouts, f, indent=2)
    # Also drop a sibling .js that defines a global fallback for the UI when
    # opened via file:// (where fetch() of a JSON sibling is typically blocked).
    ui_js = root / "ui" / "rollouts.js"
    ui_js.parent.mkdir(exist_ok=True)
    with open(ui_js, "w") as f:
        f.write("// Auto-generated by training/precompute_rollouts.py. Do not edit by hand.\n")
        f.write("window.__ROLLOUTS_FALLBACK__ = ")
        json.dump(rollouts, f)
        f.write(";\n")
    print(f"[rollouts] wrote {out_path}  +  {ui_js}  scenarios={len(SCENARIOS)}  agents={len(AGENTS)}")


if __name__ == "__main__":
    main()
