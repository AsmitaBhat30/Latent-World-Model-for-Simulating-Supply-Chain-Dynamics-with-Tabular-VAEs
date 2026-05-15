# Latent World Model for Supply Chain Dynamics with Tabular VAEs

A research-grade implementation of a **latent world model** for multi-echelon supply
chain control. The world model encodes highly structured, mixed-type business
records into a compressed latent space using **two competing tabular VAE flavours**
— a **VQ-VAE** with EMA codebook updates and dead-code revival, and a **top-down
hierarchical VAE** (NVAE / Very-Deep-VAE style) — then learns dynamics in that
latent space. Three RL agents (**Dreamer-V3, SAC, PPO**) are trained entirely on
**imagined rollouts** of the world model, and an interactive single-file React UI
lets you pick an inventory action and watch the model dream a multi-step
trajectory of demand, inventory positions, holding/stockout costs and reward.

> The project is built around the **DataCo Smart Supply Chain dataset** (Kaggle,
> `dataco-smart-supply-chain-for-big-data-analysis`). If the raw CSV is not
> present, the data pipeline transparently falls back to a tunable synthetic
> generator that matches the DataCo schema (orders, shipping, product hierarchy,
> demand) at full statistical fidelity for the dynamics of interest.

## Architecture

```
        ┌──────────────────────────────────────────────────────────────┐
        │                 Tabular world record  x_t                    │
        │   (mixed numerical / categorical / hierarchical features)    │
        └─────────────────────────┬────────────────────────────────────┘
                                  │
              ┌───────────────────┴────────────────────┐
              │            Tabular encoder             │
              │  per-feature embeddings + FT-Transformer│
              │     block + numerical Gaussian head     │
              └───────────────────┬────────────────────┘
                                  │  h_t
              ┌───────────────────┴────────────────────┐
              │     Latent representation z_t          │
              │  ┌─────────────────┐  ┌──────────────┐ │
              │  │   Hierarchical  │  │   VQ-VAE     │ │
              │  │   VAE (top-down │  │   EMA codes  │ │
              │  │   L=3 groups)   │  │   + RVQ      │ │
              │  └─────────────────┘  └──────────────┘ │
              └───────────────────┬────────────────────┘
                                  │
              ┌───────────────────┴────────────────────┐
              │   Recurrent State-Space Model (RSSM)   │
              │   deterministic h + stochastic z,      │
              │   KL-balancing, free-bits, symlog      │
              └───────────────────┬────────────────────┘
                                  │  imagined ẑ_{t+1}
              ┌───────────────────┴────────────────────┐
              │            Decoder heads                │
              │  • next-state x̂_{t+1}                  │
              │  • reward r̂  (symlog two-hot)          │
              │  • continuation γ̂                      │
              └─────────────────────────────────────────┘
                                  │
              ┌───────────────────┴────────────────────┐
              │   RL Agents on imagined rollouts        │
              │  Dreamer-V3 ───  SAC ───  PPO           │
              └─────────────────────────────────────────┘
```

## State-of-the-art design choices

| Component                | SotA technique used                                                                   |
|--------------------------|---------------------------------------------------------------------------------------|
| Tabular encoder          | FT-Transformer style feature tokenizer + transformer block                            |
| VQ-VAE                   | EMA codebook updates (van den Oord 2017 / Razavi 2019), commitment loss, dead-code revival, optional **Residual VQ** with 2 quantizer stages |
| Hierarchical VAE         | Top-down generation, bidirectional inference, KL-balancing, residual cells (NVAE)     |
| World model              | RSSM with separate deterministic GRU & stochastic head (Dreamer-V2/V3), free-bits     |
| Reward head              | Symlog transform + two-hot encoding (Dreamer-V3)                                      |
| Dreamer agent            | Actor-critic on imagined rollouts, λ-returns, EMA target critic, percentile normalization |
| SAC                      | Twin Q, target entropy auto-tuning, latent-rollout replay                              |
| PPO                      | GAE-λ, clipped surrogate, value clipping                                              |
| Privacy / sensitivity    | Latent codebook acts as a discretizer (k-anonymity-like compression of sensitive rows); optional DP-noise hook at the encoder output |

## Layout

```
data/        DataCo loader, synthetic fallback, MDP wrapper
models/      Tabular encoder, Hierarchical VAE, VQ-VAE, RSSM world model
agents/      Dreamer-V3, SAC, PPO trained on latent rollouts
training/    train_vae.py / train_world_model.py / train_agents.py
evaluation/  compare_vaes.py (ELBO / recon / codebook usage), compare_agents.py
ui/          index.html — single-file React UI for interactive rollouts
configs/     default.yaml
artifacts/   sample precomputed rollouts the UI consumes
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Train (or skip; precomputed rollouts ship in artifacts/)
python training/train_vae.py          --vae hierarchical
python training/train_vae.py          --vae vqvae
python training/train_world_model.py  --vae vqvae
python training/train_agents.py       --agent dreamer
python training/train_agents.py       --agent sac
python training/train_agents.py       --agent ppo

# 2. Evaluate
python evaluation/compare_vaes.py
python evaluation/compare_agents.py

# 3. Open the UI
open ui/index.html
```

The UI is a self-contained HTML file that uses React + Recharts via CDN and loads
the precomputed rollouts from `artifacts/rollouts.json`.
