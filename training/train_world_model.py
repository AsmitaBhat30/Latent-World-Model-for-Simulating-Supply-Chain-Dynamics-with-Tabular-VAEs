"""Train the RSSM world model on sequences, using a pre-trained VAE encoder."""
from __future__ import annotations
import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from data import load_dataco_or_synthetic, TabularSupplyChainDataset, FeatureSchema
from models import RSSM, LatentWorldModel
from training.train_vae import build_vae
from training.common import load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", choices=["hierarchical", "vqvae"], default="vqvae")
    ap.add_argument("--config", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = cfg["device"]
    trajs, schema = load_dataco_or_synthetic(cfg)
    dataset = TabularSupplyChainDataset(trajs, window=args.window)
    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"] // 4, shuffle=True, drop_last=True)

    vae = build_vae(args.vae, schema, cfg)
    ckpt_path = Path(__file__).resolve().parents[1] / args.out / f"vae_{args.vae}.pt"
    if ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location="cpu")
        vae.load_state_dict(sd["model"])
        print(f"[WM] loaded pre-trained VAE from {ckpt_path}")
    else:
        print(f"[WM] WARNING: no pre-trained VAE found at {ckpt_path}; training from scratch")
    vae = vae.to(device)

    rs = cfg["world_model"]["rssm"]
    # Determine the embed dim that the VAE encoder outputs.
    embed_dim = cfg["vae"]["shared"]["d_token"]
    latent_dim = cfg["vae"]["shared"]["d_token"] if args.vae == "hierarchical" else cfg["world_model"]["rssm"]["hidden"]

    # For the hierarchical VAE the decoder takes the top-down hidden (256 by default).
    if args.vae == "hierarchical":
        latent_dim_for_decoder = 256
    else:
        latent_dim_for_decoder = 256  # vqvae.post_vq -> decoder_hidden

    rssm = RSSM(
        embed_dim=embed_dim,
        action_dim=schema.action_dim,
        deterministic_dim=rs["deterministic_dim"],
        n_cats=rs["n_categoricals"] if args.vae == "vqvae" else 32,
        n_classes=32,
        hidden=rs["hidden"],
    ).to(device)
    wm = LatentWorldModel(vae, rssm, latent_dim=latent_dim_for_decoder,
                          action_dim=schema.action_dim, reward_bins=cfg["world_model"]["reward_bins"]).to(device)
    opt = torch.optim.AdamW(wm.parameters(), lr=3e-4, weight_decay=1e-6, eps=1e-5)

    steps = args.steps or cfg["training"]["world_model_steps"]
    log_every = cfg["training"]["log_every"]
    print(f"[WM] training for {steps} steps")

    it = iter(loader)
    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, info = wm.loss(batch, kl_balance=cfg["world_model"]["kl_balance"],
                             free_bits=cfg["world_model"]["free_bits"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
        opt.step()

        if step % log_every == 0 or step == 1:
            print(f"step {step:>6}  " + "  ".join(f"{k} {float(v):.3f}" for k, v in info.items()
                                                   if v.ndim == 0))

    out_dir = Path(__file__).resolve().parents[1] / args.out
    out_dir.mkdir(exist_ok=True)
    ckpt = out_dir / f"world_model_{args.vae}.pt"
    torch.save({"model": wm.state_dict()}, ckpt)
    print(f"[WM] saved to {ckpt}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
