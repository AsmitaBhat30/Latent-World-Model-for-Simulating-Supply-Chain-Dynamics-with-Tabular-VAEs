"""Pre-train either the Hierarchical VAE or the VQ-VAE on the tabular rows.

Usage:
    python training/train_vae.py --vae hierarchical
    python training/train_vae.py --vae vqvae
"""
from __future__ import annotations
import sys, os, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from data import load_dataco_or_synthetic, TabularSupplyChainDataset
from models import HierarchicalTabularVAE, TabularVQVAE
from training.common import load_config, set_seed, kl_warmup


def build_vae(name: str, schema, cfg):
    sh = cfg["vae"]["shared"]
    if name == "hierarchical":
        c = cfg["vae"]["hierarchical"]
        return HierarchicalTabularVAE(
            schema=schema,
            d_token=sh["d_token"],
            n_heads=sh["n_attn_heads"],
            n_blocks=sh["n_tf_blocks"],
            dropout=sh["dropout"],
            z_dims=c["z_dims"],
        )
    elif name == "vqvae":
        c = cfg["vae"]["vqvae"]
        return TabularVQVAE(
            schema=schema,
            d_token=sh["d_token"],
            n_heads=sh["n_attn_heads"],
            n_blocks=sh["n_tf_blocks"],
            dropout=sh["dropout"],
            n_codes=c["n_codes"],
            code_dim=c["code_dim"],
            n_quantizers=c["n_quantizers"],
            commitment=c["commitment"],
            ema_decay=c["ema_decay"],
            dead_code_threshold=c["dead_code_threshold"],
        )
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", choices=["hierarchical", "vqvae"], required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = cfg["device"]
    trajs, schema = load_dataco_or_synthetic(cfg)
    dataset = TabularSupplyChainDataset(trajs, window=1)   # row-level for VAE
    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"], shuffle=True, drop_last=True)

    model = build_vae(args.vae, schema, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)

    steps = args.steps or cfg["training"]["vae_steps"]
    log_every = cfg["training"]["log_every"]
    print(f"[VAE/{args.vae}] training for {steps} steps over {len(dataset)} rows")

    it = iter(loader)
    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = {k: v.squeeze(1).to(device) for k, v in batch.items()}

        if args.vae == "hierarchical":
            warm = kl_warmup(step, cfg["vae"]["hierarchical"]["warmup_steps"])
            loss, info = model.loss(batch, kl_weight=warm, free_bits=cfg["vae"]["hierarchical"]["free_bits"])
        else:
            loss, info = model.loss(batch)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % log_every == 0 or step == 1:
            msg = f"step {step:>6}  loss {float(loss):.3f}"
            for k in ("recon", "kl_total", "vq_loss", "codebook_usage"):
                if k in info:
                    v = info[k]
                    msg += f"  {k} {float(v) if v.ndim == 0 else v.tolist()}"
            print(msg, flush=True)

    out_dir = Path(__file__).resolve().parents[1] / args.out
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / f"vae_{args.vae}.pt"
    torch.save({"model": model.state_dict(), "schema": schema.__dict__}, ckpt_path)
    print(f"[VAE/{args.vae}] saved to {ckpt_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
