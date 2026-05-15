"""Head-to-head VAE comparison: ELBO, reconstruction NLL, latent geometry,
and codebook utilization (VQ-VAE only).

Outputs a small JSON summary that the README and the React UI can both consume.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from data import load_dataco_or_synthetic, TabularSupplyChainDataset
from models import HierarchicalTabularVAE, TabularVQVAE
from training.train_vae import build_vae
from training.common import load_config


def _evaluate(model, loader, device, n_batches: int = 50, is_vq: bool = False):
    model.eval()
    recon, kl, vqcomm, usage = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches: break
            batch = {k: v.squeeze(1).to(device) for k, v in batch.items()}
            if is_vq:
                _, info = model.loss(batch)
                recon.append(float(info["recon"]))
                vqcomm.append(float(info["vq_loss"]))
                usage.append(info["codebook_usage"].mean().item())
            else:
                _, info = model.loss(batch, kl_weight=1.0, free_bits=0.0)
                recon.append(float(info["recon"]))
                kl.append(float(info["kl_total"]))
    out = dict(recon=float(sum(recon)/len(recon)))
    if is_vq:
        out["vq_commit"] = float(sum(vqcomm)/len(vqcomm))
        out["codebook_usage"] = float(sum(usage)/len(usage))
    else:
        out["kl"] = float(sum(kl)/len(kl))
        out["elbo"] = -(out["recon"] + out["kl"])
    return out


def main():
    cfg = load_config()
    device = cfg["device"]
    trajs, schema = load_dataco_or_synthetic(cfg)
    dataset = TabularSupplyChainDataset(trajs, window=1)
    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"], shuffle=False, drop_last=True)

    results = {}
    root = Path(__file__).resolve().parents[1]
    for name in ("hierarchical", "vqvae"):
        model = build_vae(name, schema, cfg).to(device)
        ckpt = root / "checkpoints" / f"vae_{name}.pt"
        if ckpt.exists():
            sd = torch.load(ckpt, map_location=device)
            model.load_state_dict(sd["model"])
        results[name] = _evaluate(model, loader, device, is_vq=(name == "vqvae"))
        print(name, results[name])

    (root / "artifacts").mkdir(exist_ok=True)
    with open(root / "artifacts" / "vae_comparison.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
