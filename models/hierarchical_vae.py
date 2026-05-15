"""Top-down hierarchical VAE for tabular rows (NVAE / Very-Deep-VAE style).

Generative model::

    p(z_1) = N(0, I)
    p(z_l | z_{<l}) = N(mu_l(h_l), sigma_l(h_l))      for l = 2..L
    p(x | z_{1:L}) = TabularDecoderHeads(z_top)

Inference model (bottom-up encoder + top-down posterior corrections)::

    q(z_l | x, z_{<l}) = N(mu_l + delta_mu(h_bu),  exp(log sigma_l + delta_log_sigma(h_bu)))

This is the standard residual posterior parameterisation that NVAE / Very-Deep-
VAEs use; it consistently outperforms a flat z when the data has multi-scale
structure -- which mixed-type tabular rows certainly do (categorical clusters
at the top, numerical noise at the bottom).
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import FeatureSchema
from .tabular_encoder import TabularTransformerEncoder, TabularDecoderHeads


def _gaussian_params(h: torch.Tensor, dim: int):
    mu, log_sigma = h.chunk(2, dim=-1)
    log_sigma = log_sigma.clamp(-5.0, 2.0)
    return mu, log_sigma


class _ResCell(nn.Module):
    """Tiny residual MLP cell used inside the top-down generator."""
    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = F.gelu(self.fc1(self.norm(x)))
        return x + self.fc2(h)


class HierarchicalTabularVAE(nn.Module):
    def __init__(
        self,
        schema: FeatureSchema,
        d_token: int = 64,
        n_heads: int = 4,
        n_blocks: int = 2,
        dropout: float = 0.1,
        z_dims: List[int] = (16, 16, 16),
        decoder_hidden: int = 256,
    ):
        super().__init__()
        self.schema = schema
        self.z_dims = list(z_dims)
        self.L = len(self.z_dims)

        # Bottom-up encoder produces a single context vector.
        self.bu_encoder = TabularTransformerEncoder(schema, d_token, n_heads, n_blocks, dropout)
        d_ctx = d_token

        # For each level: a prior network conditional on top-down state and a
        # posterior correction network conditional on (top-down, bottom-up).
        self.prior_nets = nn.ModuleList()
        self.post_correct_nets = nn.ModuleList()
        self.td_cells = nn.ModuleList()
        self.td_proj_in = nn.ModuleList()

        d_td = decoder_hidden
        self.td_init = nn.Parameter(torch.zeros(1, d_td))
        for l, zd in enumerate(self.z_dims):
            # First level: prior is standard normal (no network needed but we keep one for symmetry).
            self.prior_nets.append(nn.Linear(d_td, 2 * zd))
            self.post_correct_nets.append(nn.Linear(d_td + d_ctx, 2 * zd))
            self.td_proj_in.append(nn.Linear(zd, d_td))
            self.td_cells.append(_ResCell(d_td))

        self.decoder = TabularDecoderHeads(schema, latent_dim=d_td, hidden=decoder_hidden)

    # ------------------------- encoding / decoding ------------------------- #

    def encode(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        return self.bu_encoder(num, cat)

    def _sample_level(self, prior_logits, post_logits=None):
        p_mu, p_ls = _gaussian_params(prior_logits, dim=-1)
        if post_logits is None:
            mu, ls = p_mu, p_ls
        else:
            # Residual posterior parameterisation.
            d_mu, d_ls = _gaussian_params(post_logits, dim=-1)
            mu, ls = p_mu + d_mu, p_ls + d_ls
        eps = torch.randn_like(mu)
        z = mu + eps * ls.exp()
        return z, (p_mu, p_ls), (mu, ls)

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> dict:
        B = num.shape[0] if num.numel() else cat.shape[0]
        h_bu = self.encode(num, cat)
        h_td = self.td_init.expand(B, -1)

        kl_per_level = []
        zs = []
        for l in range(self.L):
            prior_logits = self.prior_nets[l](h_td)
            post_in = torch.cat([h_td, h_bu], dim=-1)
            post_logits = self.post_correct_nets[l](post_in)
            z, (p_mu, p_ls), (q_mu, q_ls) = self._sample_level(prior_logits, post_logits)
            prior = torch.distributions.Normal(p_mu, p_ls.exp())
            post = torch.distributions.Normal(q_mu, q_ls.exp())
            kl = torch.distributions.kl_divergence(post, prior).sum(-1)
            kl_per_level.append(kl)
            zs.append(z)
            # Update top-down state.
            h_td = self.td_cells[l](h_td + self.td_proj_in[l](z))

        decoded = self.decoder(h_td)
        return {
            "decoded": decoded,
            "h_td": h_td,
            "zs": zs,
            "kl_per_level": kl_per_level,
        }

    # ------------------------- losses ------------------------- #

    def loss(self, batch: dict, kl_weight: float = 1.0, free_bits: float = 1.0) -> Tuple[torch.Tensor, dict]:
        out = self.forward(batch["num"], batch["cat"])
        recon, info = self.decoder.recon_loss(out["decoded"], batch["num"], batch["cat"])
        kls = []
        for kl in out["kl_per_level"]:
            # free-bits per level
            kls.append(torch.clamp(kl.mean(), min=free_bits))
        kl_total = sum(kls)
        loss = recon + kl_weight * kl_total
        info.update({
            "kl_total": kl_total.detach(),
            "kl_levels": torch.stack([k.detach() for k in kls]),
            "loss": loss.detach(),
        })
        return loss, info

    # ------------------------- sampling ------------------------- #

    @torch.no_grad()
    def sample(self, n: int = 1, device: str = "cpu") -> torch.Tensor:
        h_td = self.td_init.expand(n, -1).to(device)
        for l in range(self.L):
            prior_logits = self.prior_nets[l](h_td)
            z, *_ = self._sample_level(prior_logits)
            h_td = self.td_cells[l](h_td + self.td_proj_in[l](z))
        return h_td

    @torch.no_grad()
    def latent(self, num, cat) -> torch.Tensor:
        """Return the top-down summary used as the "z_t" handed to the world model."""
        out = self.forward(num, cat)
        return out["h_td"]
