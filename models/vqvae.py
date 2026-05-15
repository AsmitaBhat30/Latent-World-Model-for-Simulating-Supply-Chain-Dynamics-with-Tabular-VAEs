"""Tabular VQ-VAE with EMA codebook updates and Residual VQ.

References:
 * van den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017
 * Razavi et al., "Generating Diverse High-Fidelity Images with VQ-VAE-2", 2019
 * Zeghidour et al., "SoundStream", 2021 (Residual VQ formulation we adopt)

Design choices:
 * **EMA codebook updates** for stable training and tiny gradient pressure on
   the encoder (avoids the auxiliary "embedding loss" term).
 * **Commitment loss** with the standard 0.25 weighting from van den Oord.
 * **Dead-code revival**: codes whose EMA usage falls below a threshold are
   re-initialised from a random encoder output. This is essential on tabular
   data, where codebook collapse is otherwise very easy.
 * **Residual VQ**: ``n_quantizers`` stacked quantizers, each operating on the
   residual of the previous; the resulting representation is the sum of all
   selected codes -- giving us exponentially-many effective codes while keeping
   individual codebooks small.
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import FeatureSchema
from .tabular_encoder import TabularTransformerEncoder, TabularDecoderHeads


class VectorQuantizerEMA(nn.Module):
    """A single EMA codebook."""
    def __init__(self, n_codes: int, code_dim: int, decay: float = 0.99, eps: float = 1e-5,
                 commitment: float = 0.25, dead_code_threshold: float = 1e-2):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps
        self.commitment = commitment
        self.dead_code_threshold = dead_code_threshold

        embed = torch.randn(n_codes, code_dim) * 0.01
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_codes))
        self.register_buffer("embed_avg", embed.clone())

    @torch.no_grad()
    def _revive_dead_codes(self, flat_inputs: torch.Tensor):
        usage = self.cluster_size
        dead = usage < self.dead_code_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return
        # Replace dead codes with random input vectors.
        idx = torch.randint(0, flat_inputs.shape[0], (n_dead,), device=flat_inputs.device)
        self.embed[dead] = flat_inputs[idx]
        self.embed_avg[dead] = flat_inputs[idx]
        self.cluster_size[dead] = 1.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (..., code_dim) encoder output to quantize.
        Returns:
            quantized: same shape as x, with straight-through estimator,
            indices:   (...,) discrete code indices,
            loss:      commitment loss (scalar).
        """
        flat = x.reshape(-1, self.code_dim)
        # Squared euclidean distance to each code.
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.t()
            + self.embed.pow(2).sum(1)
        )
        idx = dist.argmin(dim=1)
        onehot = F.one_hot(idx, num_classes=self.n_codes).type(flat.dtype)
        quantized = (onehot @ self.embed).view_as(x)

        if self.training:
            with torch.no_grad():
                self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                embed_sum = onehot.t() @ flat
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cluster_size = (self.cluster_size + self.eps) / (n + self.n_codes * self.eps) * n
                self.embed.copy_(self.embed_avg / cluster_size.unsqueeze(1))
                self._revive_dead_codes(flat)

        # Commitment loss + straight-through estimator.
        commit_loss = self.commitment * F.mse_loss(x, quantized.detach())
        quantized_st = x + (quantized - x).detach()
        return quantized_st, idx.view(x.shape[:-1]), commit_loss

    @torch.no_grad()
    def codebook_usage(self) -> float:
        return float((self.cluster_size > self.dead_code_threshold).float().mean().item())


class ResidualVQ(nn.Module):
    """Stack of N quantizers operating on residuals."""
    def __init__(self, n_quantizers: int, n_codes: int, code_dim: int, **kw):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantizerEMA(n_codes, code_dim, **kw) for _ in range(n_quantizers)])

    def forward(self, x: torch.Tensor):
        residual = x
        out = torch.zeros_like(x)
        idx_list = []
        loss = 0.0
        for q in self.layers:
            q_x, idx, l = q(residual)
            out = out + q_x
            residual = residual - q_x.detach()  # detach: each quantizer sees a fixed target.
            idx_list.append(idx)
            loss = loss + l
        return out, torch.stack(idx_list, dim=-1), loss

    @torch.no_grad()
    def codebook_usage(self) -> List[float]:
        return [q.codebook_usage() for q in self.layers]


class TabularVQVAE(nn.Module):
    def __init__(
        self,
        schema: FeatureSchema,
        d_token: int = 64,
        n_heads: int = 4,
        n_blocks: int = 2,
        dropout: float = 0.1,
        n_codes: int = 512,
        code_dim: int = 32,
        n_quantizers: int = 2,
        commitment: float = 0.25,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 1e-2,
        decoder_hidden: int = 256,
    ):
        super().__init__()
        self.schema = schema
        self.encoder = TabularTransformerEncoder(schema, d_token, n_heads, n_blocks, dropout)
        self.pre_vq = nn.Linear(d_token, code_dim)
        self.vq = ResidualVQ(
            n_quantizers=n_quantizers,
            n_codes=n_codes,
            code_dim=code_dim,
            decay=ema_decay,
            commitment=commitment,
            dead_code_threshold=dead_code_threshold,
        )
        self.post_vq = nn.Linear(code_dim, decoder_hidden)
        self.decoder = TabularDecoderHeads(schema, latent_dim=decoder_hidden, hidden=decoder_hidden)
        self.code_dim = code_dim
        self.n_codes = n_codes
        self.n_quantizers = n_quantizers

    def encode(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        h = self.encoder(num, cat)
        z_e = self.pre_vq(h)
        return z_e

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> dict:
        z_e = self.encode(num, cat)
        z_q, idx, vq_loss = self.vq(z_e)
        h = self.post_vq(z_q)
        decoded = self.decoder(h)
        return {"decoded": decoded, "z_e": z_e, "z_q": z_q, "idx": idx, "vq_loss": vq_loss, "h": h}

    def loss(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        out = self.forward(batch["num"], batch["cat"])
        recon, info = self.decoder.recon_loss(out["decoded"], batch["num"], batch["cat"])
        loss = recon + out["vq_loss"]
        info.update({
            "vq_loss": out["vq_loss"].detach(),
            "loss": loss.detach(),
            "codebook_usage": torch.tensor(self.vq.codebook_usage()),
        })
        return loss, info

    @torch.no_grad()
    def latent(self, num, cat) -> torch.Tensor:
        out = self.forward(num, cat)
        return out["h"]
