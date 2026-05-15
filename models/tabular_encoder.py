"""Tabular encoder/decoder used as the front-/back-end of both VAEs.

We follow the FT-Transformer recipe (Gorishniy et al., 2021):

  * every numerical feature is projected to a ``d_token``-dim vector,
  * every categorical feature has its own embedding table,
  * a learnable ``[CLS]`` token is prepended,
  * a small transformer encoder mixes the tokens,
  * the ``[CLS]`` representation is read out as the row embedding.

For decoding we mirror the structure: a row embedding is broadcast to per-
feature tokens, then per-feature heads emit either a Gaussian (numericals) or
a categorical distribution (categoricals).
"""
from __future__ import annotations
from typing import List, Tuple, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import FeatureSchema


class TabularTokenizer(nn.Module):
    """Maps a tabular row to a sequence of feature tokens (+ CLS)."""
    def __init__(self, schema: FeatureSchema, d_token: int):
        super().__init__()
        self.schema = schema
        self.d_token = d_token
        # Numerical: per-feature affine to d_token.
        if schema.n_num > 0:
            self.num_weight = nn.Parameter(torch.randn(schema.n_num, d_token) * 0.02)
            self.num_bias = nn.Parameter(torch.zeros(schema.n_num, d_token))
        # Categorical: one embedding table per feature (each its own cardinality).
        self.cat_embeds = nn.ModuleList(
            [nn.Embedding(schema.cat_cardinalities[c], d_token) for c in schema.categoricals]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            num: (B, n_num) float
            cat: (B, n_cat) long
        Returns:
            tokens: (B, n_num + n_cat + 1, d_token)
        """
        B = num.shape[0] if self.schema.n_num > 0 else cat.shape[0]
        toks = []
        if self.schema.n_num > 0:
            # (B, n_num, 1) * (n_num, d_token) -> (B, n_num, d_token)
            num_tok = num.unsqueeze(-1) * self.num_weight + self.num_bias
            toks.append(num_tok)
        for i, emb in enumerate(self.cat_embeds):
            toks.append(emb(cat[:, i]).unsqueeze(1))
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls] + toks, dim=1) if toks else cls


class TabularTransformerEncoder(nn.Module):
    """Tokenizer + a tiny transformer trunk -> CLS embedding."""
    def __init__(self, schema: FeatureSchema, d_token: int, n_heads: int, n_blocks: int, dropout: float):
        super().__init__()
        self.tokenizer = TabularTokenizer(schema, d_token)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=4 * d_token,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.norm = nn.LayerNorm(d_token)

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        toks = self.tokenizer(num, cat)        # (B, F+1, D)
        toks = self.tf(toks)
        return self.norm(toks[:, 0])           # CLS embedding (B, D)


class TabularDecoderHeads(nn.Module):
    """Decoder: row embedding -> per-feature output distributions.

    Numerical heads predict a Gaussian (mean, log_std); categorical heads emit
    class logits.  ``recon_loss`` computes a per-feature averaged negative log-
    likelihood (heteroscedastic for numericals, cross-entropy for categoricals).
    """
    def __init__(self, schema: FeatureSchema, latent_dim: int, hidden: int = 256):
        super().__init__()
        self.schema = schema
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        # Numerical: 2 outputs per feature (mean, log_std).
        if schema.n_num > 0:
            self.num_head = nn.Linear(hidden, 2 * schema.n_num)
        # One head per categorical feature.
        self.cat_heads = nn.ModuleList(
            [nn.Linear(hidden, schema.cat_cardinalities[c]) for c in schema.categoricals]
        )

    def forward(self, z: torch.Tensor):
        h = self.trunk(z)
        out: Dict[str, torch.Tensor] = {}
        if self.schema.n_num > 0:
            num = self.num_head(h)
            mean, log_std = num.chunk(2, dim=-1)
            log_std = log_std.clamp(-5.0, 2.0)
            out["num_mean"] = mean
            out["num_log_std"] = log_std
        out["cat_logits"] = [head(h) for head in self.cat_heads]
        return out

    def recon_loss(self, out: dict, num: torch.Tensor, cat: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        losses = {}
        total = torch.zeros((), device=num.device if num.numel() else cat.device)
        if self.schema.n_num > 0:
            dist = torch.distributions.Normal(out["num_mean"], out["num_log_std"].exp())
            nll_num = -dist.log_prob(num).mean()
            losses["num_nll"] = nll_num.detach()
            total = total + nll_num
        if self.schema.n_cat > 0:
            ce_total = 0.0
            for i, logits in enumerate(out["cat_logits"]):
                ce_total = ce_total + F.cross_entropy(logits, cat[:, i])
            ce_total = ce_total / self.schema.n_cat
            losses["cat_ce"] = ce_total.detach()
            total = total + ce_total
        losses["recon"] = total.detach()
        return total, losses
