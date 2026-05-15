"""Common math utilities: symlog/symexp and two-hot encoding (Dreamer-V3)."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


def two_hot_encode(values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Two-hot encode a batch of scalar values onto a fixed bin support.

    Args:
        values: (..., ) tensor of (already-symlogged) values.
        bins:   (B,) monotonic bin centres.
    Returns:
        (..., B) tensor that sums to 1 over the last dim.
    """
    *batch, = values.shape
    B = bins.shape[0]
    v = values.clamp(bins[0], bins[-1])
    # Find upper bin index for each value.
    idx = torch.bucketize(v, bins) - 1
    idx = idx.clamp(0, B - 2)
    lower = bins[idx]
    upper = bins[idx + 1]
    weight_upper = ((v - lower) / (upper - lower + 1e-8)).clamp(0, 1)
    weight_lower = 1.0 - weight_upper
    out = torch.zeros(*batch, B, device=values.device, dtype=values.dtype)
    out.scatter_(-1, idx.unsqueeze(-1), weight_lower.unsqueeze(-1))
    out.scatter_(-1, (idx + 1).unsqueeze(-1), weight_upper.unsqueeze(-1))
    return out


def two_hot_decode(probs: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Expected value under a categorical distribution over ``bins``."""
    return (probs * bins).sum(-1)


def kl_balance(prior_logits: torch.Tensor, post_logits: torch.Tensor, balance: float, free_bits: float) -> torch.Tensor:
    """KL-balancing trick from Dreamer-V2: prior is trained faster than the posterior.

    KL = balance * KL(stop_grad(post) || prior) + (1 - balance) * KL(post || stop_grad(prior))
    With free-bits to prevent posterior collapse.
    """
    post = torch.distributions.Categorical(logits=post_logits)
    prior = torch.distributions.Categorical(logits=prior_logits)
    post_sg = torch.distributions.Categorical(logits=post_logits.detach())
    prior_sg = torch.distributions.Categorical(logits=prior_logits.detach())
    kl_lhs = torch.distributions.kl_divergence(post_sg, prior).sum(-1)
    kl_rhs = torch.distributions.kl_divergence(post, prior_sg).sum(-1)
    kl = balance * kl_lhs + (1 - balance) * kl_rhs
    kl = torch.clamp(kl, min=free_bits)
    return kl
