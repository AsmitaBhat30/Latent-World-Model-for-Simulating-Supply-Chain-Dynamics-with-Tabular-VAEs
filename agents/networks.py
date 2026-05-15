"""Small shared building blocks for the agents."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2):
    layers = [nn.Linear(in_dim, hidden), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy over a 1-d action in [0, 1].

    We model an unconstrained Gaussian, then ``0.5 * (tanh(z) + 1)`` to map to
    [0,1].  ``log_prob`` includes the change-of-variable correction.
    """
    def __init__(self, in_dim: int, hidden: int = 256, action_dim: int = 1):
        super().__init__()
        self.trunk = mlp(in_dim, hidden, 2 * action_dim, n_layers=2)
        self.action_dim = action_dim

    def forward(self, feat):
        out = self.trunk(feat)
        mu, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-5.0, 2.0)
        return mu, log_std

    def sample(self, feat, deterministic: bool = False):
        mu, log_std = self.forward(feat)
        if deterministic:
            z = mu
        else:
            std = log_std.exp()
            z = mu + std * torch.randn_like(mu)
        squashed = 0.5 * (torch.tanh(z) + 1.0)
        # log prob with tanh + scaling correction.
        normal = torch.distributions.Normal(mu, log_std.exp())
        log_prob = normal.log_prob(z).sum(-1)
        # Change of variables: d/dz [0.5*(tanh(z)+1)] = 0.5 * (1 - tanh(z)^2)
        log_prob = log_prob - torch.log(0.5 * (1 - torch.tanh(z).pow(2)) + 1e-6).sum(-1)
        return squashed, log_prob, (mu, log_std)


class TwinQ(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = mlp(feat_dim + action_dim, hidden, 1)
        self.q2 = mlp(feat_dim + action_dim, hidden, 1)

    def forward(self, feat, action):
        x = torch.cat([feat, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ValueHead(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp(feat_dim, hidden, 1)

    def forward(self, feat):
        return self.net(feat).squeeze(-1)
