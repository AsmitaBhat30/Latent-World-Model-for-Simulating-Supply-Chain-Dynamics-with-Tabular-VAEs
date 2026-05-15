"""Dreamer-V3 actor-critic trained on imagined latent rollouts.

Key implementation details (Hafner et al., 2023):
 * λ-returns over imagined trajectories with discount γ * γ̂ (continuation pred).
 * Symlog targets for the critic.
 * Percentile-based return normalization to make actor gradients scale-free.
 * EMA target critic.
 * Entropy bonus that decays with return magnitude.
"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.world_model import LatentWorldModel
from models.utils import symlog, two_hot_encode, two_hot_decode
from .networks import GaussianActor, ValueHead


class _SymlogTwoHotCritic(nn.Module):
    def __init__(self, in_dim: int, n_bins: int = 41, low: float = -20.0, high: float = 20.0, hidden: int = 256):
        super().__init__()
        self.n_bins = n_bins
        self.register_buffer("bins", torch.linspace(low, high, n_bins))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_bins),
        )

    def forward(self, feat):
        return self.net(feat)

    def value(self, feat):
        probs = F.softmax(self.forward(feat), dim=-1)
        sym = (probs * self.bins).sum(-1)
        return torch.sign(sym) * torch.expm1(sym.abs())

    def nll(self, feat, target):
        target_sym = symlog(target)
        soft = two_hot_encode(target_sym, self.bins)
        logp = F.log_softmax(self.forward(feat), dim=-1)
        return -(soft * logp).sum(-1).mean()


def lambda_returns(rewards, values, continues, lam: float, gamma: float):
    """Bootstrap λ-returns. rewards, values, continues: (B, H)."""
    B, H = rewards.shape
    returns = torch.zeros_like(rewards)
    last = values[:, -1]
    for t in reversed(range(H)):
        v_next = last if t == H - 1 else returns[:, t + 1]
        returns[:, t] = rewards[:, t] + gamma * continues[:, t] * ((1 - lam) * values[:, t] + lam * v_next)
    return returns


class DreamerV3Agent(nn.Module):
    def __init__(self, world_model: LatentWorldModel, cfg: dict):
        super().__init__()
        self.wm = world_model
        self.cfg = cfg
        feat = world_model.feat_dim
        self.actor = GaussianActor(feat, hidden=256, action_dim=world_model.action_dim)
        self.critic = _SymlogTwoHotCritic(feat)
        self.target_critic = _SymlogTwoHotCritic(feat)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=cfg["actor_lr"], eps=1e-5)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=cfg["critic_lr"], eps=1e-5)
        # Percentiles for return normalization.
        self.register_buffer("ret_lo", torch.tensor(0.0))
        self.register_buffer("ret_hi", torch.tensor(1.0))

    def _actor_fn(self, feat):
        action, log_prob, _ = self.actor.sample(feat)
        return action, dict(log_prob=log_prob)

    def update(self, init_states: dict) -> dict:
        """Update actor & critic from imagined rollouts starting at ``init_states``.
        init_states: dict with keys 'h' (B,Dh) and 'z' (B,K,C)."""
        cfg = self.cfg
        H = cfg["horizon"]
        gamma = cfg["gamma"]
        lam = cfg["lambda_"]
        ent_w = cfg["entropy"]

        # Detach the starting states (we don't backprop through the world model here).
        state = {k: v.detach() for k, v in init_states.items()}
        feats, actions, log_probs = [], [], []
        rewards, continues = [], []
        cur = state
        for _ in range(H):
            feat = torch.cat([cur["h"], cur["z"].flatten(-2)], dim=-1)
            action, log_prob, _ = self.actor.sample(feat)
            cur, prior_logits, _ = self.wm.rssm.step(cur, action, embed=None)
            feats.append(feat)
            actions.append(action)
            log_probs.append(log_prob)
            with torch.no_grad():
                rewards.append(self.wm.reward_head.predict(feat))
                continues.append(torch.sigmoid(self.wm.cont_head(feat)))

        feats = torch.stack(feats, 1)
        actions = torch.stack(actions, 1)
        log_probs = torch.stack(log_probs, 1)
        rewards = torch.stack(rewards, 1)
        continues = torch.stack(continues, 1)

        with torch.no_grad():
            target_values = self.target_critic.value(feats)
        returns = lambda_returns(rewards, target_values, continues, lam, gamma)

        # Percentile-based return normalization.
        with torch.no_grad():
            r05 = torch.quantile(returns.detach(), 0.05)
            r95 = torch.quantile(returns.detach(), 0.95)
            self.ret_lo.mul_(0.99).add_(0.01 * r05)
            self.ret_hi.mul_(0.99).add_(0.01 * r95)
            scale = torch.clamp(self.ret_hi - self.ret_lo, min=1.0)
        adv = (returns - target_values).detach() / scale

        # Actor: reinforce on advantage + entropy bonus.
        entropy = -log_probs.mean()
        actor_loss = -(log_probs * adv).mean() - ent_w * entropy

        self.actor_opt.zero_grad()
        actor_loss.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_opt.step()

        # Critic: regress to symlog λ-returns.
        critic_loss = self.critic.nll(feats.detach().reshape(-1, feats.shape[-1]), returns.reshape(-1).detach())
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_opt.step()

        # EMA target critic.
        tau = self.cfg["target_update"]
        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.mul_(1 - tau).add_(tau * p)

        return dict(
            actor_loss=actor_loss.detach().item(),
            critic_loss=critic_loss.detach().item(),
            entropy=entropy.detach().item(),
            return_mean=returns.mean().detach().item(),
        )

    @torch.no_grad()
    def act(self, feat, deterministic: bool = False):
        action, _, _ = self.actor.sample(feat, deterministic=deterministic)
        return action
