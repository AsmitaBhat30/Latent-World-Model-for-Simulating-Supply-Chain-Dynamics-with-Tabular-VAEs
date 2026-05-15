"""Soft Actor-Critic on imagined latent rollouts.

We treat the world model as the environment: the agent samples (feat, action,
reward, next_feat) transitions from imagined rollouts and stores them in an
off-policy buffer.  Twin Q with target entropy auto-tuning.
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

from models.world_model import LatentWorldModel
from .networks import GaussianActor, TwinQ


class _ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.buf = deque(maxlen=capacity)

    def add(self, feat, action, reward, next_feat, done):
        # Detach to CPU to avoid graph leaks.
        self.buf.append((feat.detach().cpu(), action.detach().cpu(), float(reward),
                         next_feat.detach().cpu(), float(done)))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        feat, action, reward, next_feat, done = zip(*batch)
        return (
            torch.stack(feat),
            torch.stack(action),
            torch.tensor(reward, dtype=torch.float32),
            torch.stack(next_feat),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


class SACAgent(nn.Module):
    def __init__(self, world_model: LatentWorldModel, cfg: dict):
        super().__init__()
        self.wm = world_model
        self.cfg = cfg
        feat = world_model.feat_dim
        adim = world_model.action_dim
        self.actor = GaussianActor(feat, hidden=256, action_dim=adim)
        self.critic = TwinQ(feat, adim, hidden=256)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=cfg["actor_lr"])
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=cfg["critic_lr"])
        # Entropy temperature (log alpha) auto-tuned.
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.alpha_opt = torch.optim.AdamW([self.log_alpha], lr=cfg["alpha_lr"])
        self.target_entropy = cfg["target_entropy"]
        self.buffer = _ReplayBuffer()

    def alpha(self):
        return self.log_alpha.exp()

    def collect(self, init_states: dict, horizon: int = 15):
        """Collect a batch of imagined transitions into the replay buffer."""
        with torch.no_grad():
            cur = {k: v.detach() for k, v in init_states.items()}
            for _ in range(horizon):
                feat = torch.cat([cur["h"], cur["z"].flatten(-2)], dim=-1)
                action, _, _ = self.actor.sample(feat)
                nxt, _, _ = self.wm.rssm.step(cur, action, embed=None)
                feat_n = torch.cat([nxt["h"], nxt["z"].flatten(-2)], dim=-1)
                reward = self.wm.reward_head.predict(feat)
                cont = torch.sigmoid(self.wm.cont_head(feat))
                done = 1.0 - cont
                for i in range(feat.shape[0]):
                    self.buffer.add(feat[i], action[i], float(reward[i]), feat_n[i], float(done[i]))
                cur = nxt

    def update(self, batch_size: int = 256) -> dict:
        if len(self.buffer) < batch_size:
            return {}
        feat, action, reward, next_feat, done = self.buffer.sample(batch_size)
        feat = feat.to(self.log_alpha.device)
        action = action.to(self.log_alpha.device)
        reward = reward.to(self.log_alpha.device)
        next_feat = next_feat.to(self.log_alpha.device)
        done = done.to(self.log_alpha.device)
        gamma = self.cfg["gamma"]

        # Critic update.
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_feat)
            q1_t, q2_t = self.critic_target(next_feat, next_action)
            q_t = torch.min(q1_t, q2_t) - self.alpha() * next_logp
            target = reward + gamma * (1 - done) * q_t
        q1, q2 = self.critic(feat, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor + alpha update.
        new_action, new_logp, _ = self.actor.sample(feat)
        q1_n, q2_n = self.critic(feat, new_action)
        q_n = torch.min(q1_n, q2_n)
        actor_loss = (self.alpha().detach() * new_logp - q_n).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (new_logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft target update.
        tau = self.cfg["tau"]
        with torch.no_grad():
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.mul_(1 - tau).add_(tau * p)

        return dict(
            actor_loss=float(actor_loss.detach()),
            critic_loss=float(critic_loss.detach()),
            alpha=float(self.alpha().detach()),
            entropy=float(-new_logp.mean().detach()),
        )

    @torch.no_grad()
    def act(self, feat, deterministic: bool = False):
        action, _, _ = self.actor.sample(feat, deterministic=deterministic)
        return action
