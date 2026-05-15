"""Proximal Policy Optimization on imagined latent rollouts.

A standard PPO-clip implementation with GAE-λ; the only twist is that the
``env`` we step is the world model's RSSM (in pure latent space), and rewards
& continuations come from their respective heads.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.world_model import LatentWorldModel
from .networks import GaussianActor, ValueHead


class PPOAgent(nn.Module):
    def __init__(self, world_model: LatentWorldModel, cfg: dict):
        super().__init__()
        self.wm = world_model
        self.cfg = cfg
        feat = world_model.feat_dim
        adim = world_model.action_dim
        self.actor = GaussianActor(feat, hidden=256, action_dim=adim)
        self.critic = ValueHead(feat, hidden=256)
        self.opt = torch.optim.AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=3e-4, eps=1e-5,
        )

    def _imagine_batch(self, init_states: dict, horizon: int):
        cur = {k: v.detach() for k, v in init_states.items()}
        feats, actions, logps, rewards, conts = [], [], [], [], []
        with torch.no_grad():
            for _ in range(horizon):
                feat = torch.cat([cur["h"], cur["z"].flatten(-2)], dim=-1)
                action, logp, _ = self.actor.sample(feat)
                cur, _, _ = self.wm.rssm.step(cur, action, embed=None)
                feats.append(feat)
                actions.append(action)
                logps.append(logp)
                rewards.append(self.wm.reward_head.predict(feat))
                conts.append(torch.sigmoid(self.wm.cont_head(feat)))
        feats = torch.stack(feats, 1)
        actions = torch.stack(actions, 1)
        logps = torch.stack(logps, 1)
        rewards = torch.stack(rewards, 1)
        conts = torch.stack(conts, 1)
        return feats, actions, logps, rewards, conts

    def update(self, init_states: dict, horizon: int = 32) -> dict:
        cfg = self.cfg
        feats, actions, logp_old, rewards, conts = self._imagine_batch(init_states, horizon)
        B, H, _ = feats.shape

        # Bootstrap value.
        with torch.no_grad():
            values = self.critic(feats)
            last_val = values[:, -1]
            # GAE.
            advantages = torch.zeros_like(rewards)
            gae = 0
            for t in reversed(range(H)):
                v_next = last_val if t == H - 1 else values[:, t + 1]
                delta = rewards[:, t] + cfg["gamma"] * conts[:, t] * v_next - values[:, t]
                gae = delta + cfg["gamma"] * cfg["lambda_"] * conts[:, t] * gae
                advantages[:, t] = gae
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # Multiple epochs of mini-batch updates.
        flat_feats = feats.reshape(B * H, -1)
        flat_actions = actions.reshape(B * H, -1)
        flat_logp_old = logp_old.reshape(B * H)
        flat_adv = advantages.reshape(B * H)
        flat_ret = returns.reshape(B * H)

        idx = torch.randperm(B * H, device=feats.device)
        mb = cfg["minibatch"]
        info = {}
        for _ in range(cfg["epochs"]):
            for i in range(0, B * H, mb):
                sel = idx[i:i + mb]
                f = flat_feats[sel]
                a = flat_actions[sel]
                _, logp_new, _ = self.actor.sample(f)
                ratio = torch.exp(logp_new - flat_logp_old[sel])
                surr1 = ratio * flat_adv[sel]
                surr2 = torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * flat_adv[sel]
                actor_loss = -torch.min(surr1, surr2).mean()
                v_pred = self.critic(f)
                critic_loss = F.mse_loss(v_pred, flat_ret[sel])
                entropy = -logp_new.mean()
                loss = actor_loss + 0.5 * critic_loss - cfg["entropy"] * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
                self.opt.step()
                info = dict(
                    actor_loss=float(actor_loss.detach()),
                    critic_loss=float(critic_loss.detach()),
                    entropy=float(entropy.detach()),
                )
        return info

    @torch.no_grad()
    def act(self, feat, deterministic: bool = False):
        action, _, _ = self.actor.sample(feat, deterministic=deterministic)
        return action
