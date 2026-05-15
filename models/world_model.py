"""Recurrent State-Space Model (RSSM) for latent supply-chain dynamics.

Architecture follows DreamerV2/V3 closely:

  h_t   = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
  prior:        p(z_t | h_t)          -- categorical over 32 vars x 32 classes
                                          (or Gaussian when paired with the
                                          hierarchical VAE)
  posterior:    q(z_t | h_t, x_t)     -- where x_t is the VAE-derived embedding

Output heads:
  decoder: x̂_t  (passed back to the tabular decoder)
  reward:  symlog two-hot distribution over a fixed support
  cont:    Bernoulli for continuation γ_t

Training loss = recon + KL(post || prior, balanced + free-bits) + reward_NLL + cont_BCE.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import symlog, two_hot_encode, two_hot_decode


class _CategoricalRSSMHead(nn.Module):
    """Returns logits for n_cats categorical variables with n_classes each.
    Sampling uses straight-through gumbel-softmax."""
    def __init__(self, in_dim: int, n_cats: int, n_classes: int):
        super().__init__()
        self.n_cats = n_cats
        self.n_classes = n_classes
        self.fc = nn.Linear(in_dim, n_cats * n_classes)

    def forward(self, h: torch.Tensor):
        logits = self.fc(h).view(*h.shape[:-1], self.n_cats, self.n_classes)
        return logits

    def sample(self, logits, temperature: float = 1.0):
        # Straight-through one-hot sample.
        y_soft = F.softmax(logits / temperature, dim=-1)
        idx = torch.distributions.Categorical(logits=logits).sample()
        y_hard = F.one_hot(idx, num_classes=self.n_classes).type(logits.dtype)
        z = y_hard + (y_soft - y_soft.detach())   # straight-through
        return z, idx


class RSSM(nn.Module):
    """Recurrent State-Space Model with categorical latents (Dreamer-V3 default)."""
    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        deterministic_dim: int = 256,
        n_cats: int = 32,
        n_classes: int = 32,
        hidden: int = 256,
    ):
        super().__init__()
        self.deterministic_dim = deterministic_dim
        self.n_cats = n_cats
        self.n_classes = n_classes
        self.stoch_dim = n_cats * n_classes

        self.act_proj = nn.Sequential(
            nn.Linear(action_dim + self.stoch_dim, hidden), nn.GELU(),
        )
        self.cell = nn.GRUCell(hidden, deterministic_dim)

        self.prior_head = _CategoricalRSSMHead(deterministic_dim, n_cats, n_classes)
        self.post_head = _CategoricalRSSMHead(deterministic_dim + embed_dim, n_cats, n_classes)

    @property
    def state_dim(self) -> int:
        return self.deterministic_dim + self.stoch_dim

    def initial(self, batch_size: int, device) -> dict:
        return dict(
            h=torch.zeros(batch_size, self.deterministic_dim, device=device),
            z=torch.zeros(batch_size, self.n_cats, self.n_classes, device=device),
        )

    def step(self, state: dict, action: torch.Tensor, embed: Optional[torch.Tensor]):
        """One RSSM transition.
        If ``embed`` is given, returns posterior; otherwise returns prior only."""
        z_flat = state["z"].flatten(-2)
        x = torch.cat([z_flat, action], dim=-1)
        x = self.act_proj(x)
        h = self.cell(x, state["h"])

        prior_logits = self.prior_head(h)
        if embed is not None:
            post_logits = self.post_head(torch.cat([h, embed], dim=-1))
            z, _ = self.post_head.sample(post_logits)
        else:
            post_logits = None
            z, _ = self.prior_head.sample(prior_logits)

        new_state = dict(h=h, z=z)
        return new_state, prior_logits, post_logits

    def observe(self, embeds: torch.Tensor, actions: torch.Tensor):
        """Run the RSSM over a sequence with the posterior.
        embeds, actions: (B, T, *)"""
        B, T, _ = embeds.shape
        state = self.initial(B, embeds.device)
        priors, posts, hs, zs = [], [], [], []
        for t in range(T):
            state, prior_logits, post_logits = self.step(state, actions[:, t], embeds[:, t])
            priors.append(prior_logits)
            posts.append(post_logits)
            hs.append(state["h"])
            zs.append(state["z"])
        return dict(
            prior_logits=torch.stack(priors, 1),
            post_logits=torch.stack(posts, 1),
            h=torch.stack(hs, 1),
            z=torch.stack(zs, 1),
        )

    def imagine(self, state: dict, actor, horizon: int):
        """Imagined latent rollout using the prior; ``actor`` is a callable
        ``state -> (action, extra)`` that produces actions on the fly."""
        states = []
        actions = []
        extras = []
        cur = state
        for _ in range(horizon):
            feat = torch.cat([cur["h"], cur["z"].flatten(-2)], dim=-1)
            action, extra = actor(feat)
            cur, prior_logits, _ = self.step(cur, action, embed=None)
            states.append(cur)
            actions.append(action)
            extras.append(extra)
        return states, actions, extras


# ---------------------------------------------------------------------------- #
# Reward / continuation heads (Dreamer-V3 style)
# ---------------------------------------------------------------------------- #

class _SymlogTwoHotHead(nn.Module):
    def __init__(self, in_dim: int, n_bins: int = 41, low: float = -20.0, high: float = 20.0, hidden: int = 256):
        super().__init__()
        self.n_bins = n_bins
        bins = torch.linspace(low, high, n_bins)
        self.register_buffer("bins", bins)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_bins),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)

    def nll(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self.forward(feat)
        target_sym = symlog(target)
        soft_target = two_hot_encode(target_sym, self.bins)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_target * log_probs).sum(-1).mean()

    @torch.no_grad()
    def predict(self, feat: torch.Tensor) -> torch.Tensor:
        logits = self.forward(feat)
        probs = F.softmax(logits, dim=-1)
        sym = two_hot_decode(probs, self.bins)
        # Inverse symlog.
        return torch.sign(sym) * torch.expm1(sym.abs())


class _ContinueHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat):
        return self.net(feat).squeeze(-1)


# ---------------------------------------------------------------------------- #
# Full latent world model
# ---------------------------------------------------------------------------- #

class LatentWorldModel(nn.Module):
    """Wraps a tabular VAE encoder + RSSM dynamics + decoder + reward/cont heads.

    The VAE is used purely as a *feature extractor* during world-model training
    (the VAE itself is pre-trained and either frozen or fine-tuned slowly).
    """
    def __init__(
        self,
        vae,                     # HierarchicalTabularVAE or TabularVQVAE
        rssm: RSSM,
        latent_dim: int,
        action_dim: int,
        reward_bins: int = 41,
    ):
        super().__init__()
        self.vae = vae
        self.rssm = rssm
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.feat_dim = rssm.state_dim
        # Decoder back to the VAE latent (so we can pipe through the existing tabular heads).
        self.feat_to_latent = nn.Linear(self.feat_dim, latent_dim)
        self.reward_head = _SymlogTwoHotHead(self.feat_dim, n_bins=reward_bins)
        self.cont_head = _ContinueHead(self.feat_dim)

    # ----------------------------- embeddings ----------------------------- #
    def embed(self, num, cat):
        # Use the VAE encoder's CLS embedding -- richer than the bottleneck.
        if hasattr(self.vae, "encoder"):
            return self.vae.encoder(num, cat)
        # Hierarchical VAE
        return self.vae.bu_encoder(num, cat)

    def features(self, state: dict) -> torch.Tensor:
        return torch.cat([state["h"], state["z"].flatten(-2)], dim=-1)

    # ----------------------------- training step ----------------------------- #
    def loss(self, batch: dict, kl_balance: float = 0.8, free_bits: float = 1.0) -> Tuple[torch.Tensor, dict]:
        num, cat = batch["num"], batch["cat"]
        B, T = num.shape[:2]
        actions = batch["action"]                     # (B, T, A)
        rewards = batch["reward"]                     # (B, T)
        dones = batch["done"]                         # (B, T)

        # Embed each step with the VAE.
        flat_num = num.reshape(B * T, *num.shape[2:])
        flat_cat = cat.reshape(B * T, *cat.shape[2:])
        flat_embed = self.embed(flat_num, flat_cat)
        embeds = flat_embed.view(B, T, -1)

        out = self.rssm.observe(embeds, actions)
        feats = torch.cat([out["h"], out["z"].flatten(-2)], dim=-1)   # (B, T, F)

        # ---- recon: feed feats back through the VAE decoder ----
        z_for_dec = self.feat_to_latent(feats)
        decoded = self.vae.decoder(z_for_dec.reshape(B * T, -1))
        recon_loss, recon_info = self.vae.decoder.recon_loss(
            decoded, flat_num, flat_cat
        )

        # ---- KL between post & prior with balancing and free-bits ----
        post = torch.distributions.Categorical(logits=out["post_logits"])
        prior = torch.distributions.Categorical(logits=out["prior_logits"])
        post_sg = torch.distributions.Categorical(logits=out["post_logits"].detach())
        prior_sg = torch.distributions.Categorical(logits=out["prior_logits"].detach())
        kl_lhs = torch.distributions.kl_divergence(post_sg, prior).sum(-1)
        kl_rhs = torch.distributions.kl_divergence(post, prior_sg).sum(-1)
        kl = kl_balance * kl_lhs + (1 - kl_balance) * kl_rhs
        kl = torch.clamp(kl.mean(), min=free_bits)

        # ---- reward & continuation ----
        reward_loss = self.reward_head.nll(feats.reshape(B * T, -1), rewards.reshape(-1))
        cont_logits = self.cont_head(feats.reshape(B * T, -1))
        cont_target = 1.0 - dones.reshape(-1)
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

        total = recon_loss + kl + reward_loss + cont_loss
        info = {
            "wm/recon": recon_loss.detach(),
            "wm/kl": kl.detach(),
            "wm/reward": reward_loss.detach(),
            "wm/cont": cont_loss.detach(),
            "wm/total": total.detach(),
        }
        info.update({f"vae/{k}": v for k, v in recon_info.items()})
        return total, info

    # ----------------------------- imagination ----------------------------- #
    @torch.no_grad()
    def imagine(self, num0, cat0, action0, actor, horizon: int = 15) -> dict:
        """Imagine a trajectory starting from a real observation, using ``actor``."""
        embed = self.embed(num0, cat0)
        state = self.rssm.initial(embed.shape[0], embed.device)
        state, _, _ = self.rssm.step(state, action0, embed=embed)

        feats, actions, rewards, conts = [], [], [], []
        cur = state
        for _ in range(horizon):
            feat = torch.cat([cur["h"], cur["z"].flatten(-2)], dim=-1)
            action, _ = actor(feat)
            cur, prior_logits, _ = self.rssm.step(cur, action, embed=None)
            feats.append(feat)
            actions.append(action)
            rewards.append(self.reward_head.predict(feat))
            conts.append(torch.sigmoid(self.cont_head(feat)))
        feats = torch.stack(feats, 1)
        return {
            "feats": feats,
            "actions": torch.stack(actions, 1),
            "rewards": torch.stack(rewards, 1),
            "continues": torch.stack(conts, 1),
            "decoded": self.vae.decoder(self.feat_to_latent(feats).reshape(-1, self.latent_dim)),
        }
