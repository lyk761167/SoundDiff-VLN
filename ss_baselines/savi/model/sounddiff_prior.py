#!/usr/bin/env python3
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _linear_beta_schedule(T: int, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)


class FiLMDenoiser(nn.Module):
    """
    Minimal denoiser: eps_theta(z_t, t, cond)
    cond: [B, C] -> project -> FiLM
    """

    def __init__(self, z_dim: int, cond_dim: int, hidden: int = 1024):
        super().__init__()
        self.z_dim = z_dim
        self.cond_dim = cond_dim

        self.t_embed = nn.Embedding(1024, hidden)  # enough for typical T
        self.cond_proj = nn.Linear(cond_dim, hidden)

        self.fc1 = nn.Linear(z_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, z_dim)

        # FiLM
        self.film1 = nn.Linear(hidden, hidden * 2)  # gamma,beta
        self.film2 = nn.Linear(hidden, hidden * 2)

        for m in [self.fc1, self.fc2, self.fc3, self.cond_proj]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        z_t:  [B, z_dim]
        t:    [B] long
        cond: [B, cond_dim]
        """
        h = self.fc1(z_t)

        te = self.t_embed(t.clamp(min=0, max=self.t_embed.num_embeddings - 1))
        ce = self.cond_proj(cond)
        ctx = torch.tanh(te + ce)  # [B, hidden]

        g1, b1 = self.film1(ctx).chunk(2, dim=-1)
        h = F.silu(h * (1.0 + g1) + b1)
        h = self.fc2(h)

        g2, b2 = self.film2(ctx).chunk(2, dim=-1)
        h = F.silu(h * (1.0 + g2) + b2)
        eps = self.fc3(h)
        return eps


class LatentDiffusionPrior(nn.Module):
    """
    DDIM sampling for latent z in R^H, conditioned on cond.
    This module is meant to be loaded from a pretrained checkpoint.
    """

    def __init__(self, z_dim: int, cond_dim: int, T: int = 1000, hidden: int = 1024):
        super().__init__()
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.T = int(T)

        betas = _linear_beta_schedule(self.T)  # [T]
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # [T]

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.denoiser = FiLMDenoiser(z_dim=z_dim, cond_dim=cond_dim, hidden=hidden)

    @torch.no_grad()
    def ddim_sample(self, cond: torch.Tensor, steps: int = 10, eta: float = 0.0) -> torch.Tensor:
        """
        cond: [B, cond_dim]
        returns: z0_hat [B, z_dim]
        """
        device = cond.device
        B = cond.size(0)
        z = torch.randn(B, self.z_dim, device=device, dtype=cond.dtype)

        # choose a uniform stride
        steps = int(steps)
        assert steps >= 1
        t_seq = torch.linspace(self.T - 1, 0, steps, device=device).long()

        for i in range(steps):
            t = t_seq[i].expand(B)  # [B]
            a_t = self.alphas_cumprod[t].view(B, 1)
            sqrt_a_t = torch.sqrt(a_t)
            sqrt_om_a_t = torch.sqrt(1.0 - a_t)

            eps = self.denoiser(z, t, cond)
            x0 = (z - sqrt_om_a_t * eps) / (sqrt_a_t + 1e-8)

            if i == steps - 1:
                z = x0
                break

            t_prev = t_seq[i + 1].expand(B)
            a_prev = self.alphas_cumprod[t_prev].view(B, 1)
            sqrt_a_prev = torch.sqrt(a_prev)
            sqrt_om_a_prev = torch.sqrt(1.0 - a_prev)

            # DDIM deterministic when eta=0
            z = sqrt_a_prev * x0 + sqrt_om_a_prev * eps

        return z

    def load_ckpt(self, ckpt_path: str, strict: bool = True):
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        self.load_state_dict(sd, strict=strict)
