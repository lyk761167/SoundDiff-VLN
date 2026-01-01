# ss_baselines/av_nav/models/audio_latent_diffusion.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int):
    """
    Create sinusoidal embeddings for diffusion timesteps.
    
    Args:
        timesteps: Tensor of shape [B] containing timestep indices
        dim: Dimension of the embedding
    
    Returns:
        Tensor of shape [B, dim] containing sinusoidal embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None]  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb  # [B, dim]


class CondFiLM(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) layer for conditional input.
    
    Args:
        cond_dim: Dimension of the conditional input
        hidden_dim: Dimension of the hidden features
    """
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

    def forward(self, h: torch.Tensor, cond: torch.Tensor):
        """
        Apply FiLM modulation to the input features.
        
        Args:
            h: Features tensor of shape [B, H] or [B, T, H]
            cond: Conditional tensor of shape [B, cond_dim]
        
        Returns:
            Modulated features of same shape as h
        """
        scale_shift = self.net(cond)  # [B, 2H]
        scale, shift = scale_shift.chunk(2, dim=-1)
        if h.dim() == 3:
            scale = scale[:, None, :]
            shift = shift[:, None, :]
        return h * (1 + scale) + shift


class AudioLatentDiffusionHead(nn.Module):
    """
    Latent-space DDPM: Predicts noise (epsilon) as a denoising prior.
    
    Args:
        latent_dim: Dimension of the latent space
        cond_dim: Dimension of the conditional input
        hidden_dim: Hidden dimension of the network (default: 512)
        timesteps: Number of diffusion timesteps (default: 200)
        beta_start: Starting beta value for the noise schedule (default: 1e-4)
        beta_end: Ending beta value for the noise schedule (default: 2e-2)
    """
    def __init__(self, latent_dim: int, cond_dim: int, hidden_dim: int = 512,
                 timesteps: int = 200, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.timesteps = timesteps

        # Precompute noise schedule parameters
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # Network architecture
        time_dim = 256
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.film = CondFiLM(cond_dim + hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        """
        Forward diffusion process: Add noise to clean latents at timestep t.
        
        Args:
            z0: Clean latents of shape [B, ...]
            t: Timestep indices of shape [B]
            eps: Random noise of same shape as z0
        
        Returns:
            Noisy latents at timestep t
        """
        a = self.alphas_cumprod[t]  # [B]
        while a.dim() < z0.dim():
            a = a.unsqueeze(-1)
        return torch.sqrt(a) * z0 + torch.sqrt(1 - a) * eps

    def diffusion_loss(self, z_clean: torch.Tensor, cond: torch.Tensor):
        """
        Compute the diffusion loss (MSE between predicted and true noise).
        
        Args:
            z_clean: Clean latents of shape [B, latent_dim]
            cond: Conditional input of shape [B, cond_dim]
        
        Returns:
            MSE loss between predicted and true noise
        """
        B = z_clean.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=z_clean.device, dtype=torch.long)
        eps = torch.randn_like(z_clean)
        z_t = self.q_sample(z_clean, t, eps)

        # Predict noise
        h = F.silu(self.in_proj(z_t))
        t_emb = sinusoidal_timestep_embedding(t, 256)
        t_h = self.time_mlp(t_emb)
        cond_all = torch.cat([cond, t_h], dim=-1)

        h = self.film(h, cond_all)
        h = F.silu(h)
        eps_hat = self.out_proj(h)
        return F.mse_loss(eps_hat, eps)

    @torch.no_grad()
    def one_step_denoise(self, z: torch.Tensor, cond: torch.Tensor):
        """
        Approximate one-step denoising: Predict clean latents from noisy input.
        
        Note: This is a simplified version that assumes t = timesteps - 1.
        
        Args:
            z: Noisy latents of shape [B, latent_dim]
            cond: Conditional input of shape [B, cond_dim]
        
        Returns:
            Predicted clean latents
        """
        B = z.shape[0]
        t = torch.full((B,), self.timesteps - 1, device=z.device, dtype=torch.long)

        h = F.silu(self.in_proj(z))
        t_emb = sinusoidal_timestep_embedding(t, 256)
        t_h = self.time_mlp(t_emb)
        cond_all = torch.cat([cond, t_h], dim=-1)

        h = self.film(h, cond_all)
        h = F.silu(h)
        eps_hat = self.out_proj(h)

        # Approximate z0 from predicted noise
        a = self.alphas_cumprod[t].view(-1, *([1] * (z.dim() - 1)))
        z0_hat = (z - torch.sqrt(1 - a) * eps_hat) / torch.sqrt(a).clamp_min(1e-6)
        return z0_hat