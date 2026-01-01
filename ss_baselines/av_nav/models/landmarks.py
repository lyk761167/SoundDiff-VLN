# ss_baselines/av_nav/models/landmarks.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LandmarkTokenizer(nn.Module):
    """
    Compress a single embedding [B, D] into K landmark tokens [B, K, D].
    """
    def __init__(self, in_dim: int, token_dim: int, k: int):
        super().__init__()
        self.k = k
        self.token_dim = token_dim
        self.proj = nn.Linear(in_dim, k * token_dim)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        B = x.size(0)
        t = self.proj(x).view(B, self.k, self.token_dim)  # [B, K, D]
        t = self.norm(t)
        return t


class AVLandmarkAligner(nn.Module):
    """
    Align audio landmark tokens and visual landmark tokens.
    Produces an alignment loss (InfoNCE over batch, plus optional cosine for B==1).
    """
    def __init__(self, token_dim: int, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.a_pool = nn.LayerNorm(token_dim)
        self.v_pool = nn.LayerNorm(token_dim)

    def _global_pool(self, toks: torch.Tensor, which: str) -> torch.Tensor:
        # toks: [B, K, D] -> global [B, D]
        g = toks.mean(dim=1)  # mean pool over K
        if which == "a":
            return self.a_pool(g)
        else:
            return self.v_pool(g)

    def info_nce(self, a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Symmetric InfoNCE between audio globals and visual globals.
        a, v: [B, D]
        """
        B = a.size(0)
        logits = (a @ v.t()) / self.temperature  # [B, B]
        labels = torch.arange(B, device=a.device)
        loss_av = F.cross_entropy(logits, labels)
        loss_va = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_av + loss_va)

    def forward(self, a_toks: torch.Tensor, v_toks: torch.Tensor) -> torch.Tensor:
        """
        a_toks, v_toks: [B, K, D]
        return: scalar loss
        """
        a_g = self._global_pool(a_toks, "a")
        v_g = self._global_pool(v_toks, "v")

        B = a_g.size(0)
        if B >= 2:
            loss = self.info_nce(a_g, v_g)
        else:
            # batch=1 时 InfoNCE 没意义，用 cosine 做个弱对齐
            cos = F.cosine_similarity(a_g, v_g, dim=-1)  # [1]
            loss = (1.0 - cos).mean()

        return loss, a_g, v_g
