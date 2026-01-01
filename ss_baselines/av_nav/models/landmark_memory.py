#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F


class LandmarkMemory(nn.Module):
    """
    Per-env FIFO memory:
      - store audio keys:   mem_k  [N, K, H]
      - store context vals: mem_v  [N, K, H]   (e.g., visual summary or fused state)
      - store valid mask:   mem_m  [N, K]
    Retrieval:
      q [N, H] -> attn over K -> m [N, H]
    """

    def __init__(self, num_envs: int, hidden_size: int, K: int = 128, topk: int = 8):
        super().__init__()
        self.num_envs = int(num_envs)
        self.hidden_size = int(hidden_size)
        self.K = int(K)
        self.topk = int(topk)

        # registered buffers so they move with .to(device) and saved in state_dict if needed
        self.register_buffer("mem_k", torch.zeros(self.num_envs, self.K, self.hidden_size))
        self.register_buffer("mem_v", torch.zeros(self.num_envs, self.K, self.hidden_size))
        self.register_buffer("mem_m", torch.zeros(self.num_envs, self.K))  # 0/1
        self.register_buffer("ptr", torch.zeros(self.num_envs, dtype=torch.long))

    @torch.no_grad()
    def reset_where(self, done_mask: torch.Tensor):
        """
        done_mask: [N, 1] or [N]  where 1 means done -> reset that env memory
        """
        if done_mask.dim() == 2:
            done_mask = done_mask.squeeze(1)
        done_mask = done_mask.bool()
        if done_mask.numel() == 0:
            return
        idx = torch.where(done_mask)[0]
        if idx.numel() == 0:
            return
        self.mem_k[idx].zero_()
        self.mem_v[idx].zero_()
        self.mem_m[idx].zero_()
        self.ptr[idx].zero_()

    @torch.no_grad()
    def push(self, k: torch.Tensor, v: torch.Tensor, not_done_masks: torch.Tensor):
        """
        k, v: [N, H]
        not_done_masks: [N, 1] where 1 means continue, 0 means episode ended (so we reset before push)
        """
        # reset where done
        done_mask = (not_done_masks <= 0.0).float()
        self.reset_where(done_mask)

        N = k.size(0)
        assert N == self.num_envs, f"LandmarkMemory num_envs mismatch: got {N} vs {self.num_envs}"

        p = self.ptr  # [N]
        self.mem_k[torch.arange(N), p] = k
        self.mem_v[torch.arange(N), p] = v
        self.mem_m[torch.arange(N), p] = 1.0
        self.ptr[:] = (p + 1) % self.K

    def retrieve(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: [N, H]
        return: [N, H]
        """
        # cosine similarity with mask
        qn = F.normalize(q, dim=-1)                           # [N,H]
        kn = F.normalize(self.mem_k, dim=-1)                  # [N,K,H]
        sim = torch.einsum("nh,nkh->nk", qn, kn)              # [N,K]
        sim = sim.masked_fill(self.mem_m <= 0.0, -1e9)

        # topk attention
        k = min(self.topk, self.K)
        vals, idx = torch.topk(sim, k=k, dim=1)               # [N,k]
        attn = torch.softmax(vals, dim=1)                     # [N,k]

        v_sel = torch.gather(
            self.mem_v, 1, idx.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        )                                                     # [N,k,H]
        out = torch.einsum("nk,nkh->nh", attn, v_sel)         # [N,H]
        return out
