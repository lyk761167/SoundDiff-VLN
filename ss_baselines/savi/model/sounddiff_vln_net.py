
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F



try:
    from qwen_audio_encoder_lora_sharded import QwenAudioEncoderLoraSharded
except Exception as e:
    raise ImportError(
        "Cannot find QwenAudioEncoderLoraSharded in qwen_audio_encoder_lora_sharded.py. "
        "Ensure it's on PYTHONPATH, or change to your real import path."
    ) from e

try:
    from vggt_encoder import VGGTEncoder
except Exception as e:
    raise ImportError(
        "Cannot find VGGTEncoder (e.g. vggt_encoder.py / VGGTEncoder). "
        "Change to your repo's real VGGT import path/class name."
    ) from e


try:
    from habitat_baselines.rl.ppo.policy import Net
except Exception as e:
    raise ImportError(
        "Cannot find habitat_baselines.rl.ppo.policy.Net. "
        "Change to your project's Net base class."
    ) from e


@dataclass
class SoundDiffCfg:
    embed_dim: int = 512
    diffusion_steps: int = 10
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    hidden_dim: int = 1024

 
    freeze_in_rl: bool = True

   
    sample_steps: int = 4 


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] int/float
        return: [B, dim]
        """
        half = self.dim // 2
        device = t.device
        if half <= 1:
        
            emb = t.float().unsqueeze(-1)
            return F.pad(emb, (0, max(0, self.dim - 1)))

        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, device=device)) / (half - 1))
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class SoundDiffPrior(nn.Module):
    """
    Conditional diffusion in embedding space: learn eps_theta(z_t, cond, t).
    Training: MSE noise prediction. Inference: iterative denoising from N(0,1).
    """

    def __init__(self, cfg: SoundDiffCfg, cond_dim: int):
        super().__init__()
        self.cfg = cfg

        self.register_buffer("betas", torch.linspace(cfg.beta_start, cfg.beta_end, cfg.diffusion_steps))
        alphas = 1.0 - self.betas
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

        self.time_emb = SinusoidalTimeEmbedding(cfg.embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(cfg.embed_dim + cond_dim + cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )

    def q_sample(self, z0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        z_t = sqrt(alpha_bar_t) z0 + sqrt(1-alpha_bar_t) noise
        z0/noise: [B, D], t_idx: [B] long in [0..T-1]
        """
        ab = self.alpha_bars[t_idx].unsqueeze(-1)  # [B,1]
        return torch.sqrt(ab) * z0 + torch.sqrt(1.0 - ab) * noise

    def eps_theta(self, zt: torch.Tensor, cond: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_emb(t_idx))  # [B,H]
        x = torch.cat([zt, cond, t_emb], dim=-1)
        return self.net(x)

    def diffusion_loss(self, z0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, _ = z0.shape
        device = z0.device
        t_idx = torch.randint(0, self.cfg.diffusion_steps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(z0)
        zt = self.q_sample(z0, t_idx, noise)
        pred = self.eps_theta(zt, cond, t_idx)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def p_sample(self, zt: torch.Tensor, cond: torch.Tensor, t_idx: int) -> torch.Tensor:
        """
        DDPM single step: z_t -> z_{t-1} using noise prediction form.
        """
        beta_t = self.betas[t_idx]
        alpha_t = self.alphas[t_idx]
        ab_t = self.alpha_bars[t_idx]

        B, _ = zt.shape
        t = torch.full((B,), t_idx, device=zt.device, dtype=torch.long)
        eps = self.eps_theta(zt, cond, t)

        mean = (1.0 / torch.sqrt(alpha_t)) * (zt - (beta_t / torch.sqrt(1.0 - ab_t)) * eps)

        if t_idx == 0:
            return mean
        noise = torch.randn_like(zt)
        sigma = torch.sqrt(beta_t)
        return mean + sigma * noise

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, sample_steps: Optional[int] = None) -> torch.Tensor:
        """
        Iterative denoising from N(0,1) to get z0_hat.
        For compute-saving, you can use fewer steps by skipping timesteps.
        """
        B = cond.shape[0]
        D = self.cfg.embed_dim
        z = torch.randn(B, D, device=cond.device)

        T = self.cfg.diffusion_steps
        K = int(sample_steps) if sample_steps is not None else int(self.cfg.sample_steps)
        K = max(1, min(K, T))

        if K == T:
            timesteps = list(reversed(range(T)))
        else:
            
            idx = torch.linspace(0, T - 1, steps=K, device=cond.device)
            timesteps = [int(i.item()) for i in torch.round(idx)]
            timesteps = sorted(set(timesteps))
            timesteps = list(reversed(timesteps))

        for t in timesteps:
            z = self.p_sample(z, cond, t)
        return z


class KVLandmarkMemory(nn.Module):
    """
    Fixed-slot KV memory with attention read.
    NOTE: This is a simplified "growing memory" approximation via shift-register.
    """

    def __init__(self, mem_slots: int, dim: int):
        super().__init__()
        self.mem_slots = mem_slots
        self.dim = dim
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.val_proj = nn.Linear(dim, dim, bias=False)
        self.query_proj = nn.Linear(dim, dim, bias=False)

    def write(self, memory: torch.Tensor, new_item: torch.Tensor) -> torch.Tensor:
        """
        memory: [B, M, D], new_item: [B, D]
        """
        memory = torch.roll(memory, shifts=1, dims=1)
        memory[:, 0, :] = new_item
        return memory

    def read(self, memory: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        returns retrieval summary: [B, D]
        """
        K = self.key_proj(memory) 
        V = self.val_proj(memory)  
        q = self.query_proj(query).unsqueeze(1) 
        attn = torch.softmax((q * K).sum(-1) / (self.dim**0.5), dim=-1)  
        summary = (attn.unsqueeze(-1) * V).sum(1)  
        return summary


class SimpleFusionTransformer(nn.Module):
    """
    Cross-modal fusion: stack tokens then TransformerEncoder; use CLS output.
    """

    def __init__(self, dim: int, nhead: int = 8, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, T, D]
        return: [B, D]
        """
        B = tokens.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        y = self.enc(x)
        return y[:, 0, :]


class SoundDiffVLNNet(Net):
    """
    Fixes vs your previous version:
      1) Memory write now explicitly binds (audio, vision, pose/topo) -> landmark embedding.
      2) Prior token is injected as a separate token (audio_obs and audio_prior both exist).
      3) True "freeze_in_rl": diffusion loss computed under no_grad when freeze_in_rl and training.
      4) Compute-saving: SoundDiff sampling supports timestep skipping via cfg.sample_steps.
      5) Adds optional alignment loss (lightweight) to encourage cross-modal landmark alignment.

    Output:
      features [B, H], new_state [B, H + M*D], aux dict
    """

    def __init__(self, observation_space, config):
        super().__init__()

        self.cfg = config
        self.embed_dim = int(getattr(config.MODEL, "EMBED_DIM", 512))
        self.rnn_hidden_size = int(getattr(config.MODEL, "RNN_HIDDEN_SIZE", 512))
        self.mem_slots = int(getattr(config.MODEL.MEMORY, "SLOTS", 16))

    
        self.pose_input_dim = int(getattr(config.MODEL, "POSE_INPUT_DIM", 4))
        self.use_pose = bool(getattr(config.MODEL, "USE_POSE", True))
        self.pose_adapter = nn.Sequential(
            nn.Linear(self.pose_input_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.audio_encoder = QwenAudioEncoderLoraSharded(config.MODEL.AUDIO)
        self.audio_adapter = nn.Linear(self.audio_encoder.output_dim, self.embed_dim)

        self.vision_encoder = VGGTEncoder(config.MODEL.VISION)
        self.vision_adapter = nn.Linear(self.vision_encoder.output_dim, self.embed_dim)

        task_in_dim = int(getattr(config.MODEL.TASK, "INPUT_DIM", self.embed_dim))
        self.task_adapter = nn.Linear(task_in_dim, self.embed_dim)

 
        n_actions = int(getattr(config.TASK_CONFIG.TASK, "POSSIBLE_ACTIONS", 6))
        self.prev_action_emb = nn.Embedding(n_actions, self.embed_dim)
        self.prev_action_proj = nn.Linear(self.embed_dim, self.embed_dim)

    
        self.memory = KVLandmarkMemory(self.mem_slots, self.embed_dim)
      
        self.landmark_bind = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        sd_cfg = SoundDiffCfg(
            embed_dim=self.embed_dim,
            diffusion_steps=int(getattr(config.MODEL.SOUNDDIFF, "STEPS", 10)),
            beta_start=float(getattr(config.MODEL.SOUNDDIFF, "BETA_START", 1e-4)),
            beta_end=float(getattr(config.MODEL.SOUNDDIFF, "BETA_END", 2e-2)),
            hidden_dim=int(getattr(config.MODEL.SOUNDDIFF, "HIDDEN_DIM", 1024)),
            freeze_in_rl=bool(getattr(config.MODEL.SOUNDDIFF, "FREEZE_IN_RL", True)),
            sample_steps=int(getattr(config.MODEL.SOUNDDIFF, "SAMPLE_STEPS", 4)),
        )
        self.sounddiff_cfg = sd_cfg

       
        self.cond_proj = nn.Linear(self.embed_dim * 3, self.embed_dim)
        self.sounddiff = SoundDiffPrior(sd_cfg, cond_dim=self.embed_dim)

       
        self.consistency_w = nn.Parameter(torch.tensor(3.0))
        self.consistency_b = nn.Parameter(torch.tensor(0.0))

       
        self.gate_token = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

      
        self.align_w = float(getattr(config.MODEL, "ALIGN_LOSS_W", 0.0))

      
        self.fusion = SimpleFusionTransformer(self.embed_dim, nhead=8, nlayers=2)
        self.rnn = nn.GRU(self.embed_dim, self.rnn_hidden_size, num_layers=1)
        self._output_size = self.rnn_hidden_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def is_blind(self):
        return False

    def _unpack_state(self, rnn_hidden_states: Optional[torch.Tensor], B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        rnn_hidden_states: [B, H + M*D]
        -> gru_h: [1,B,H], mem: [B,M,D]
        """
        H = self.rnn_hidden_size
        D = self.embed_dim
        M = self.mem_slots

        if rnn_hidden_states is None or rnn_hidden_states.numel() == 0:
            device = next(self.parameters()).device
            gru_h = torch.zeros(1, B, H, device=device)
            mem = torch.zeros(B, M, D, device=device)
            return gru_h, mem

        if rnn_hidden_states.shape[1] != (H + M * D):
            raise ValueError(
                f"rnn_hidden_states dim mismatch: expected {H + M*D}, got {rnn_hidden_states.shape[1]}. "
                "Make sure trainer initializes hidden state with correct size."
            )

        gru_flat = rnn_hidden_states[:, :H]        
        mem_flat = rnn_hidden_states[:, H:]        
        gru_h = gru_flat.unsqueeze(0).contiguous() 
        mem = mem_flat.view(B, M, D).contiguous()  
        return gru_h, mem

    def _pack_state(self, gru_h: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """
        gru_h: [1,B,H], mem: [B,M,D] -> [B,H+M*D]
        """
        B = mem.size(0)
        gru_flat = gru_h.squeeze(0)  
        mem_flat = mem.view(B, -1)   
        return torch.cat([gru_flat, mem_flat], dim=1)

    def _get_pose_tensor(self, observations: Dict[str, Any], B: int, device: torch.device) -> torch.Tensor:
        """
        Tries common keys. If not found, returns zeros.
        Expected pose shape: [B, pose_input_dim] (or can be flattened).
        """
        if not self.use_pose:
            return torch.zeros(B, self.embed_dim, device=device)

        pose = observations.get("pose", None)
        if pose is None:
            pose = observations.get("gps_compass", None)  
        if pose is None:
            raw = torch.zeros(B, self.pose_input_dim, device=device)
            return self.pose_adapter(raw)

        if not torch.is_tensor(pose):
            pose = torch.as_tensor(pose, device=device)

        pose = pose.to(device)
        pose = pose.view(B, -1)
        if pose.size(1) < self.pose_input_dim:
            pose = F.pad(pose, (0, self.pose_input_dim - pose.size(1)))
        elif pose.size(1) > self.pose_input_dim:
            pose = pose[:, : self.pose_input_dim]

        return self.pose_adapter(pose)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        """
        Required observation keys (adjust to your project):
          - "rgb" or "image": for VGGT
          - "audio" or "spectrogram": for QwenAudioEncoder
          - "task_embedding": instruction/goal embedding
        Optional:
          - "pose" or "gps_compass": pose/topo info for memory binding
        """
        device = next(self.parameters()).device
        B = masks.shape[0]

        gru_h, mem = self._unpack_state(rnn_hidden_states, B)

    
        done = (masks.view(B) == 0)
        if done.any():
            mem = mem.clone()
            mem[done] = 0.0
            gru_h = gru_h.clone()
            gru_h[:, done, :] = 0.0

    
        img = observations.get("rgb", observations.get("image", None))
        aud = observations.get("audio", observations.get("spectrogram", None))
        task_emb = observations.get("task_embedding", None)

        if img is None or aud is None or task_emb is None:
            missing = [k for k, v in [("rgb/image", img), ("audio/spectrogram", aud), ("task_embedding", task_emb)] if v is None]
            raise KeyError(f"observations missing required keys: {missing}")

        v = self.vision_adapter(self.vision_encoder(img))    
        a_obs = self.audio_adapter(self.audio_encoder(aud)) 
        t = self.task_adapter(task_emb)                      

        pose_emb = self._get_pose_tensor(observations, B, device)  


        query = 0.5 * (t + v)
        mem_summary = self.memory.read(mem, query) 

   
        cond = self.cond_proj(torch.cat([t, v, mem_summary], dim=-1))  

    
        if self.sounddiff_cfg.freeze_in_rl and self.training:
            with torch.no_grad():
                diff_loss = self.sounddiff.diffusion_loss(z0=a_obs, cond=cond)
        else:
            diff_loss = self.sounddiff.diffusion_loss(z0=a_obs, cond=cond)

        a_prior = self.sounddiff.sample(cond, sample_steps=self.sounddiff_cfg.sample_steps)  

      
        sim = F.cosine_similarity(
            F.normalize(a_obs, dim=-1),
            F.normalize(a_prior, dim=-1),
            dim=-1
        )
        gate = torch.sigmoid(self.consistency_w * sim + self.consistency_b).unsqueeze(-1) 

       
        a_fused = gate * a_obs + (1.0 - gate) * a_prior  

      
        landmark_item = self.landmark_bind(torch.cat([a_fused, v, pose_emb], dim=-1))  
        mem = self.memory.write(mem, landmark_item)

        align_loss = torch.tensor(0.0, device=device)
        if self.align_w > 0.0:
           
            sim_av = F.cosine_similarity(F.normalize(a_fused, dim=-1), F.normalize(v, dim=-1), dim=-1)
            sim_pv = F.cosine_similarity(F.normalize(a_prior, dim=-1), F.normalize(v, dim=-1), dim=-1)
       
            align_loss = (1.0 - sim_av).mean() + 0.5 * (1.0 - sim_pv).mean()


        prev_a = prev_actions.view(B)
        prev_tok = self.prev_action_proj(self.prev_action_emb(prev_a))  

        gate_tok = self.gate_token(gate) 


        tokens = torch.stack(
            [t, v, a_obs, a_prior, mem_summary, prev_tok, gate_tok],
            dim=1
        )  

        fused = self.fusion(tokens)  

        x = fused.unsqueeze(0)           
        out, gru_h = self.rnn(x, gru_h)  
        feat = out.squeeze(0)            

        new_state = self._pack_state(gru_h, mem)

        aux = {
            "sounddiff_loss": diff_loss,
            "align_loss": align_loss,
            "consistency": sim.detach(),
            "gate": gate.detach(),
            "sample_steps": torch.tensor(int(self.sounddiff_cfg.sample_steps), device=device),
        }
        return feat, new_state, aux
