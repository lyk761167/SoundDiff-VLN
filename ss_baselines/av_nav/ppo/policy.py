#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import math
import os
import os.path as osp
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchsummary import summary
except Exception:
    summary = None

from ss_baselines.common.utils import CategoricalNet
from ss_baselines.av_nav.models.rnn_state_encoder import RNNStateEncoder
from ss_baselines.av_nav.models.visual_cnn import VisualCNN
from ss_baselines.av_nav.models.audio_cnn import AudioCNN

# Qwen audio encoder (optional)
try:
    from ss_baselines.av_nav.models.qwen_audio_encoder import QwenAudioEncoder
except Exception:
    QwenAudioEncoder = None

DUAL_GOAL_DELIMITER = ","


# -------------------------
# Utils
# -------------------------
def _find_first_tensor_ob(obs: Dict[str, Any], keys: List[str]) -> Optional[torch.Tensor]:
    for k in keys:
        if k in obs and torch.is_tensor(obs[k]):
            return obs[k]
    return None


def _infer_instr_dim_from_space(observation_space) -> Tuple[str, int]:
    """
    Try to infer an instruction embedding key and its dim from observation_space.
    """
    cand_keys = [
        "instr_embed",
        "instruction_embedding",
        "text_embed",
        "lang_embed",
        "sentence_embed",
        "instruction",
        "language",
    ]
    for k in cand_keys:
        if hasattr(observation_space, "spaces") and k in observation_space.spaces:
            sp = observation_space.spaces[k]
            shape = getattr(sp, "shape", None)
            if shape is not None and len(shape) == 1:
                return k, int(shape[0])
            if shape is not None and len(shape) == 2:
                # maybe [T, D] token-like; we only take pooled later, but dim is D
                return k, int(shape[-1])
    # fallback: none
    return "", 0


def _infer_pose_dim_from_space(observation_space) -> Tuple[str, int]:
    cand_keys = [
        "gps_compass",
        "agent_pose",
        "pose",
        "position",
        "ego_pose",
        "pointgoal_with_gps_compass",
    ]
    for k in cand_keys:
        if hasattr(observation_space, "spaces") and k in observation_space.spaces:
            sp = observation_space.spaces[k]
            shape = getattr(sp, "shape", None)
            if shape is not None and len(shape) == 1:
                return k, int(shape[0])
            if shape is not None and len(shape) == 2:
                return k, int(shape[-1])
    return "", 0


# -------------------------
# (2) Landmark Memory (KV + pose optional) with attention/topk
# -------------------------
class LandmarkMemoryKV(nn.Module):
    """
    Per-env ring buffer memory. Store keys/values (and optional pose embedding).
    Retrieve with attention. Designed to be simple, stable, and "actually memory".

    Shapes:
      keys:   [B, C, H]
      values: [B, C, H]
      filled: [B]
      ptr:    [B]
    """
    def __init__(self, hidden_size: int, capacity: int = 256, topk: int = 8, use_topk: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.capacity = capacity
        self.topk = topk
        self.use_topk = use_topk

        # buffers initialized lazily on first push (need batch size + device)
        self.register_buffer("_keys", torch.empty(0), persistent=False)
        self.register_buffer("_vals", torch.empty(0), persistent=False)
        self.register_buffer("_filled", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_ptr", torch.empty(0, dtype=torch.long), persistent=False)

    def _maybe_init(self, batch_size: int, device, dtype):
        if self._keys.numel() != 0 and self._keys.shape[0] == batch_size:
            return
        self._keys = torch.zeros(batch_size, self.capacity, self.hidden_size, device=device, dtype=dtype)
        self._vals = torch.zeros(batch_size, self.capacity, self.hidden_size, device=device, dtype=dtype)
        self._filled = torch.zeros(batch_size, device=device, dtype=torch.long)
        self._ptr = torch.zeros(batch_size, device=device, dtype=torch.long)

    @torch.no_grad()
    def reset_where(self, reset_mask: torch.Tensor):
        """
        reset_mask: [B] bool, True means reset memory for that env
        """
        if self._keys.numel() == 0:
            return
        if reset_mask.dtype != torch.bool:
            reset_mask = reset_mask.bool()
        if reset_mask.any():
            idx = reset_mask.nonzero(as_tuple=False).view(-1)
            self._keys[idx].zero_()
            self._vals[idx].zero_()
            self._filled[idx].zero_()
            self._ptr[idx].zero_()

    @torch.no_grad()
    def push(
        self,
        key: torch.Tensor,     # [B,H]
        value: torch.Tensor,   # [B,H]
        masks: Optional[torch.Tensor] = None,  # [B,1] or [B], 0 means new episode
    ):
        assert key.dim() == 2 and value.dim() == 2, "key/value must be [B,H]"
        B, H = key.shape
        self._maybe_init(B, device=key.device, dtype=key.dtype)

        if masks is not None:
            m = masks
            if m.dim() == 2:
                m = m.view(-1)
            # m==0 means reset
            self.reset_where(m == 0)

        ptr = self._ptr  # [B]
        b_idx = torch.arange(B, device=key.device)

        self._keys[b_idx, ptr] = key.detach()
        self._vals[b_idx, ptr] = value.detach()

        self._ptr = (ptr + 1) % self.capacity
        self._filled = torch.clamp(self._filled + 1, max=self.capacity)

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        query: [B,H]
        return: [B,H] retrieved summary
        """
        assert query.dim() == 2
        B, H = query.shape
        if self._keys.numel() == 0 or self._keys.shape[0] != B:
            # not initialized or batch mismatch
            return torch.zeros(B, H, device=query.device, dtype=query.dtype)

        filled = self._filled  # [B]
        empty = (filled == 0)

        # mask invalid slots
        slot_mask = torch.arange(self.capacity, device=query.device).view(1, -1)  # [1,C]
        valid = slot_mask < filled.view(-1, 1)  # [B,C]

        # attention scores: [B,C]
        scores = torch.einsum("bh,bch->bc", query, self._keys) / math.sqrt(H)
        scores = scores.masked_fill(~valid, float("-inf"))

        # IMPORTANT: avoid softmax over all -inf (when memory empty)
        if empty.any():
            scores = scores.clone()
            scores[empty] = 0.0

        if self.use_topk:
            k = min(self.topk, self.capacity)
            topk_scores, topk_idx = torch.topk(scores, k=k, dim=1)  # [B,k]
            attn = F.softmax(topk_scores, dim=1)  # [B,k]
            gathered = torch.gather(
                self._vals, dim=1, index=topk_idx.unsqueeze(-1).expand(-1, -1, H)
            )  # [B,k,H]
            out = torch.einsum("bk,bkh->bh", attn, gathered)
        else:
            attn = F.softmax(scores, dim=1)  # [B,C]
            out = torch.einsum("bc,bch->bh", attn, self._vals)

        if empty.any():
            out = out.clone()
            out[empty] = 0.0
        return out


# -------------------------
# (4) SoundDiff: minimal conditional latent diffusion prior
# -------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] int/float in [0, T-1]
        return: [B, dim]
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class SoundDiffPriorDiffusion(nn.Module):
    """
    Conditional diffusion in embedding space:
      x0: landmark embedding (e.g., target acoustic landmark) [B,H]
      cond: condition embedding (instr+visual summary) [B,H]
    """
    def __init__(self, hidden_size: int, T: int = 50):
        super().__init__()
        self.hidden_size = hidden_size
        self.T = T

        tdim = int(os.environ.get("SS_SOUNDDIFF_TEMB", "128"))
        self.time_emb = SinusoidalTimeEmbedding(tdim)

        # epsilon predictor: MLP([xt, cond, temb]) -> eps
        in_dim = hidden_size + hidden_size + tdim
        mid = int(os.environ.get("SS_SOUNDDIFF_MLP", str(hidden_size * 4)))
        self.eps_net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, mid),
            nn.SiLU(),
            nn.Linear(mid, mid),
            nn.SiLU(),
            nn.Linear(mid, hidden_size),
        )
        for m in self.eps_net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

        # DDPM schedule
        beta_start = float(os.environ.get("SS_SOUNDDIFF_BETA_START", "1e-4"))
        beta_end = float(os.environ.get("SS_SOUNDDIFF_BETA_END", "0.02"))
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        xt = sqrt(a_bar) * x0 + sqrt(1-a_bar) * noise
        """
        a_bar = self.alphas_cumprod[t].view(-1, 1).to(x0.device)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

    def predict_eps(self, xt: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_emb(t)  # [B,tdim]
        inp = torch.cat([xt, cond, temb.to(device=xt.device, dtype=xt.dtype)], dim=1)
        return self.eps_net(inp)

    def diff_loss(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        MSE(eps_pred, eps)
        """
        B = x0.shape[0]
        device = x0.device
        t = torch.randint(low=0, high=self.T, size=(B,), device=device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps_pred = self.predict_eps(xt, cond, t)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        DDPM sampling (few-step for speed). steps <= T. Use uniform stride.
        """
        B, H = cond.shape
        device = cond.device
        x = torch.randn(B, H, device=device, dtype=cond.dtype)

        steps = max(1, min(steps, self.T))
        stride = max(1, self.T // steps)
        ts = list(range(self.T - 1, -1, -stride))
        if ts[-1] != 0:
            ts.append(0)

        for ti in ts:
            t = torch.full((B,), ti, device=device, dtype=torch.long)
            beta = self.betas[t].view(-1, 1).to(device)
            alpha = self.alphas[t].view(-1, 1).to(device)
            a_bar = self.alphas_cumprod[t].view(-1, 1).to(device)

            eps = self.predict_eps(x, cond, t)

            # mean of posterior
            mean = (1.0 / torch.sqrt(alpha + 1e-8)) * (x - (beta / torch.sqrt(1.0 - a_bar + 1e-8)) * eps)

            if ti > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean

        return x


# -------------------------
# (3) Light cross-modal fusion Transformer (vector tokens)
# -------------------------
class FusionTransformer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        n_layers = int(os.environ.get("SS_FUSE_LAYERS", "1"))
        n_heads = int(os.environ.get("SS_FUSE_HEADS", "8"))
        ff = int(os.environ.get("SS_FUSE_FF", str(hidden_size * 4)))
        dropout = float(os.environ.get("SS_FUSE_DROPOUT", "0.0"))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=max(1, n_heads),
            dim_feedforward=max(hidden_size, ff),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=max(1, n_layers))
        self.out_ln = nn.LayerNorm(hidden_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, H]
        return: fused token for decision, default take token0 after encoding
        """
        y = self.enc(tokens)
        y0 = y[:, 0, :]
        return self.out_ln(y0)


# -------------------------
# PPO Policy
# -------------------------
class Policy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions
        self.action_distribution = CategoricalNet(self.net.output_size, self.dim_actions)
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states, prev_actions, masks)
        distribution = self.action_distribution(features)
        value = self.critic(features)
        action = distribution.mode() if deterministic else distribution.sample()
        action_log_probs = distribution.log_probs(action)
        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        return self.critic(features)

    def evaluate_actions(self, observations, rnn_hidden_states, prev_actions, masks, action):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states, prev_actions, masks)
        distribution = self.action_distribution(features)
        value = self.critic(features)
        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()
        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class AudioNavBaselinePolicy(Policy):
    def __init__(self, observation_space, action_space, goal_sensor_uuid, hidden_size=512, extra_rgb=False):
        super().__init__(
            AudioNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                extra_rgb=extra_rgb,
            ),
            action_space.n,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class AudioNavBaselineNet(Net):
    """
    AudioNav baseline +:
      (2) LandmarkMemoryKV (per-env KV memory, attention retrieve)
      (3) FusionTransformer (vector tokens)
      (4) SoundDiffPriorDiffusion (latent conditional diffusion prior)
      (5) Consistency gating (FIXED: gate increases when inconsistency increases)
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, extra_rgb=False):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self._audiogoal = False
        self._pointgoal = False
        self._n_pointgoal = 0

        # detect goal types
        if DUAL_GOAL_DELIMITER in self.goal_sensor_uuid:
            goal1_uuid, _ = self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)
            self._audiogoal = self._pointgoal = True
            self._n_pointgoal = observation_space.spaces[goal1_uuid].shape[0]
        else:
            if "pointgoal_with_gps_compass" == self.goal_sensor_uuid:
                self._pointgoal = True
                self._n_pointgoal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
            else:
                self._audiogoal = True

        # encoders
        self.visual_encoder = VisualCNN(observation_space, hidden_size, extra_rgb)

        # infer instruction & pose dims from observation_space (no Lazy modules!)
        self.instr_key, self.instr_dim = _infer_instr_dim_from_space(observation_space)
        self.pose_key, self.pose_dim = _infer_pose_dim_from_space(observation_space)

        # condition projector: cond = proj([visual, instr_pooled, pose])
        cond_in_dim = 0
        cond_in_dim += (0 if self.is_blind else hidden_size)
        cond_in_dim += self.instr_dim
        cond_in_dim += self.pose_dim

        self.cond_proj = nn.Sequential(
            nn.LayerNorm(cond_in_dim if cond_in_dim > 0 else hidden_size),
            nn.Linear(cond_in_dim if cond_in_dim > 0 else hidden_size, hidden_size),
            nn.Tanh(),
        )
        nn.init.orthogonal_(self.cond_proj[1].weight)
        nn.init.constant_(self.cond_proj[1].bias, 0.0)

        # NEW: project instr/pose to hidden_size for fusion tokens (avoid skipping)
        self.instr_proj = None
        if self.instr_dim > 0 and self.instr_dim != hidden_size:
            self.instr_proj = nn.Linear(self.instr_dim, hidden_size)
            nn.init.orthogonal_(self.instr_proj.weight)
            nn.init.constant_(self.instr_proj.bias, 0.0)

        self.pose_proj = None
        if self.pose_dim > 0 and self.pose_dim != hidden_size:
            self.pose_proj = nn.Linear(self.pose_dim, hidden_size)
            nn.init.orthogonal_(self.pose_proj.weight)
            nn.init.constant_(self.pose_proj.bias, 0.0)

        # audio
        self.audiogoal_sensor = None
        self.audio_cnn = None
        self.audio_qwen = None

        self.audio_adapter = None
        self.audio_gate = None
        self.audio_post_ln = None

        # SoundDiff + Memory switches
        self.use_sounddiff = os.environ.get("SS_USE_SOUNDDIFF", "0") == "1"
        self.sounddiff_mode = os.environ.get("SS_SOUNDDIFF_MODE", "diffusion").lower()  # diffusion/noise/dummy
        self.sounddiff_ckpt = os.environ.get("SS_SOUNDDIFF_CKPT", "")
        self.sounddiff_allow_no_ckpt = os.environ.get("SS_SOUNDDIFF_ALLOW_NO_CKPT", "1") == "1"
        self.freeze_sounddiff = os.environ.get("SS_SOUNDDIFF_FREEZE", "1") == "1"
        self.sounddiff_steps = int(os.environ.get("SS_SOUNDDIFF_STEPS", "10"))
        self.sounddiff_train = os.environ.get("SS_SOUNDDIFF_TRAIN", "0") == "1"
        self.sounddiff_lambda = float(os.environ.get("SS_SOUNDDIFF_LAMBDA", "0.1"))
        self.sounddiff_detach_x0 = os.environ.get("SS_SOUNDDIFF_DETACH_X0", "1") == "1"
        self.sounddiff_sigma = float(os.environ.get("SS_SOUNDDIFF_SIGMA", "0.1"))

        self.use_landmarks = (
            os.environ.get("SS_USE_LANDMARKS", "0") == "1"
            or os.environ.get("SS_LANDMARK", "0") == "1"
        )
        self.landmark_k = int(os.environ.get("SS_LANDMARK_K", "8"))
        self.landmark_capacity = int(os.environ.get("SS_LANDMARK_CAPACITY", "256"))
        self.landmark_use_topk = os.environ.get("SS_LANDMARK_USE_TOPK", "1") == "1"

        # (3) fusion transformer
        self.use_fusion_tf = os.environ.get("SS_USE_FUSION_TF", "1") == "1"
        self.fusion_tf = FusionTransformer(hidden_size) if self.use_fusion_tf else None

        # (5) dynamic gates from consistency
        # FIXED semantics: we gate by inconsistency distance (handled in _dynamic_gate_from_sim)
        self.prior_gate_w = nn.Parameter(torch.tensor(float(os.environ.get("SS_PRIOR_GATE_W", "4.0"))))
        self.prior_gate_b = nn.Parameter(torch.tensor(float(os.environ.get("SS_PRIOR_GATE_B", "-4.0"))))
        self.mem_gate_w = nn.Parameter(torch.tensor(float(os.environ.get("SS_MEM_GATE_W", "4.0"))))
        self.mem_gate_b = nn.Parameter(torch.tensor(float(os.environ.get("SS_MEM_GATE_B", "-4.0"))))
        self.gate_temp = float(os.environ.get("SS_GATE_TEMP", "1.0"))
        self.fuse_ln = nn.LayerNorm(hidden_size)

        # prior/mem projectors
        self.prior_proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size))
        nn.init.orthogonal_(self.prior_proj[1].weight)
        nn.init.constant_(self.prior_proj[1].bias, 0.0)

        self.mem_proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size))
        nn.init.orthogonal_(self.mem_proj[1].weight)
        nn.init.constant_(self.mem_proj[1].bias, 0.0)

        # build audio branch
        if self._audiogoal:
            if "audiogoal" in self.goal_sensor_uuid:
                self.audiogoal_sensor = "audiogoal"
            elif "spectrogram" in self.goal_sensor_uuid:
                self.audiogoal_sensor = "spectrogram"
            else:
                self.audiogoal_sensor = "audiogoal"

            enc_type = os.environ.get("SS_AUDIO_ENCODER", "cnn").lower()

            self.use_qwen_residual = enc_type in ("qwen_cnn", "hybrid", "qwen+cnn")
            self.freeze_qwen = os.environ.get("SS_FREEZE_QWEN", "1") == "1"
            self.freeze_cnn = os.environ.get("SS_FREEZE_CNN", "0") == "1"
            self.use_adapter = os.environ.get("SS_AUDIO_ADAPTER", "1") == "1"
            gate_init = float(os.environ.get("SS_AUDIO_GATE_INIT", "-2.0"))  # slightly less conservative default

            self.qwen_autocast = os.environ.get("SS_QWEN_AUTOCAST", "1") == "1"
            qwen_dtype = os.environ.get("SS_QWEN_DTYPE", "fp16").lower()
            self._qwen_amp_dtype = torch.bfloat16 if qwen_dtype == "bf16" else torch.float16

            print(
                f"[AUDIO] SS_AUDIO_ENCODER={enc_type} use_qwen_residual={int(self.use_qwen_residual)} "
                f"freeze_qwen={int(self.freeze_qwen)} freeze_cnn={int(self.freeze_cnn)}",
                flush=True,
            )

            self.audio_cnn = AudioCNN(observation_space, hidden_size, self.audiogoal_sensor)
            if self.freeze_cnn:
                for p in self.audio_cnn.parameters():
                    p.requires_grad = False

            if self.use_qwen_residual:
                if QwenAudioEncoder is None:
                    raise ImportError("QwenAudioEncoder not found. Please ensure qwen_audio_encoder.py exists.")
                self.audio_qwen = QwenAudioEncoder(observation_space, hidden_size, self.audiogoal_sensor)
                if self.freeze_qwen:
                    for p in self.audio_qwen.parameters():
                        p.requires_grad = False

                if self.use_adapter:
                    self.audio_adapter = nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Linear(hidden_size, hidden_size),
                        nn.Tanh(),
                    )
                    nn.init.orthogonal_(self.audio_adapter[1].weight)
                    nn.init.constant_(self.audio_adapter[1].bias, 0.0)
                    self.audio_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
                    self.audio_post_ln = nn.LayerNorm(hidden_size)
                    print(f"[AUDIO] adapter=ON gate_init={gate_init}", flush=True)
                else:
                    print("[AUDIO] adapter=OFF (will use gate * qwen directly)", flush=True)

        # (2) build memory
        self.landmark_memory = None
        if self.use_landmarks:
            self.landmark_memory = LandmarkMemoryKV(
                hidden_size=hidden_size,
                capacity=self.landmark_capacity,
                topk=self.landmark_k,
                use_topk=self.landmark_use_topk,
            )

        # (4) build sounddiff prior
        self.sounddiff = None
        if self.use_sounddiff:
            if self.sounddiff_mode == "diffusion":
                T = int(os.environ.get("SS_SOUNDDIFF_T", "50"))
                self.sounddiff = SoundDiffPriorDiffusion(hidden_size=hidden_size, T=T)
            elif self.sounddiff_mode == "noise":
                self.sounddiff = None  # handled in forward
            else:
                # dummy: a small MLP prior
                self.sounddiff = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                )
                for m in self.sounddiff:
                    if isinstance(m, nn.Linear):
                        nn.init.orthogonal_(m.weight)
                        nn.init.constant_(m.bias, 0.0)

            # load ckpt if provided
            if self.sounddiff_ckpt:
                if osp.isfile(self.sounddiff_ckpt):
                    ckpt = torch.load(self.sounddiff_ckpt, map_location="cpu")
                    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                    if self.sounddiff is not None:
                        missing, unexpected = self.sounddiff.load_state_dict(sd, strict=False)
                        print(
                            f"[SOUNDDIFF] loaded ckpt={self.sounddiff_ckpt} missing={len(missing)} unexpected={len(unexpected)}",
                            flush=True,
                        )
                else:
                    if not self.sounddiff_allow_no_ckpt:
                        raise FileNotFoundError(f"SS_SOUNDDIFF_CKPT not found: {self.sounddiff_ckpt}")
                    print(f"[SOUNDDIFF] ckpt not found -> run without ckpt (mode={self.sounddiff_mode})", flush=True)

            if self.freeze_sounddiff and self.sounddiff is not None:
                for p in self.sounddiff.parameters():
                    p.requires_grad = False
                self.sounddiff.eval()

            print(
                f"[SOUNDDIFF] use={int(self.use_sounddiff)} mode={self.sounddiff_mode} "
                f"freeze={int(self.freeze_sounddiff)} train={int(self.sounddiff_train)} ckpt='{self.sounddiff_ckpt}'",
                flush=True,
            )
        self._append_v_feat_to_rnn = (not self.is_blind) and not (self.use_fusion_tf and self.fusion_tf is not None)

        # RNN backbone
        rnn_input_size = (
            (hidden_size if self._append_v_feat_to_rnn else 0)
            + (self._n_pointgoal if self._pointgoal else 0)
            + (hidden_size if self._audiogoal else 0)
        )
        self.state_encoder = RNNStateEncoder(rnn_input_size, hidden_size)

        # aux loss cache
        self._last_aux: Dict[str, torch.Tensor] = {}

        # optional summaries
        print(self.visual_encoder.cnn)
        if self._audiogoal and summary is not None and isinstance(self.audio_cnn, AudioCNN):
            audio_shape = observation_space.spaces[self.audiogoal_sensor].shape
            try:
                summary(self.audio_cnn.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device="cpu")
            except Exception:
                pass

        self.train()

    def get_aux_losses(self) -> Dict[str, torch.Tensor]:
        return self._last_aux

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _forward_qwen(self, observations):
        assert self.audio_qwen is not None
        if self.freeze_qwen:
            with torch.no_grad():
                if torch.cuda.is_available() and self.qwen_autocast:
                    with torch.autocast(device_type="cuda", dtype=self._qwen_amp_dtype):
                        return self.audio_qwen(observations)
                return self.audio_qwen(observations)
        if torch.cuda.is_available() and self.qwen_autocast:
            with torch.autocast(device_type="cuda", dtype=self._qwen_amp_dtype):
                return self.audio_qwen(observations)
        return self.audio_qwen(observations)

    def _pool_instr(self, t: torch.Tensor) -> torch.Tensor:
        # accept [B,D] or [B,T,D] -> pooled [B,D]
        if t.dim() == 2:
            return t
        if t.dim() == 3:
            return t.mean(dim=1)
        return t.view(t.shape[0], -1)

    def _get_instr_feat(self, observations: Dict[str, Any], device, dtype) -> Optional[torch.Tensor]:
        if self.instr_key and self.instr_key in observations:
            t = observations[self.instr_key]
            if torch.is_tensor(t):
                t = self._pool_instr(t)
                if t.dim() == 2:
                    return t.to(device=device, dtype=dtype)
        # fallback search
        t2 = _find_first_tensor_ob(
            observations,
            ["instr_embed", "instruction_embedding", "text_embed", "lang_embed", "sentence_embed"],
        )
        if t2 is not None:
            t2 = self._pool_instr(t2)
            if t2.dim() == 2:
                return t2.to(device=device, dtype=dtype)
        return None

    def _get_pose_feat(self, observations: Dict[str, Any], device, dtype) -> Optional[torch.Tensor]:
        if self.pose_key and self.pose_key in observations:
            t = observations[self.pose_key]
            if torch.is_tensor(t):
                t = self._pool_instr(t)
                if t.dim() == 2:
                    return t.to(device=device, dtype=dtype)
        t2 = _find_first_tensor_ob(
            observations,
            ["gps_compass", "agent_pose", "pose", "position", "ego_pose", "pointgoal_with_gps_compass"],
        )
        if t2 is not None:
            t2 = self._pool_instr(t2)
            if t2.dim() == 2:
                return t2.to(device=device, dtype=dtype)
        return None

    def _build_cond(self, v_feat: Optional[torch.Tensor], observations: Dict[str, Any], ref: torch.Tensor) -> torch.Tensor:
        parts = []
        if v_feat is not None:
            parts.append(v_feat)
        instr = self._get_instr_feat(observations, device=ref.device, dtype=ref.dtype)
        if instr is not None:
            parts.append(instr)
        pose = self._get_pose_feat(observations, device=ref.device, dtype=ref.dtype)
        if pose is not None:
            parts.append(pose)

        if len(parts) == 0:
            # fallback: use ref itself
            return self.cond_proj(ref)
        x = torch.cat(parts, dim=1)
        return self.cond_proj(x)

    def _sounddiff_prior(self, cond: torch.Tensor) -> torch.Tensor:
        if not self.use_sounddiff:
            return torch.zeros_like(cond)

        if self.sounddiff_mode == "noise":
            return cond + torch.randn_like(cond) * self.sounddiff_sigma

        if self.sounddiff_mode == "diffusion":
            assert isinstance(self.sounddiff, SoundDiffPriorDiffusion)
            return self.sounddiff.sample(cond, steps=self.sounddiff_steps)

        # dummy
        assert self.sounddiff is not None
        return self.sounddiff(cond)

    def _dynamic_gate_from_sim(self, sim: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        FIXED: gate increases when inconsistency increases.
        sim in [-1,1]
        dist = (1 - sim)/2 in [0,1], dist越大越不一致 -> gate越大
        gate = sigmoid(w*(dist-0.5)+b)
        """
        dist = 0.5 * (1.0 - sim)  # [B]
        x = (w * (dist - 0.5) + b) / max(self.gate_temp, 1e-6)
        return torch.sigmoid(x).unsqueeze(1)  # [B,1]

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        self._last_aux = {}

        x = []

        # pointgoal
        if self._pointgoal:
            x.append(observations[self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)[0]])

        # visual
        v_feat = None
        if not self.is_blind:
            v_feat = self.visual_encoder(observations)  # [B,H]

        # audio branch
        if self._audiogoal:
            a_obs = self.audio_cnn(observations)  # [B,H] fp32 usually
            a = a_obs

            # qwen residual
            if self.use_qwen_residual:
                q = self._forward_qwen(observations)
                g0 = torch.sigmoid(self.audio_gate) if self.audio_gate is not None else 1.0
                if self.audio_adapter is not None:
                    res = self.audio_adapter(q.float()).to(dtype=a.dtype)
                    a = a + (g0.to(dtype=a.dtype) * res)
                    if self.audio_post_ln is not None:
                        a = self.audio_post_ln(a)
                else:
                    a = a + (g0.to(dtype=a.dtype) * q.to(dtype=a.dtype))

                if not hasattr(self, "_printed_gate_once") and (self.audio_gate is not None):
                    self._printed_gate_once = True
                    gv = float(torch.sigmoid(self.audio_gate).detach().cpu().item())
                    print(f"[AUDIO] gate(sigmoid)={gv:.6f} (audio = cnn + gate*adapter(qwen))", flush=True)

            # build condition (instr+visual+pose)
            cond = self._build_cond(v_feat=v_feat, observations=observations, ref=a)
            cond = cond.to(device=a.device, dtype=a.dtype)

            # (4) prior
            prior = self._sounddiff_prior(cond).to(device=a.device, dtype=a.dtype)
            prior = self.prior_proj(prior)

            # (2) memory: push + retrieve
            mem = None
            if self.landmark_memory is not None:
                self.landmark_memory.push(
                    key=cond.detach(),
                    value=a.detach(),
                    masks=masks,
                )
                mem = self.landmark_memory.retrieve(query=cond)
                mem = self.mem_proj(mem)

            # (5) dynamic gating (FIXED)
            sim_prior = F.cosine_similarity(a_obs, prior, dim=1)  # [B]
            gp = self._dynamic_gate_from_sim(sim_prior, self.prior_gate_w, self.prior_gate_b).to(dtype=a.dtype)  # [B,1]

            if mem is not None:
                sim_mem = F.cosine_similarity(a_obs, mem, dim=1)
                gm = self._dynamic_gate_from_sim(sim_mem, self.mem_gate_w, self.mem_gate_b).to(dtype=a.dtype)
            else:
                gm = None

            # fuse tokens (3) with transformer OR residual
            if self.use_fusion_tf and self.fusion_tf is not None:
                tokens = [a_obs]  # token0 as main
                tokens.append(gp * prior)
                if mem is not None and gm is not None:
                    tokens.append(gm * mem)
                if v_feat is not None:
                    tokens.append(v_feat.to(dtype=a.dtype))

                # NEW: always include instr/pose tokens with projection if needed
                instr = self._get_instr_feat(observations, device=a.device, dtype=a.dtype)
                if instr is not None:
                    if instr.shape[1] == self._hidden_size:
                        tokens.append(instr)
                    elif self.instr_proj is not None and instr.shape[1] == self.instr_dim:
                        tokens.append(self.instr_proj(instr))

                pose = self._get_pose_feat(observations, device=a.device, dtype=a.dtype)
                if pose is not None:
                    if pose.shape[1] == self._hidden_size:
                        tokens.append(pose)
                    elif self.pose_proj is not None and pose.shape[1] == self.pose_dim:
                        tokens.append(self.pose_proj(pose))

                tok = torch.stack(tokens, dim=1)  # [B,N,H]
                a = self.fusion_tf(tok)  # [B,H]
            else:
                a = a_obs + gp * prior
                if mem is not None and gm is not None:
                    a = a + gm * mem

            a = self.fuse_ln(a)

            # (4) diffusion training loss (aux) if enabled
            if (
                self.use_sounddiff
                and self.sounddiff_mode == "diffusion"
                and self.sounddiff_train
                and isinstance(self.sounddiff, SoundDiffPriorDiffusion)
            ):
                x0 = a_obs.detach() if self.sounddiff_detach_x0 else a_obs
                diff_loss = self.sounddiff.diff_loss(x0=x0, cond=cond)
                self._last_aux["sounddiff_loss"] = diff_loss * self.sounddiff_lambda

            # one-time debug
            if not hasattr(self, "_printed_fuse_once"):
                self._printed_fuse_once = True
                print(
                    f"[FUSE] sounddiff={int(self.use_sounddiff)} mode={self.sounddiff_mode} "
                    f"memory={int(self.landmark_memory is not None)} "
                    f"use_tf={int(self.use_fusion_tf)} instr_key='{self.instr_key}' pose_key='{self.pose_key}'",
                    flush=True,
                )

            x.append(a)

        # append visual to RNN input (baseline style)
        # IMPORTANT: avoid double-injecting v_feat when it is already a fusion token
        if v_feat is not None:
            if not (self.use_fusion_tf and self.fusion_tf is not None):
                x.append(v_feat)

        x1 = torch.cat(x, dim=1)
        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        if torch.isnan(x2).any().item():
            raise RuntimeError("NaN detected in policy forward")

        return x2, rnn_hidden_states1
