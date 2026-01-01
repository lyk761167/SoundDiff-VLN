import os
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen2AudioEncoder, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map

from peft import LoraConfig, get_peft_model


def _parse_mem(s: str) -> str:
    return s.strip()


def _get_base_model(m: nn.Module) -> nn.Module:
    # PeftModel -> underlying model with LoRA injected
    if hasattr(m, "get_base_model"):
        return m.get_base_model()
    if hasattr(m, "base_model"):
        return m.base_model
    return m


def _guess_no_split_module_classes(model: nn.Module):
    cand = set()
    for mod in model.modules():
        n = mod.__class__.__name__
        if ("EncoderLayer" in n) or ("AudioEncoderLayer" in n) or (n.endswith("Layer") and ("Qwen" in n or "Audio" in n)):
            cand.add(n)
    if not cand:
        cand = {"Qwen2AudioEncoderLayer"}
    return sorted(list(cand))


def _get_max_memory(n_gpus: int):
    """
    Preferred:
      SS_QWEN_MAX_MEMORY="500MiB,22GiB,22GiB,22GiB"
    Fallback:
      SS_QWEN_MAX_MEM_PER_GPU="22GiB"
    """
    s = os.environ.get("SS_QWEN_MAX_MEMORY", "").strip()
    if s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        max_memory = {}
        for i in range(n_gpus):
            max_memory[i] = _parse_mem(parts[i] if i < len(parts) else parts[-1])
        # optional CPU offload:
        cpu_mem = os.environ.get("SS_QWEN_MAX_MEM_CPU", "").strip()
        if cpu_mem:
            max_memory["cpu"] = _parse_mem(cpu_mem)
        return max_memory

    per = os.environ.get("SS_QWEN_MAX_MEM_PER_GPU", "22GiB")
    max_memory = {i: _parse_mem(per) for i in range(n_gpus)}
    cpu_mem = os.environ.get("SS_QWEN_MAX_MEM_CPU", "").strip()
    if cpu_mem:
        max_memory["cpu"] = _parse_mem(cpu_mem)
    return max_memory


def _build_device_map(model_dir: str, n_gpus: int):
    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with init_empty_weights():
        try:
            empty = Qwen2AudioEncoder(cfg)
        except TypeError:
            empty = Qwen2AudioEncoder(config=cfg)

    no_split = _guess_no_split_module_classes(empty)
    max_memory = _get_max_memory(n_gpus)

    device_map = infer_auto_device_map(
        empty,
        max_memory=max_memory,
        no_split_module_classes=no_split,
    )
    return device_map, no_split, max_memory


class QwenAudioEncoderLoRASharded(nn.Module):
    """
    Input: observations[audiogoal_sensor] [B,H,W,C]
    Output: [B, output_size]
    """

    def __init__(self, observation_space, output_size: int, audiogoal_sensor: str):
        super().__init__()
        self._audiogoal_sensor = audiogoal_sensor

        self.audio_tower_dir = os.environ.get(
            "SS_QWEN_AUDIO_TOWER",
            "data/hf_models/Qwen2-Audio-7B-audio_tower",
        )

        # Qwen expects F=128 and T=3000 (hard requirement)
        self._target_F = int(os.environ.get("SS_QWEN_F", "128"))
        self._target_T = int(os.environ.get("SS_QWEN_T", "3000"))

        # dtype
        tower_dtype_str = os.environ.get("SS_QWEN_DTYPE", "bf16").lower()
        if tower_dtype_str in ("bf16", "bfloat16"):
            self._tower_dtype = torch.bfloat16
        elif tower_dtype_str in ("fp16", "float16", "half"):
            self._tower_dtype = torch.float16
        else:
            self._tower_dtype = torch.float32

        # sharding
        visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        n_gpus = int(os.environ.get("SS_QWEN_GPU_COUNT", str(min(4, visible_gpus))))
        n_gpus = max(1, n_gpus) if torch.cuda.is_available() else 0
        self._use_shard = bool(torch.cuda.is_available() and n_gpus >= 2)

        # LoRA
        self._use_lora = os.environ.get("SS_QWEN_LORA", "1") == "1"
        lora_r = int(os.environ.get("SS_QWEN_LORA_R", "4"))
        lora_alpha = int(os.environ.get("SS_QWEN_LORA_ALPHA", "16"))
        lora_dropout = float(os.environ.get("SS_QWEN_LORA_DROPOUT", "0.05"))
        target_modules = os.environ.get("SS_QWEN_LORA_TARGET", "q_proj,v_proj")
        target_modules = [s.strip() for s in target_modules.split(",") if s.strip()]

        # load tower
        if self._use_shard:
            device_map, no_split, max_memory = _build_device_map(self.audio_tower_dir, n_gpus)
            self._device_map = device_map
            self._no_split = no_split
            self._max_memory = max_memory

            base = Qwen2AudioEncoder.from_pretrained(
                self.audio_tower_dir,
                local_files_only=True,
                 device_map="balanced_low_0",   
                dtype=self._tower_dtype,   # avoid torch_dtype deprecation
                low_cpu_mem_usage=True,
                max_memory={i: "48GiB" for i in range(n_gpus)}, 
            )
            # ---- DEBUG: check real sharding result ----
            if hasattr(base, "hf_device_map"):
              from collections import Counter
              c = Counter(base.hf_device_map.values())
              print("[AUDIO_TOWER] hf_device_map counts:", dict(c), flush=True)
            else:
              print("[AUDIO_TOWER] no hf_device_map on base model", flush=True)

        else:
            self._device_map = None
            self._no_split = None
            self._max_memory = None
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            base = Qwen2AudioEncoder.from_pretrained(
                self.audio_tower_dir,
                local_files_only=True,
                dtype=self._tower_dtype,
                low_cpu_mem_usage=True,
            ).to(dev)

        # reduce memory
        if hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable()
        if hasattr(base, "config"):
            base.config.use_cache = False

        hidden = getattr(base.config, "hidden_size", 1280)

        # proj on cuda:0
        self.proj = nn.Linear(hidden, output_size)
        if torch.cuda.is_available():
            self.proj = self.proj.to("cuda:0")

        # attach LoRA or freeze
        if self._use_lora:
            for p in base.parameters():
                p.requires_grad = False

            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=target_modules,
            )
            self.audio_tower = get_peft_model(base, lora_cfg)
        else:
            self.audio_tower = base
            self.audio_tower.eval()
            for p in self.audio_tower.parameters():
                p.requires_grad = False

        # input device must match first real parameter device
        base_for_device = _get_base_model(self.audio_tower)
        self._tower_input_device = next(base_for_device.parameters()).device

        # debug
        print(f"[AUDIO_TOWER] use_shard={self._use_shard} visible_gpus={visible_gpus} n_gpus={n_gpus}", flush=True)
        print(f"[AUDIO_TOWER] tower_dtype={self._tower_dtype} input_device={self._tower_input_device}", flush=True)
        if self._device_map is not None:
            c = Counter(self._device_map.values())
            print(f"[AUDIO_TOWER] device_map len={len(self._device_map)} summary={dict(c)}", flush=True)
            print(f"[AUDIO_TOWER] no_split_module_classes={self._no_split}", flush=True)
            print(f"[AUDIO_TOWER] max_memory={self._max_memory}", flush=True)

        trainable = [(n, p.numel()) for n, p in self.named_parameters() if p.requires_grad]
        print("[TRAINABLE] count =", len(trainable), " params =", sum(x[1] for x in trainable), flush=True)
        print("[TRAINABLE] top names:", [n for n, _ in trainable[:20]], flush=True)

    def _apply(self, fn):
        # protect sharded tower from .to(device)
        if "audio_tower" in self._modules:
            tower = self._modules.pop("audio_tower")
            try:
                super()._apply(fn)
            finally:
                self._modules["audio_tower"] = tower
            return self
        return super()._apply(fn)

    def _tower_forward(self, input_features: torch.Tensor) -> torch.Tensor:
        # DO NOT call PeftModel.forward
        m = _get_base_model(self.audio_tower)
        out = m(input_features=input_features, return_dict=True)
        return out.last_hidden_state

    def forward(self, observations):
        x = observations[self._audiogoal_sensor]  # [B,H,W,C]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.float()

        x = x.permute(0, 3, 1, 2).contiguous()   # [B,C,H,W]
        x = x.mean(dim=1, keepdim=True)          # [B,1,H,W]

        # enforce (F=128, T=3000)
        x = F.interpolate(
            x,
            size=(self._target_F, self._target_T),
            mode="bilinear",
            align_corners=False,
        )
        mel = x.squeeze(1)  # [B,128,3000]

        mel = mel.to(self._tower_input_device, dtype=self._tower_dtype)

        last_hidden = self._tower_forward(mel)   # [B,T,D]
        pooled = last_hidden.mean(dim=1)         # [B,D]

        pooled = pooled.to(self.proj.weight.device, dtype=self.proj.weight.dtype)
        out = self.proj(pooled)
       
        return out
