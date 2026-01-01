# ss_baselines/av_nav/models/qwen_audio_encoder_lora.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen2AudioEncoder

# pip install peft
from peft import LoraConfig, get_peft_model


class QwenAudioEncoder(nn.Module):
    """
    LoRA version: Maintains the same interface
    observations[audiogoal_sensor] -> [B, output_size]
    """

    def __init__(self, observation_space, output_size, audiogoal_sensor):
        super().__init__()
        self._audiogoal_sensor = audiogoal_sensor

        self.audio_tower_dir = os.environ.get(
            "SS_QWEN_AUDIO_TOWER",
            "data/hf_models/Qwen2-Audio-7B-audio_tower",
        )

        # LoRA toggle (default ON: this is the LoRA version)
        self._enable_lora = os.environ.get("SS_QWEN_LORA", "1") == "1"

        # LoRA hyperparameters
        lora_r = int(os.environ.get("SS_QWEN_LORA_R", "16"))
        lora_alpha = int(os.environ.get("SS_QWEN_LORA_ALPHA", "32"))
        lora_dropout = float(os.environ.get("SS_QWEN_LORA_DROPOUT", "0.05"))

        self.audio_tower = Qwen2AudioEncoder.from_pretrained(
              self.audio_tower_dir,
              torch_dtype=torch.bfloat16,
            
           
              local_files_only=True,
        )

        hidden = getattr(self.audio_tower.config, "hidden_size", 1280)

        # Projection layer: typically you want to train the policy, so don't freeze this
        self.proj = nn.Linear(hidden, output_size)

        if self._enable_lora:
            # These names will match layers.*.self_attn.{q,k,v,out}_proj and layers.*.{fc1,fc2}
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=target_modules,
                task_type="FEATURE_EXTRACTION",
            )
            self.audio_tower = get_peft_model(self.audio_tower, lora_cfg)

            # Freeze non-LoRA parameters (for safety)
            for n, p in self.audio_tower.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

            # LoRA training requires train() (for dropout to take effect)
            self.audio_tower.train()

            # Optional: check how many parameters are trainable
            try:
                self.audio_tower.print_trainable_parameters()
            except Exception:
                pass
        else:
            # Pure frozen inference version (same as original file)
            self.audio_tower.eval()
            for p in self.audio_tower.parameters():
                p.requires_grad = False

        self._debug = os.environ.get("SS_DEBUG_AUDIO_STATS", "0") == "1"
        self._debug_every = int(os.environ.get("SS_DEBUG_AUDIO_STATS_EVERY", "200"))
        self._forward_calls = 0

    def _get_audio_core(self):
     m = self.audio_tower
     
     if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
         return m.base_model.model
    
     if hasattr(m, "get_base_model"):
         return m.get_base_model()
     return m

    def _tower_forward(self, input_features):
     m = self.audio_tower
  
     if hasattr(m, "base_model"):
         m = m.base_model  # 这是 LoraModel，forward 不会额外注入 input_ids
     out = m(input_features=input_features, return_dict=True)
     return out.last_hidden_state


    def forward(self, observations):
        x = observations[self._audiogoal_sensor]  # [B,H,W,C]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        x = x.float()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.mean(dim=1, keepdim=True)  # [B,1,H,W]

        # Your "align to Qwen tower input" method (verified to work)
        x = F.interpolate(x, size=(128, 3000), mode="bilinear", align_corners=False)
        mel = x.squeeze(1)  # [B,128,3000]

        # Align dtype/device with tower
        p0 = next(self.audio_tower.parameters())
        mel = mel.to(dtype=p0.dtype, device=p0.device)

        last_hidden = self._tower_forward(mel)  # [B,T,D]
        pooled = last_hidden.mean(dim=1)        # [B,D]

        # Avoid the Half/Float matmul error you encountered: ensure pooled matches proj weight dtype
        pooled = pooled.to(dtype=self.proj.weight.dtype, device=self.proj.weight.device)

        out = self.proj(pooled)  # [B, output_size]

        if self._debug:
            self._forward_calls += 1
            if self._forward_calls % self._debug_every == 0:
                m = out.mean().item()
                s = out.std(unbiased=False).item()
                has_nan = torch.isnan(out).any().item()
                print(f"[AUDIO_DEBUG] out mean={m:.6f} std={s:.6f} "
                      f"dtype={out.dtype} device={out.device} has_nan={has_nan}")

        return out