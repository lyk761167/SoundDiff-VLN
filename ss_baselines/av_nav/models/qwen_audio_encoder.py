# ss_baselines/av_nav/models/qwen_audio_encoder.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen2AudioEncoder


class QwenAudioEncoder(nn.Module):
    
    def __init__(self, observation_space, output_size, audiogoal_sensor):
        super().__init__()
        self._audiogoal_sensor = audiogoal_sensor

 
        self.audio_tower_dir = os.environ.get(
            "SS_QWEN_AUDIO_TOWER",
            "data/hf_models/Qwen2-Audio-7B-audio_tower",
        )

       
        self.audio_tower = Qwen2AudioEncoder.from_pretrained(
            self.audio_tower_dir,
            local_files_only=True,
        )

       
        hidden = getattr(self.audio_tower.config, "hidden_size", 1280)

     
        self.proj = nn.Linear(hidden, output_size)

        
        self.audio_tower.eval()
        for p in self.audio_tower.parameters():
            p.requires_grad = False

      
        self._debug = os.environ.get("SS_DEBUG_AUDIO_STATS", "0") == "1"
        self._debug_every = int(os.environ.get("SS_DEBUG_AUDIO_STATS_EVERY", "200"))
        self._forward_calls = 0

    @torch.no_grad()
    def _tower_forward(self, input_features):
        
        return self.audio_tower(input_features=input_features, return_dict=True).last_hidden_state

    def forward(self, observations):
        x = observations[self._audiogoal_sensor]  # [B,H,W,C]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

      
        x = x.float()

       
        x = x.permute(0, 3, 1, 2).contiguous()

      
        x = x.mean(dim=1, keepdim=True)  # [B,1,H,W]

      

        x = F.interpolate(x, size=(128, 3000), mode="bilinear", align_corners=False)


        mel = x.squeeze(1)


        
        tower_dtype = next(self.audio_tower.parameters()).dtype
        mel = mel.to(dtype=tower_dtype, device=next(self.audio_tower.parameters()).device)

        
        last_hidden = self._tower_forward(mel)  # [B,T,D]

        pooled = last_hidden.mean(dim=1)

        pooled = pooled.to(dtype=self.proj.weight.dtype)

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
