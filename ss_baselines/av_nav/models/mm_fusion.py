# ss_baselines/av_nav/models/mm_fusion.py
import torch
import torch.nn as nn


class MMTokenFusion(nn.Module):
    """
    Token-level fusion with a small TransformerEncoder.
    Input: list of tokens [B, D] -> stack -> [B, T, D]
    Output: fused [B, D] (CLS token)
    """
    def __init__(self, dim: int, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        """
        tokens: List[Tensor] each is [B, D]
        """
        assert isinstance(tokens, (list, tuple)) and len(tokens) > 0
        B = tokens[0].shape[0]
        x = torch.stack(tokens, dim=1)  # [B, T, D]

        cls = self.cls.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, x], dim=1)    # [B, 1+T, D]

        x = self.encoder(x)               # [B, 1+T, D]
        fused = self.out_norm(x[:, 0])    # CLS token [B, D]
        return fused
