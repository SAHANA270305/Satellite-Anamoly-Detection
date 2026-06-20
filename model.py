"""
model.py
Per-channel PatchTST for satellite anomaly detection.
Architecture matches Section 5.2 of the project report.
"""

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    """
    Patch Time Series Transformer for univariate forecasting.
    One model instance is trained per telemetry channel.

    Args:
        seq_len    : context window length (default 96)
        pred_len   : forecast horizon      (default 24)
        patch_len  : patch size            (default 16)
        d_model    : transformer d_model   (default 96)
        nhead      : attention heads       (default 4)
        num_layers : encoder layers        (default 3)
        dropout    : dropout rate          (default 0.3)
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 24,
        patch_len: int = 16,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.patch_len = patch_len

        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"
        self.num_patches = seq_len // patch_len

        # Patch embedding  (patch_len → d_model)
        self.embed = nn.Linear(patch_len, d_model)

        # Learnable positional encoding
        self.pos = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN stabilises training
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Prediction head
        self.head = nn.Sequential(
            nn.Flatten(),                            # (B, num_patches * d_model)
            nn.Linear(self.num_patches * d_model, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, seq_len)   — single-channel input
        returns : (B, pred_len)
        """
        B = x.size(0)
        # Reshape into patches  (B, num_patches, patch_len)
        x = x.view(B, self.num_patches, self.patch_len)

        # Embed + positional encoding
        x = self.embed(x) + self.pos          # (B, num_patches, d_model)

        # Transformer
        x = self.encoder(x)                   # (B, num_patches, d_model)
        x = self.norm(x)

        # Predict
        out = self.head(x)                    # (B, pred_len)
        return out


# ── quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = PatchTST()
    dummy = torch.randn(8, 96)
    out = m(dummy)
    print("Output shape:", out.shape)   # Expected: torch.Size([8, 24])
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Trainable parameters: {n_params:,}")