"""
models.py
---------
All model architectures used in the BrainWave project:

  SpatialCNN        — standalone CNN backbone (experiment 3)
  PositionalEncoding — sinusoidal positional encoding
  BrainWave         — full CNN + Transformer (our main model, experiment 5)
  CNNBiLSTM         — CNN backbone + bidirectional LSTM (experiment 4)

All models take input of shape [B, 64, 480] and produce logits [B, 4].
"""

import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    N_CHANNELS, N_TIMES, N_CLASSES,
    EMBED_DIM, N_HEADS, N_LAYERS, FF_DIM,
    DROPOUT, MAX_SEQ_LEN,
)


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Stores a [1, max_len, d] buffer and adds it to the input sequence.
    Buffer is not a trainable parameter.
    """
    def __init__(self, d: int, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d, 2).float() * -(math.log(10000.0) / d)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d]
        return x + self.pe[:, :x.size(1), :]


# ─────────────────────────────────────────────────────────────────────────────
# Spatial CNN Backbone
# ─────────────────────────────────────────────────────────────────────────────

class SpatialCNN(nn.Module):
    """
    Two-block CNN that extracts spatial and short-range temporal features.

    Block 1 (spatial):
        Depthwise conv with kernel (n_channels, 1) — mixes all 64 electrodes
        at each timestep independently. Learned analogue of CSP spatial filter.

    Block 2 (temporal):
        Conv with kernel (1, 15) — captures 15-sample (93 ms) local dynamics.

    Input:  [B, n_channels, n_times]
    Output: [B, n_times, d]   — sequence ready for Transformer or pooling
    """
    def __init__(
        self,
        n_channels: int  = N_CHANNELS,
        d:          int  = EMBED_DIM,
        dropout:    float = DROPOUT,
    ):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(1, d, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(d),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(d),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 64, 480]
        x = x.unsqueeze(1)              # [B, 1, 64, 480]
        x = self.spatial(x)             # [B, d, 1, 480]
        x = self.temporal(x)            # [B, d, 1, 480]
        x = x.squeeze(2)               # [B, d, 480]
        return x.transpose(1, 2)        # [B, 480, d]


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CNN (experiment 3 — no Transformer)
# ─────────────────────────────────────────────────────────────────────────────

class CNNOnly(nn.Module):
    """
    SpatialCNN backbone + global average pooling + linear head.
    Used as the ablation baseline to isolate the Transformer's contribution.

    Input:  [B, 64, 480]
    Output: [B, n_classes]  logits
    """
    def __init__(
        self,
        n_channels: int   = N_CHANNELS,
        d:          int   = EMBED_DIM,
        n_classes:  int   = N_CLASSES,
        dropout:    float = DROPOUT,
    ):
        super().__init__()
        self.cnn     = SpatialCNN(n_channels, d, dropout)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)                # [B, 480, d]
        x = x.mean(dim=1)             # [B, d]  global avg pool over time
        x = self.dropout(x)
        return self.head(x)            # [B, n_classes]


# ─────────────────────────────────────────────────────────────────────────────
# BrainWave — full CNN + Transformer (main model)
# ─────────────────────────────────────────────────────────────────────────────

class BrainWave(nn.Module):
    """
    BrainWave: SpatialCNN + Positional Encoding + Transformer Encoder + CLS head.

    Architecture:
        x [B, 64, 480]
        → SpatialCNN              → [B, 480, d]
        → prepend CLS token       → [B, 481, d]
        → PositionalEncoding
        → Dropout
        → TransformerEncoder (L layers, H heads, FF_DIM hidden)
        → CLS output [B, d]
        → Linear → [B, n_classes]

    The CLS token aggregates the full-trial temporal context through
    self-attention and is passed to the classification head.

    Args:
        n_channels : number of EEG channels (default 64)
        d          : model/embedding dimension (default 64)
        n_heads    : number of attention heads (default 8)
        n_layers   : number of Transformer encoder layers (default 4)
        ff_dim     : Transformer feed-forward hidden dim (default 256 = 4*d)
        n_classes  : output classes (default 4)
        dropout    : dropout probability (default 0.3)
    """
    def __init__(
        self,
        n_channels: int   = N_CHANNELS,
        d:          int   = EMBED_DIM,
        n_heads:    int   = N_HEADS,
        n_layers:   int   = N_LAYERS,
        ff_dim:     int   = FF_DIM,
        n_classes:  int   = N_CLASSES,
        dropout:    float = DROPOUT,
    ):
        super().__init__()
        self.d = d

        # Stage 1: spatial CNN
        self.cnn     = SpatialCNN(n_channels, d, dropout)

        # Learnable CLS token (prepended to the sequence before Transformer)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # Positional encoding (applied after CLS prepend)
        self.pos_enc = PositionalEncoding(d)

        self.dropout = nn.Dropout(dropout)

        # Stage 2: Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # input is [B, L, d], not [L, B, d]
            norm_first=False,   # post-norm (standard)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Classification head on the CLS token output
        self.head = nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # Stage 1: CNN feature extraction
        x = self.cnn(x)                              # [B, 480, d]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)       # [B, 1, d]
        x   = torch.cat([cls, x], dim=1)             # [B, 481, d]

        # Positional encoding + dropout
        x = self.pos_enc(x)
        x = self.dropout(x)

        # Stage 2: Transformer
        x = self.transformer(x)                      # [B, 481, d]

        # CLS token output (position 0) → classification
        cls_out = x[:, 0, :]                         # [B, d]
        return self.head(cls_out)                    # [B, n_classes]

    def get_attention_weights(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass that also returns attention weights from each layer.
        Used for visualization.

        Returns:
            logits      : [B, n_classes]
            attn_list   : list of [B, n_heads, L+1, L+1] tensors (one per layer)
        """
        B = x.size(0)
        x = self.cnn(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_enc(x)
        x   = self.dropout(x)

        attn_list = []
        for layer in self.transformer.layers:
            # Extract attention weights using need_weights=True
            # nn.TransformerEncoderLayer doesn't expose this directly,
            # so we call self_attn manually.
            x_norm = layer.norm1(x) if layer.norm_first else x
            attn_out, attn_weights = layer.self_attn(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=False,  # keep per-head
            )
            attn_list.append(attn_weights.detach())  # [B, H, L+1, L+1]

            # Complete the rest of the layer manually
            x = x + layer.dropout1(attn_out)
            if not layer.norm_first:
                x = layer.norm1(x)
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = x + layer.dropout2(ff_out)
            if not layer.norm_first:
                x = layer.norm2(x)

        cls_out = x[:, 0, :]
        logits  = self.head(cls_out)
        return logits, attn_list


# ─────────────────────────────────────────────────────────────────────────────
# CNN + Bidirectional LSTM (experiment 4)
# ─────────────────────────────────────────────────────────────────────────────

class CNNBiLSTM(nn.Module):
    """
    SpatialCNN backbone + 2-layer bidirectional LSTM + linear head.
    Used to test whether Transformer attention adds anything over a
    standard recurrent model.

    Input:  [B, 64, 480]
    Output: [B, n_classes]  logits
    """
    def __init__(
        self,
        n_channels:  int   = N_CHANNELS,
        d:           int   = EMBED_DIM,
        lstm_hidden: int   = 128,
        n_layers:    int   = 2,
        n_classes:   int   = N_CLASSES,
        dropout:     float = DROPOUT,
    ):
        super().__init__()
        self.cnn = SpatialCNN(n_channels, d, dropout)
        self.lstm = nn.LSTM(
            input_size=d,
            hidden_size=lstm_hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # BiLSTM output dim = 2 * lstm_hidden
        self.head = nn.Linear(2 * lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)            # [B, 480, d]
        _, (h_n, _) = self.lstm(x) # h_n: [2*n_layers, B, lstm_hidden]
        # Take the last layer's forward and backward hidden states
        fwd = h_n[-2]              # [B, lstm_hidden]
        bwd = h_n[-1]              # [B, lstm_hidden]
        out = torch.cat([fwd, bwd], dim=1)  # [B, 2*lstm_hidden]
        out = self.dropout(out)
        return self.head(out)      # [B, n_classes]


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def get_model(name: str, **kwargs) -> nn.Module:
    """
    Convenience factory. name must be one of:
        'brainwave', 'cnn_only', 'cnn_bilstm'

    Extra kwargs are forwarded to the model constructor.
    """
    registry = {
        "brainwave":   BrainWave,
        "cnn_only":    CNNOnly,
        "cnn_bilstm":  CNNBiLSTM,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(registry)}")
    return registry[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check — run each model on a dummy batch
    x = torch.randn(4, 64, 480)

    for name in ["brainwave", "cnn_only", "cnn_bilstm"]:
        model = get_model(name)
        logits = model(x)
        n_params = count_parameters(model)
        print(f"{name:15s} | output: {tuple(logits.shape)} | params: {n_params:,}")
