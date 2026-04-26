"""
PeriECG-RiskNet: Lead-Aware CNN-iTransformer-LSTM Model
with Monte Carlo Dropout Uncertainty Estimation

Corrected, runnable reference implementation.

Key fixes vs. the original draft:
1. LeadAwareAttention now correctly consumes (B, num_leads, d_model)
   and uses Linear + MultiheadAttention instead of an invalid Conv1d path.
2. Monte Carlo dropout inference no longer relies on BatchNorm in train mode,
   so batch size 1 works correctly.
3. Prediction utilities are aligned with the multi-label 13-class target setup
   used by the provided loader/metrics pipeline (sigmoid probabilities).
4. Forward / uncertainty utilities / attention-map extraction are all runnable.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation1D(nn.Module):
    """Channel-wise SE block for 1D convolutions."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResBlock1D(nn.Module):
    """Residual 1D block with optional SE attention."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 7,
        stride: int = 1,
        use_se: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SqueezeExcitation1D(out_ch) if use_se else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + residual
        out = self.relu(out)
        out = self.dropout(out)
        return out


class CNNBackbone(nn.Module):
    """
    Multi-scale 1D CNN encoder for ECG feature extraction.
    Processes all leads jointly, then builds progressively deeper features.
    """

    def __init__(
        self,
        in_channels: int = 7,
        base_filters: int = 64,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        layers = []
        in_ch = base_filters
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            layers.append(ResBlock1D(in_ch, out_ch, kernel_size=7, stride=2, dropout=dropout))
            layers.append(ResBlock1D(out_ch, out_ch, kernel_size=7, stride=1, dropout=dropout))
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)
        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.encoder(x)
        return x  # (B, C, T')


class LeadAwareAttention(nn.Module):
    """
    Cross-lead attention over lead tokens.

    Input:
        x: (B, num_leads, embed_dim)
    Output:
        attn_out: (B, num_leads, embed_dim)
        attn_weights: (B, num_leads, num_leads)
    """

    def __init__(
        self,
        embed_dim: int,
        num_leads: int = 7,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_leads = num_leads
        self.embed_dim = embed_dim
        self.in_proj = nn.Linear(embed_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"LeadAwareAttention expects 3D input, got shape={tuple(x.shape)}")
        if x.shape[1] != self.num_leads:
            raise ValueError(
                f"Expected {self.num_leads} lead tokens, got shape={tuple(x.shape)}"
            )

        x_proj = self.in_proj(x)
        attn_out, attn_weights = self.cross_attn(
            x_proj,
            x_proj,
            x_proj,
            need_weights=True,
            average_attn_weights=True,
        )
        x = self.norm1(x_proj + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, attn_weights


class iTransformerEncoder(nn.Module):
    """
    Inverted Transformer for ECG.
    Treats leads as tokens and compressed temporal features as token embeddings.
    """

    def __init__(
        self,
        num_leads: int = 7,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lead_embedding = nn.Parameter(torch.randn(1, num_leads, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.lead_embedding
        x = self.transformer(x)
        x = self.norm(x)
        return x


class LSTMAggregator(nn.Module):
    """Bidirectional LSTM over lead tokens."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        return torch.cat([h_forward, h_backward], dim=-1)


class UncertaintyHead(nn.Module):
    """Classifier head with explicit MC dropout support."""

    def __init__(self, in_features: int, num_classes: int = 13, dropout_p: float = 0.3):
        super().__init__()
        hidden = max(8, in_features // 2)
        self.fc1 = nn.Linear(in_features, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        if mc_dropout:
            x = F.dropout(x, p=self.dropout.p, training=True)
        else:
            x = self.dropout(x)
        x = self.fc2(x)
        return x


class PeriECGRiskNet(nn.Module):
    """Complete PeriECG-RiskNet model."""

    def __init__(
        self,
        num_classes: int = 13,
        num_leads: int = 7,
        signal_length: int = 5000,
        sampling_rate: int = 500,
        mc_dropout: bool = True,
        mc_samples: int = 50,
        cnn_filters: int = 64,
        cnn_depth: int = 4,
        d_model: int = 256,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        uncertainty_dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_leads = num_leads
        self.signal_length = signal_length
        self.sampling_rate = sampling_rate
        self.mc_dropout = mc_dropout
        self.mc_samples = mc_samples

        self.cnn = CNNBackbone(
            in_channels=num_leads,
            base_filters=cnn_filters,
            depth=cnn_depth,
            dropout=dropout,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, num_leads, signal_length)
            dummy_out = self.cnn(dummy)
            _, cnn_out_ch, cnn_out_t = dummy_out.shape
        self.cnn_out_ch = cnn_out_ch
        self.cnn_out_t = cnn_out_t

        self.temporal_pool = nn.AdaptiveAvgPool1d(d_model)
        self.lead_attn = LeadAwareAttention(
            embed_dim=d_model,
            num_leads=num_leads,
            num_heads=transformer_heads,
            dropout=dropout,
        )
        self.itransformer = iTransformerEncoder(
            num_leads=num_leads,
            d_model=d_model,
            n_layers=transformer_layers,
            n_heads=transformer_heads,
            dropout=dropout,
        )
        self.lstm = LSTMAggregator(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
        )
        self.head = UncertaintyHead(
            in_features=2 * lstm_hidden,
            num_classes=num_classes,
            dropout_p=uncertainty_dropout,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _ensure_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected ECG input with 3 dims, got shape={tuple(x.shape)}")
        if x.shape[1] == self.num_leads:
            return x
        if x.shape[-1] == self.num_leads:
            return x.permute(0, 2, 1)
        raise ValueError(
            f"Input must be (B, {self.num_leads}, T) or (B, T, {self.num_leads}), got {tuple(x.shape)}"
        )

    def _channels_to_lead_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_pool(x)  # (B, C, d_model)
        b, c, d = x.shape
        if c % self.num_leads != 0:
            pad = self.num_leads - (c % self.num_leads)
            x = F.pad(x, (0, 0, 0, pad))
            c += pad
        group = c // self.num_leads
        x = x.view(b, self.num_leads, group, d).mean(dim=2)
        return x  # (B, num_leads, d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_input_shape(x)
        x = self.cnn(x)
        x = self._channels_to_lead_tokens(x)
        x, _ = self.lead_attn(x)
        x = self.itransformer(x)
        x = self.lstm(x)
        return x

    def forward(
        self, x: torch.Tensor, return_uncertainty: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        logits = self.head(features, mc_dropout=False)
        if not return_uncertainty:
            return logits

        probs = torch.sigmoid(logits)
        entropy = -(
            probs * torch.log(probs + 1e-8)
            + (1.0 - probs) * torch.log(1.0 - probs + 1e-8)
        ).mean(dim=-1)
        normalized_entropy = entropy / math.log(2.0)
        return logits, normalized_entropy

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        if was_training:
            self.train()
        return probs

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor, mc_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mc_samples is None:
            mc_samples = self.mc_samples
        if mc_samples < 1:
            raise ValueError("mc_samples must be >= 1")

        x = self._ensure_input_shape(x)
        was_training = self.training
        self.eval()

        features = self.encode(x)
        logits_list = []
        for _ in range(mc_samples):
            logits = self.head(features, mc_dropout=True)
            logits_list.append(logits)

        logits_stack = torch.stack(logits_list, dim=0)  # (T, B, C)
        probs_stack = torch.sigmoid(logits_stack)       # multi-label probabilities

        mean_probs = probs_stack.mean(dim=0)
        std_probs = probs_stack.std(dim=0)

        mean_entropy = -(
            mean_probs * torch.log(mean_probs + 1e-8)
            + (1.0 - mean_probs) * torch.log(1.0 - mean_probs + 1e-8)
        ).mean(dim=-1)
        sample_entropy = -(
            probs_stack * torch.log(probs_stack + 1e-8)
            + (1.0 - probs_stack) * torch.log(1.0 - probs_stack + 1e-8)
        ).mean(dim=-1)
        mutual_info = mean_entropy - sample_entropy.mean(dim=0)
        normalized_uncertainty = mutual_info / math.log(2.0)

        if was_training:
            self.train()
        return mean_probs, normalized_uncertainty, std_probs

    @torch.no_grad()
    def get_attention_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()

        x = self._ensure_input_shape(x)
        x_enc = self.cnn(x)
        x_enc = self._channels_to_lead_tokens(x_enc)
        _, attn_weights = self.lead_attn(x_enc)

        if was_training:
            self.train()
        return {"lead_attention": attn_weights}


def build_model(config: Optional[dict] = None) -> PeriECGRiskNet:
    default_config = {
        "num_classes": 13,
        "num_leads": 7,
        "signal_length": 5000,
        "sampling_rate": 500,
        "mc_dropout": True,
        "mc_samples": 50,
        "cnn_filters": 64,
        "cnn_depth": 4,
        "d_model": 256,
        "transformer_layers": 4,
        "transformer_heads": 8,
        "lstm_hidden": 128,
        "lstm_layers": 2,
        "dropout": 0.1,
        "uncertainty_dropout": 0.3,
    }
    if config is not None:
        default_config.update(config)
    return PeriECGRiskNet(**default_config)


if __name__ == "__main__":
    torch.manual_seed(42)

    model = PeriECGRiskNet(num_classes=13, num_leads=7, signal_length=5000)
    x = torch.randn(4, 7, 5000)

    logits = model(x)
    print(f"Logits shape: {tuple(logits.shape)}")

    logits2, entropy = model(x, return_uncertainty=True)
    print(f"Forward+uncertainty logits shape: {tuple(logits2.shape)}")
    print(f"Forward uncertainty shape: {tuple(entropy.shape)}")

    mean_probs, uncertainty, std_probs = model.predict_with_uncertainty(x, mc_samples=10)
    print(f"Mean probs shape: {tuple(mean_probs.shape)}")
    print(f"Uncertainty shape: {tuple(uncertainty.shape)}")
    print(f"Std probs shape: {tuple(std_probs.shape)}")

    attn = model.get_attention_maps(x)
    print(f"Lead attention shape: {tuple(attn['lead_attention'].shape)}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
