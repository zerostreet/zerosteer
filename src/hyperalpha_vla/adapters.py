"""Ticker-agnostic timeseries encoder, dynamic z_mm adapter, and fusion layer.

Ticker-Agnostic 시계열 인코더, 가변 차원 동적 임베딩 어댑터 및 관측 융합 레이어입니다.
"""

from typing import Literal

import torch
from torch import nn


class DynamicEmbeddingAdapter(nn.Module):
    """Adapter projecting variable-dimension z_mm (0/32/128/256d) with blackout fallback.

    가변 차원(0/32/128/256d) 멀티모달 임베딩(z_mm) 투영 및 블랙아웃 폴백 어댑터입니다.
    """

    def __init__(
        self,
        d_model: int,
        supported_dims: tuple[int, ...] = (32, 128, 256),
        full_blackout_prob: float = 0.15,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.supported_dims = tuple(sorted(set(supported_dims)))
        self.full_blackout_prob = full_blackout_prob

        self.projections = nn.ModuleDict(
            {
                str(dim): nn.Sequential(
                    nn.Linear(dim, d_model),
                    nn.LayerNorm(d_model),
                    nn.SiLU(),
                )
                for dim in self.supported_dims
            }
        )
        self.mask_signal_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_signal_token, mean=0.0, std=0.02)

    def forward(
        self,
        z_mm: torch.Tensor | None,
        is_blackout: torch.Tensor | None = None,
        batch_size: int | None = None,
        n_assets: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Project variable-dimension z_mm (B, N, D_z) or None into (B, N, d_model).

        가변 차원 z_mm (B, N, D_z) 또는 결측(None)을 (B, N, d_model) 표현으로 변환합니다.
        """
        if z_mm is None or z_mm.shape[-1] == 0:
            if batch_size is None or n_assets is None:
                raise ValueError(
                    "batch_size and n_assets must be provided when z_mm is None or dim=0."
                )
            return self.mask_signal_token.expand(batch_size, n_assets, self.d_model).to(
                device=device, dtype=dtype
            )

        bsz, num_assets, sig_dim = z_mm.shape
        dim_key = str(sig_dim)
        if dim_key not in self.projections:
            raise ValueError(
                f"Unsupported z_mm embedding dim={sig_dim}. Supported: {self.supported_dims}."
            )

        projected = self.projections[dim_key](z_mm)
        mask_tokens = self.mask_signal_token.expand(bsz, num_assets, self.d_model)

        if is_blackout is not None:
            blk = is_blackout.to(device=projected.device, dtype=torch.bool).unsqueeze(-1)
            projected = torch.where(blk, mask_tokens, projected)

        if self.training and self.full_blackout_prob > 0.0:
            drop_mask = (
                torch.rand((bsz, 1, 1), device=projected.device) < self.full_blackout_prob
            )
            projected = torch.where(drop_mask, mask_tokens, projected)

        return projected


class TimeseriesEncoder(nn.Module):
    """Ticker-agnostic 1D-CNN + attention-pooling encoder over OHLCV sequences.

    고정 종목 매핑 없이 시계열 텐서를 인코딩하는 Ticker-Agnostic 인코더입니다.
    """

    def __init__(self, in_features: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(d_model)
        self.attn_pool = nn.Linear(d_model, 1)

    def forward(self, ohlcv_seq: torch.Tensor) -> torch.Tensor:
        """Encode input (B, N, T, F) into per-asset embeddings (B, N, d_model).

        입력 (B, N, T, F)를 자산별 시계열 표현 (B, N, d_model)로 압축합니다.
        """
        bsz, n_assets, seq_len, n_feat = ohlcv_seq.shape
        flat = ohlcv_seq.reshape(bsz * n_assets, seq_len, n_feat).transpose(1, 2)
        hidden = self.conv(flat).transpose(1, 2)
        hidden = self.norm(hidden)
        weights = torch.softmax(self.attn_pool(hidden), dim=1)
        pooled = (hidden * weights).sum(dim=1)
        return pooled.view(bsz, n_assets, -1)


class ObservationFusionLayer(nn.Module):
    """Single-pass layer fusing OHLCV features with dynamic-dimension z_mm embeddings.

    시계열(OHLCV) 인코더와 가변 차원 z_mm 어댑터를 단일 패스로 융합하는 레이어입니다.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        supported_dims: tuple[int, ...] = (32, 128, 256),
        fusion_mode: Literal["film", "gated"] = "film",
        full_blackout_prob: float = 0.15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.fusion_mode = fusion_mode
        self.ts_encoder = TimeseriesEncoder(
            in_features=in_features, d_model=d_model, dropout=dropout
        )
        self.signal_adapter = DynamicEmbeddingAdapter(
            d_model=d_model,
            supported_dims=supported_dims,
            full_blackout_prob=full_blackout_prob,
        )
        self.film_gamma = nn.Linear(d_model, d_model)
        self.film_beta = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.obs_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        ohlcv_seq: torch.Tensor,
        z_mm: torch.Tensor | None = None,
        is_blackout: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run single-pass fusion once per batch to emit (B, N, d_model) observation tokens.

        배치당 1회만 실행되어 (B, N, d_model) 관측 토큰 시퀀스를 반환합니다.
        """
        bsz, n_assets = ohlcv_seq.shape[:2]
        ts_emb = self.ts_encoder(ohlcv_seq)
        sig_emb = self.signal_adapter(
            z_mm=z_mm,
            is_blackout=is_blackout,
            batch_size=bsz,
            n_assets=n_assets,
            device=ts_emb.device,
            dtype=ts_emb.dtype,
        )

        if self.fusion_mode == "film":
            gamma = self.film_gamma(sig_emb)
            beta = self.film_beta(sig_emb)
            fused = ts_emb * (1.0 + torch.tanh(gamma)) + beta
        else:
            gate = torch.sigmoid(self.gate_proj(torch.cat([ts_emb, sig_emb], dim=-1)))
            fused = gate * sig_emb + (1.0 - gate) * ts_emb

        return self.out_norm(fused + self.obs_type_emb)
