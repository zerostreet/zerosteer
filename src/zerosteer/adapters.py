"""Ticker-agnostic timeseries encoder, calendar encoder, dynamic z_mm adapter, and fusion layer.

Ticker-Agnostic 시계열 인코더, 금융 캘린더 인코더, 가변 차원 동적 임베딩 어댑터 및 관측 융합 레이어입니다.
"""

import math
from collections.abc import Sequence
from datetime import date, datetime
from typing import Literal

import torch
from torch import nn

from zerosteer.moe import RMSNorm

CALENDAR_FEATURE_DIM = 10


def _parse_timestamp(value: datetime | date | str) -> date:
    """Normalize datetime, date, or ISO string into a calendar date.

    datetime, date 또는 ISO 문자열을 표준 캘린더 날짜 객체로 변환합니다.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return date.fromisoformat(text[:10])


def extract_calendar_features(timestamps: Sequence[datetime | date | str]) -> torch.Tensor:
    """Extract deterministic 10-dim multi-periodic financial calendar harmonics.

    결정론적 10차원 다주기(요일/분기/연간) 금융 캘린더 고조파 피처를 추출합니다.
    """
    if not timestamps:
        return torch.empty((0, CALENDAR_FEATURE_DIM), dtype=torch.float32)

    rows: list[list[float]] = []
    for raw_ts in timestamps:
        dt = _parse_timestamp(raw_ts)
        weekday = float(min(dt.weekday(), 4))
        phi_week = 2.0 * math.pi * (weekday / 5.0)

        q_idx = (dt.month - 1) // 3
        q_start_month = q_idx * 3 + 1
        q_start = date(dt.year, q_start_month, 1)
        if q_start_month == 10:
            q_end = date(dt.year + 1, 1, 1)
        else:
            q_end = date(dt.year, q_start_month + 3, 1)
        quarter_progress = min(max((dt - q_start).days / max((q_end - q_start).days, 1), 0.0), 1.0)

        y_start = date(dt.year, 1, 1)
        y_end = date(dt.year + 1, 1, 1)
        year_progress = min(max((dt - y_start).days / max((y_end - y_start).days, 1), 0.0), 1.0)

        phi_q1 = 2.0 * math.pi * quarter_progress
        phi_q2 = 4.0 * math.pi * quarter_progress
        phi_y1 = 2.0 * math.pi * year_progress
        phi_y2 = 4.0 * math.pi * year_progress

        rows.append(
            [
                math.sin(phi_week),
                math.cos(phi_week),
                math.sin(phi_q1),
                math.cos(phi_q1),
                math.sin(phi_q2),
                math.cos(phi_q2),
                math.sin(phi_y1),
                math.cos(phi_y1),
                math.sin(phi_y2),
                math.cos(phi_y2),
            ]
        )

    return torch.tensor(rows, dtype=torch.float32)


class FinancialCalendarEncoder(nn.Module):
    """Fixed harmonic projection with learnable gate for financial calendar conditioning.

    고정 다주기 고조파 투영 행렬과 학습 가능한 게이트를 결합한 금융 캘린더 인코더입니다.
    """

    proj_matrix: torch.Tensor

    def __init__(
        self,
        d_model: int,
        calendar_dim: int = CALENDAR_FEATURE_DIM,
        init_gate: float = 0.05,
    ) -> None:
        super().__init__()
        proj = torch.zeros((calendar_dim, d_model), dtype=torch.float32)
        for i in range(calendar_dim):
            for j in range(d_model):
                freq = 1.0 / (10000.0 ** (2.0 * (j // 2) / d_model))
                angle = (i + 1) * freq
                proj[i, j] = math.sin(angle) if j % 2 == 0 else math.cos(angle)
        self.register_buffer("proj_matrix", proj)
        self.norm = RMSNorm(d_model)
        self.gate = nn.Parameter(torch.tensor(init_gate, dtype=torch.float32))

    def forward(self, calendar_feats: torch.Tensor) -> torch.Tensor:
        """Project (B, 10) or (B, N, 10) calendar features into gated (B, 1|N, d_model).

        (B, 10) 또는 (B, N, 10) 캘린더 피처를 게이트된 (B, 1|N, d_model) 임베딩으로 변환합니다.
        """
        proj_mat = self.proj_matrix.to(device=calendar_feats.device, dtype=calendar_feats.dtype)
        projected = self.norm(calendar_feats @ proj_mat) * self.gate
        if projected.ndim == 2:
            return projected.unsqueeze(1)
        return projected


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
                    RMSNorm(d_model),
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
            drop_mask = torch.rand((bsz, 1, 1), device=projected.device) < self.full_blackout_prob
            projected = torch.where(drop_mask, mask_tokens, projected)

        return projected


class TimeseriesEncoder(nn.Module):
    """Ticker-agnostic encoder with RevIN, multi-scale depthwise TCN, and window-stats FiLM.

    RevIN 정규화, 다중 스케일 깊이별 TCN 및 윈도우 통계 FiLM을 결합한 Ticker-Agnostic 시계열 인코더입니다.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        dropout: float = 0.1,
        *,
        use_revin_multiscale: bool = True,
        temporal_kernel_sizes: tuple[int, ...] = (3, 7, 15),
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.use_revin_multiscale = use_revin_multiscale
        self.max_seq_len = max_seq_len

        if use_revin_multiscale:
            if not temporal_kernel_sizes:
                raise ValueError("temporal_kernel_sizes must contain at least one kernel size.")
            self.stem = nn.Conv1d(in_features, d_model, kernel_size=1)
            self.ms_convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        d_model,
                        d_model,
                        kernel_size=k,
                        padding=k // 2,
                        groups=d_model,
                    )
                    for k in temporal_kernel_sizes
                ]
            )
            self.ms_mix = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
            )
            self.recency_pos = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
            nn.init.normal_(self.recency_pos, mean=0.0, std=0.02)
            self.norm = RMSNorm(d_model)
            self.attn_pool = nn.Linear(d_model, 1)
            self.window_film = nn.Linear(in_features * 2, d_model * 2)
            self.out_norm = RMSNorm(d_model)
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_features, d_model, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.norm = nn.LayerNorm(d_model)
            self.attn_pool = nn.Linear(d_model, 1)

    def _forward_v1(self, ohlcv_seq: torch.Tensor) -> torch.Tensor:
        bsz, n_assets, seq_len, n_feat = ohlcv_seq.shape
        flat = ohlcv_seq.reshape(bsz * n_assets, seq_len, n_feat).transpose(1, 2)
        hidden = self.conv(flat).transpose(1, 2)
        hidden = self.norm(hidden)
        weights = torch.softmax(self.attn_pool(hidden), dim=1)
        pooled = (hidden * weights).sum(dim=1)
        return pooled.view(bsz, n_assets, -1)

    def forward(self, ohlcv_seq: torch.Tensor) -> torch.Tensor:
        """Encode input (B, N, T, F) into per-asset embeddings (B, N, d_model).

        입력 (B, N, T, F)를 자산별 시계열 표현 (B, N, d_model)로 압축합니다.
        """
        if not self.use_revin_multiscale:
            return self._forward_v1(ohlcv_seq)

        bsz, n_assets, seq_len, n_feat = ohlcv_seq.shape
        x = ohlcv_seq.reshape(bsz * n_assets, seq_len, n_feat)

        # unbiased=False + eps inside sqrt ensures constant assets (e.g. CASH) remain finite 0.0.
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        std = torch.sqrt(var + 1e-5)
        x_norm = (x - mean) / std
        window_stats = torch.cat([mean.squeeze(1), torch.log1p(std.squeeze(1))], dim=-1)

        stem = self.stem(x_norm.transpose(1, 2))
        branch_outputs = []
        for branch in self.ms_convs:
            out = branch(stem)
            if out.shape[-1] != seq_len:
                out = out[:, :, :seq_len]
            branch_outputs.append(out)

        ms_hidden = torch.stack(branch_outputs, dim=0).mean(dim=0).transpose(1, 2)
        hidden = stem.transpose(1, 2) + self.ms_mix(self.norm(ms_hidden))

        # Flipped recency index: index 0 always corresponds to the most recent bar (t-1).
        flip_idx = torch.arange(seq_len - 1, -1, -1, device=hidden.device).clamp(
            max=self.max_seq_len - 1
        )
        hidden = self.norm(hidden + self.recency_pos[:, flip_idx, :])

        weights = torch.softmax(self.attn_pool(hidden), dim=1)
        pooled = (hidden * weights).sum(dim=1)

        gamma_w, beta_w = self.window_film(window_stats).chunk(2, dim=-1)
        pooled = self.out_norm(pooled * (1.0 + torch.tanh(gamma_w)) + beta_w)
        return pooled.view(bsz, n_assets, -1)


class ObservationFusionLayer(nn.Module):
    """Single-pass layer fusing OHLCV features, dynamic-dim z_mm, and calendar embeddings.

    시계열(OHLCV), 가변 차원 z_mm 어댑터 및 금융 캘린더 임베딩을 단일 패스로 융합하는 레이어입니다.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        supported_dims: tuple[int, ...] = (32, 128, 256),
        fusion_mode: Literal["film", "gated"] = "film",
        full_blackout_prob: float = 0.15,
        dropout: float = 0.1,
        *,
        use_revin_multiscale: bool = True,
        temporal_kernel_sizes: tuple[int, ...] = (3, 7, 15),
        use_calendar_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.fusion_mode = fusion_mode
        self.ts_encoder = TimeseriesEncoder(
            in_features=in_features,
            d_model=d_model,
            dropout=dropout,
            use_revin_multiscale=use_revin_multiscale,
            temporal_kernel_sizes=temporal_kernel_sizes,
        )
        self.signal_adapter = DynamicEmbeddingAdapter(
            d_model=d_model,
            supported_dims=supported_dims,
            full_blackout_prob=full_blackout_prob,
        )
        self.calendar_encoder: FinancialCalendarEncoder | None = (
            FinancialCalendarEncoder(d_model=d_model) if use_calendar_encoder else None
        )
        self.film_gamma = nn.Linear(d_model, d_model)
        self.film_beta = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.obs_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.out_norm = RMSNorm(d_model)

    def forward(
        self,
        ohlcv_seq: torch.Tensor,
        z_mm: torch.Tensor | None = None,
        is_blackout: torch.Tensor | None = None,
        calendar_feats: torch.Tensor | None = None,
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

        fused = fused + self.obs_type_emb
        if self.calendar_encoder is not None and calendar_feats is not None:
            fused = fused + self.calendar_encoder(calendar_feats)

        return self.out_norm(fused)
