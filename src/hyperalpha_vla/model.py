"""Model B (HyperAlpha-VLA): Cross-Asset 관계형 백본, H=20 Action Chunking 및 Policy Head 분리 모듈."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from zerostreet_schema import HorizonMaskMode, UserPolicyConfig

from hyperalpha_vla.activations import entmax15, sparsemax
from hyperalpha_vla.adapters import ObservationFusionLayer
from hyperalpha_vla.config import HyperAlphaVLAConfig
from hyperalpha_vla.moe import MoETransformerBackbone


@dataclass
class HyperAlphaVLAOutputs:
    """HyperAlpha-VLA 다기간 행동 청킹 추론 결과 데이터클래스."""

    weights_long: torch.Tensor
    weights_long_trajectory: torch.Tensor
    logits_long_trajectory: torch.Tensor
    final_embs_obs: torch.Tensor
    final_embs_long_trajectory: torch.Tensor


def build_horizon_attention_mask(
    obs_num: int,
    step_asset_num: int,
    horizon: int,
    mode: HorizonMaskMode | str = HorizonMaskMode.BIDIRECTIONAL_PORTFOLIO,
    obs_attend_p0: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """관측 토큰(obs_embs)의 미래 오염을 차단하고 P_0..P_{H-1} 상호작용을 제어하는 2D bool 마스크를 생성합니다.

    PyTorch MultiheadAttention 규격에 따라 True는 참조 차단(Masked Out), False는 참조 허용입니다.
    """
    if step_asset_num == 0 or (horizon <= 1 and obs_attend_p0):
        return None

    total_len = obs_num + horizon * step_asset_num
    mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)

    if obs_attend_p0:
        if horizon > 1:
            mask[:obs_num, obs_num + step_asset_num :] = True
    else:
        mask[:obs_num, obs_num:] = True

    resolved_mode = HorizonMaskMode(str(mode).lower())
    if resolved_mode == HorizonMaskMode.CAUSAL_PORTFOLIO:
        for k in range(horizon):
            q_start = obs_num + k * step_asset_num
            q_end = q_start + step_asset_num
            k_mask_start = obs_num + (k + 1) * step_asset_num
            if k_mask_start < total_len:
                mask[q_start:q_end, k_mask_start:] = True
    elif resolved_mode == HorizonMaskMode.BACKWARD_DP:
        for k in range(1, horizon):
            q_start = obs_num + k * step_asset_num
            q_end = q_start + step_asset_num
            k_mask_end = obs_num + k * step_asset_num
            mask[q_start:q_end, obs_num:k_mask_end] = True

    return mask


class HyperAlphaVLABackbone(nn.Module):
    """오픈소스(MIT) 공개 범위인 Cross-Asset 관계형 백본.

    회사가 학습한 종목별 매수/매도 Policy Head 가중치를 0바이트 포함하며,
    Future-Label-Free 관계형 사전학습 가중치만 저장·배포합니다.
    """

    def __init__(self, config: HyperAlphaVLAConfig | None = None) -> None:
        super().__init__()
        self.config = config or HyperAlphaVLAConfig()
        d_model = self.config.d_model

        self.obs_fusion = ObservationFusionLayer(
            in_features=self.config.timeseries_features,
            d_model=d_model,
            supported_dims=self.config.supported_signal_dims,
            fusion_mode=self.config.signal_fusion_mode,
            full_blackout_prob=self.config.full_blackout_prob,
            dropout=self.config.dropout,
        )
        self.asset_query_proj = nn.Linear(d_model, d_model)
        self.horizon_encodings = nn.Embedding(self.config.max_horizon_num, d_model)
        with torch.no_grad():
            self.horizon_encodings.weight[0].zero_()

        self.weight_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.transformer = MoETransformerBackbone(
            d_model=d_model,
            nhead=self.config.nhead,
            nlayers=self.config.nlayers,
            n_dense_layers=self.config.n_dense_layers,
            n_routed_experts=self.config.n_routed_experts,
            n_activated_experts=self.config.n_activated_experts,
            inter_dim=self.config.expert_inter_dim,
            dropout=self.config.dropout,
        )

    def encode_observations(
        self,
        ohlcv_seq: torch.Tensor,
        z_mm: torch.Tensor | None = None,
        is_blackout: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """배치당 1회 시계열과 동적 멀티모달 임베딩을 융합합니다."""
        return self.obs_fusion(ohlcv_seq=ohlcv_seq, z_mm=z_mm, is_blackout=is_blackout)

    def forward_backbone(
        self,
        ohlcv_seq: torch.Tensor,
        z_mm: torch.Tensor | None = None,
        is_blackout: torch.Tensor | None = None,
        prev_weights: torch.Tensor | None = None,
        horizon: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """관측 융합 -> H 스텝 자산 블록 생성(P_0 전용 prev_weights) -> MoE 백본 통과."""
        obs_embs = self.encode_observations(ohlcv_seq, z_mm=z_mm, is_blackout=is_blackout)
        bsz, n_assets, d_model = obs_embs.shape
        h_steps = int(horizon if horizon is not None else self.config.portfolio_horizon)
        if h_steps < 1 or h_steps > self.config.max_horizon_num:
            raise ValueError(
                f"horizon ({h_steps}) must be between 1 and max_horizon_num "
                f"({self.config.max_horizon_num})."
            )

        base_asset = self.asset_query_proj(obs_embs)  # (B, N, D)
        h_idx = torch.arange(h_steps, device=obs_embs.device)
        h_emb = self.horizon_encodings(h_idx).view(1, h_steps, 1, d_model)
        p_blocks = base_asset.unsqueeze(1) + h_emb  # (B, H, N, D)

        if prev_weights is not None:
            pw = prev_weights.to(device=obs_embs.device, dtype=obs_embs.dtype).unsqueeze(-1)
            pw_emb = self.weight_encoder(pw)  # (B, N, D)
            p_blocks = p_blocks.clone()
            p_blocks[:, 0, :, :] = p_blocks[:, 0, :, :] + pw_emb

        p_flat = p_blocks.reshape(bsz, h_steps * n_assets, d_model)
        seq = torch.cat([obs_embs, p_flat], dim=1)

        attn_mask = build_horizon_attention_mask(
            obs_num=n_assets,
            step_asset_num=n_assets,
            horizon=h_steps,
            mode=self.config.horizon_mask_mode,
            obs_attend_p0=self.config.obs_attend_p0,
            device=obs_embs.device,
        )
        encoded = self.transformer(seq, attn_mask=attn_mask)
        final_obs = encoded[:, :n_assets, :]
        final_traj = encoded[:, n_assets:, :].view(bsz, h_steps, n_assets, d_model)
        return final_obs, final_traj

    def save_backbone(self, path: str | Path) -> Path:
        """Policy Head 가중치를 0바이트 포함한 순수 관계형 백본만 저장합니다."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.model_dump(mode="json"),
            "backbone_state_dict": self.state_dict(),
        }
        torch.save(payload, out_path)
        return out_path

    @classmethod
    def load_backbone(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "HyperAlphaVLABackbone":
        """저장된 관계형 백본 체크포인트를 로드합니다."""
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
        cfg = HyperAlphaVLAConfig.model_validate(payload["config"])
        model = cls(cfg)
        model.load_state_dict(payload["backbone_state_dict"], strict=True)
        return model


class WeightAllocatorHead(nn.Module):
    """사용자 에이전트가 개별 컴파일·소유하는 Action-Chunking 비중 할당기(Policy Head)."""

    def __init__(
        self,
        d_model: int,
        weights_output_mode: Literal["entmax15", "sparsemax", "softmax"] = "entmax15",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.weights_output_mode = weights_output_mode
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        embs_traj: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, H, N, D) 임베딩 궤적을 희소 심플렉스 비중 (B, H, N)과 마스킹 로짓으로 변환합니다."""
        logits_traj = self.head(embs_traj.contiguous()).squeeze(-1).float()
        if valid_mask is not None:
            v_mask = valid_mask.to(device=logits_traj.device, dtype=torch.float32)
            if v_mask.dim() == 2:
                v_mask = v_mask.unsqueeze(1)
            logits_traj = logits_traj + (1.0 - v_mask) * -1e9

        if self.weights_output_mode == "entmax15":
            weights_traj = entmax15(logits_traj, dim=-1)
        elif self.weights_output_mode == "sparsemax":
            weights_traj = sparsemax(logits_traj, dim=-1)
        else:
            weights_traj = logits_traj.softmax(dim=-1)

        return weights_traj, logits_traj


class HyperAlphaVLA(nn.Module):
    """관계형 백본(Model B)과 사용자 컴파일 전용 Policy Head 결합 모델.

    자본시장법 규제 안전 경계:
        `user_config` 기본값은 반드시 `None`이며, 미입력 시 즉시 `ValueError`를 발생시킵니다.
    """

    def __init__(
        self,
        backbone: HyperAlphaVLABackbone | None = None,
        user_config: UserPolicyConfig | None = None,
    ) -> None:
        if user_config is None:
            raise ValueError(
                "user_config is required to instantiate HyperAlphaVLA. "
                "ZeroStreet distributes 0 bytes of pre-trained policy head weights."
            )
        super().__init__()
        self.backbone = backbone or HyperAlphaVLABackbone()
        self.user_config = user_config
        self.policy_head = WeightAllocatorHead(
            d_model=self.backbone.config.d_model,
            weights_output_mode=user_config.weights_output_mode,
            dropout=self.backbone.config.dropout,
        )

    def forward(
        self,
        ohlcv_seq: torch.Tensor,
        z_mm: torch.Tensor | None = None,
        is_blackout: torch.Tensor | None = None,
        prev_weights: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        horizon: int | None = None,
    ) -> HyperAlphaVLAOutputs:
        eff_horizon = int(horizon if horizon is not None else self.user_config.portfolio_horizon)
        final_obs, final_traj = self.backbone.forward_backbone(
            ohlcv_seq=ohlcv_seq,
            z_mm=z_mm,
            is_blackout=is_blackout,
            prev_weights=prev_weights,
            horizon=eff_horizon,
        )
        weights_traj, logits_traj = self.policy_head(final_traj, valid_mask=valid_mask)
        return HyperAlphaVLAOutputs(
            weights_long=weights_traj[:, 0, :],
            weights_long_trajectory=weights_traj,
            logits_long_trajectory=logits_traj,
            final_embs_obs=final_obs,
            final_embs_long_trajectory=final_traj,
        )

    @staticmethod
    def blend_temporal_ensemble(
        trajectory_history: list[torch.Tensor],
        decay_beta: float = 0.5,
    ) -> torch.Tensor:
        """과거 추론 궤적들과 당일 추론 비중을 지수 감쇠 가중치로 결합하는 시간축 앙상블."""
        if not trajectory_history:
            raise ValueError("trajectory_history must not be empty.")

        blended = torch.zeros_like(trajectory_history[0][:, 0, :])
        total_weight = 0.0
        for lag_m, traj in enumerate(trajectory_history):
            if traj.dim() == 3 and lag_m < traj.shape[1]:
                w_m = float(decay_beta**lag_m)
                blended = blended + w_m * traj[:, lag_m, :]
                total_weight += w_m

        blended = blended / max(total_weight, 1e-8)
        return blended / blended.sum(dim=-1, keepdim=True).clamp(min=1e-8)
