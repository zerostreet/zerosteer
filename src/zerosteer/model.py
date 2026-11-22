"""ZeroSteer: Cross-Asset backbone, H=20 Action Chunking & Policy Head split.

ZeroSteer: Cross-Asset 관계형 백본, 다기간 행동 청킹 및 Policy Head 분리 모듈입니다.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from zerostreet_schema import HorizonMaskMode, UserPolicyConfig

from zerosteer.activations import entmax15, sparsemax
from zerosteer.adapters import ObservationFusionLayer
from zerosteer.config import ZeroSteerConfig
from zerosteer.moe import MoETransformerBackbone, RMSNorm, TopKMoELayer

BACKBONE_ARTIFACT_VERSION = "zerosteer-backbone-v2"


def _compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest for a checkpoint artifact on disk.

    디스크에 저장된 체크포인트 아티팩트의 16진수 SHA-256 해시를 계산합니다.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path_for(path: Path) -> Path:
    """Return sibling manifest path `<stem>.manifest.json` for a checkpoint file.

    체크포인트 파일에 대응하는 동위 매니페스트 경로(`<stem>.manifest.json`)를 반환합니다.
    """
    return path.with_name(f"{path.stem}.manifest.json")


def _verify_manifest_sha256(path: Path) -> None:
    """Verify artifact SHA-256 against its sibling manifest when present.

    동위 매니페스트 파일이 존재할 경우 아티팩트의 SHA-256 무결성을 검증합니다.
    """
    manifest_path = _manifest_path_for(path)
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("sha256")
    actual = _compute_file_sha256(path)
    if expected != actual:
        raise ValueError(
            f"Checkpoint SHA-256 integrity verification failed for {path}: "
            f"expected {expected}, got {actual}."
        )


@dataclass
class ZeroSteerOutputs:
    """Inference output container for ZeroSteer multi-horizon action chunking.

    ZeroSteer 다기간 행동 청킹 추론 결과 데이터클래스입니다.
    """

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
    """Build 2D bool attention mask blocking obs leakage while shaping P_0..P_{H-1} attention.

    Following PyTorch MultiheadAttention convention, True masks out attention.

    관측 토큰(obs_embs)의 미래 오염을 차단하고 P_0..P_{H-1} 상호작용을 제어하는 2D 마스크입니다.
    PyTorch MultiheadAttention 규격에 따라 True는 참조 차단을 의미합니다.
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


class ZeroSteerBackbone(nn.Module):
    """Open-source (MIT) Cross-Asset relational backbone with 0 bytes of trading policy weights.

    Stores and distributes only Future-Label-Free relational pretraining weights.

    오픈소스(MIT) 공개 범위인 Cross-Asset 관계형 백본으로 정책 가중치를 0바이트 포함하며,
    Future-Label-Free 관계형 사전학습 가중치만 저장·배포합니다.
    """

    def __init__(self, config: ZeroSteerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ZeroSteerConfig()
        d_model = self.config.d_model

        self.obs_fusion = ObservationFusionLayer(
            in_features=self.config.timeseries_features,
            d_model=d_model,
            supported_dims=self.config.supported_signal_dims,
            fusion_mode=self.config.signal_fusion_mode,
            full_blackout_prob=self.config.full_blackout_prob,
            dropout=self.config.dropout,
            use_revin_multiscale=self.config.use_revin_multiscale,
            temporal_kernel_sizes=self.config.temporal_kernel_sizes,
            use_calendar_encoder=self.config.use_calendar_encoder,
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
            use_qk_norm=self.config.use_qk_norm,
            ffn_activation=self.config.ffn_activation,
            use_aux_loss_free=self.config.use_aux_loss_free_moe,
            moe_score_func=self.config.moe_score_func,
            moe_bias_update_speed=self.config.moe_bias_update_speed,
            moe_route_scale=self.config.moe_route_scale,
        )

    def encode_observations(
        self,
        ohlcv_seq: torch.Tensor,
        z_mm: torch.Tensor | None = None,
        is_blackout: torch.Tensor | None = None,
        calendar_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse OHLCV sequences, dynamic multimodal embeddings, and optional calendar features.

        배치당 1회 시계열, 동적 멀티모달 임베딩 및 선택적 캘린더 피처를 융합합니다.
        """
        return self.obs_fusion(
            ohlcv_seq=ohlcv_seq,
            z_mm=z_mm,
            is_blackout=is_blackout,
            calendar_feats=calendar_feats,
        )

    def forward_backbone(
        self,
        ohlcv_seq: torch.Tensor,
        z_mm: torch.Tensor | None = None,
        is_blackout: torch.Tensor | None = None,
        prev_weights: torch.Tensor | None = None,
        horizon: int | None = None,
        calendar_feats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse observations, inject P_0 prev_weights, and run relational MoE backbone.

        관측 융합, P_0 전용 직전 비중 주입 및 MoE 관계형 백본 순전파를 실행합니다.
        """
        obs_embs = self.encode_observations(
            ohlcv_seq,
            z_mm=z_mm,
            is_blackout=is_blackout,
            calendar_feats=calendar_feats,
        )
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

    def update_gate_bias(self) -> None:
        """Update aux-loss-free MoE gate biases across the backbone transformer.

        백본 트랜스포머 전반의 Aux-Loss-Free MoE 게이트 바이어스를 갱신합니다.
        """
        self.transformer.update_gate_bias()

    def get_moe_load_stats(self) -> dict[str, dict[str, list[float] | list[int]]]:
        """Return per-layer MoE routing load counts and bias vectors.

        레이어별 MoE 라우팅 부하 카운트와 게이트 바이어스 벡터를 반환합니다.
        """
        return self.transformer.get_moe_load_stats()

    def save_backbone(self, path: str | Path) -> Path:
        """Persist pure relational backbone weights and write a SHA-256 sibling manifest.

        Policy Head 가중치를 0바이트 포함한 순수 백본 체크포인트와 SHA-256 매니페스트를 저장합니다.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        config_dump = self.config.model_dump(mode="json")
        payload = {
            "config": config_dump,
            "backbone_state_dict": self.state_dict(),
        }
        torch.save(payload, out_path)

        manifest = {
            "artifact_path": out_path.name,
            "sha256": _compute_file_sha256(out_path),
            "backbone_version": BACKBONE_ARTIFACT_VERSION,
            "has_policy_head": False,
            "config": config_dump,
        }
        _manifest_path_for(out_path).write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return out_path

    @classmethod
    def load_backbone(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
        verify_sha256: bool = True,
    ) -> "ZeroSteerBackbone":
        """Load saved relational backbone checkpoint after optional SHA-256 verification.

        선택적 SHA-256 무결성 검증 후 저장된 관계형 백본 체크포인트를 로드합니다.
        """
        in_path = Path(path)
        if verify_sha256:
            _verify_manifest_sha256(in_path)
        payload = torch.load(in_path, map_location=map_location, weights_only=True)
        cfg = ZeroSteerConfig.model_validate(payload["config"])
        model = cls(cfg)
        model.load_state_dict(payload["backbone_state_dict"], strict=True)
        return model


class WeightAllocatorHead(nn.Module):
    """Per-user compiled Action-Chunking Weight Allocator (Policy Head).

    사용자 에이전트가 개별 컴파일·소유하는 Action-Chunking 비중 할당기(Policy Head)입니다.
    """

    def __init__(
        self,
        d_model: int,
        weights_output_mode: Literal["entmax15", "sparsemax", "softmax"] = "entmax15",
        dropout: float = 0.1,
        *,
        use_moe_allocator: bool = False,
        n_routed_experts: int = 4,
        n_activated_experts: int = 2,
        inter_dim: int | None = None,
        ffn_activation: Literal["swiglu", "geglu"] = "swiglu",
        use_aux_loss_free: bool = True,
        moe_score_func: Literal["sigmoid", "softmax"] = "sigmoid",
        moe_bias_update_speed: float = 0.01,
        moe_route_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.weights_output_mode = weights_output_mode
        self.use_moe_allocator = use_moe_allocator

        if use_moe_allocator:
            resolved_inter = inter_dim if inter_dim is not None else max(d_model * 2, 32)
            self.norm: nn.Module = RMSNorm(d_model)
            self.head: nn.Module = TopKMoELayer(
                d_model=d_model,
                inter_dim=resolved_inter,
                n_routed_experts=n_routed_experts,
                n_activated_experts=n_activated_experts,
                dropout=dropout,
                out_dim=1,
                ffn_activation=ffn_activation,
                use_aux_loss_free=use_aux_loss_free,
                score_func=moe_score_func,
                bias_update_speed=moe_bias_update_speed,
                route_scale=moe_route_scale,
            )
        else:
            self.norm = nn.Identity()
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
        """Map (B, H, N, D) embeddings into sparse simplex weights (B, H, N) and logits.

        (B, H, N, D) 임베딩 궤적을 희소 심플렉스 비중 (B, H, N)과 마스킹 로짓으로 변환합니다.
        """
        logits_traj = self.head(self.norm(embs_traj.contiguous())).squeeze(-1).float()
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

    def update_gate_bias(self) -> None:
        """Update aux-loss-free router bias on the MoE allocator head if enabled.

        MoE 할당기 헤드가 활성화된 경우 Aux-Loss-Free 라우터 바이어스를 갱신합니다.
        """
        if isinstance(self.head, TopKMoELayer):
            self.head.update_gate_bias()

    def get_load_stats(self) -> dict[str, list[float] | list[int]]:
        """Return expert routing stats when the MoE allocator head is enabled.

        MoE 할당기 헤드가 활성화된 경우 전문가 라우팅 부하 통계를 반환합니다.
        """
        if isinstance(self.head, TopKMoELayer):
            return self.head.get_load_stats()
        return {}


class ZeroSteer(nn.Module):
    """Combined ZeroSteer relational backbone and user-compiled Policy Head model.

    Requires explicit `user_config` (`None` raises `ValueError` to prevent advisory defaults).

    ZeroSteer 관계형 백본과 사용자 컴파일 전용 Policy Head 결합 모델입니다.
    자본시장법 규제 준수를 위해 `user_config` 미입력 시 즉시 `ValueError`를 발생시킵니다.
    """

    def __init__(
        self,
        backbone: ZeroSteerBackbone | None = None,
        user_config: UserPolicyConfig | None = None,
    ) -> None:
        if user_config is None:
            raise ValueError(
                "user_config is required to instantiate ZeroSteer. "
                "ZeroStreet distributes 0 bytes of pre-trained policy head weights."
            )
        super().__init__()
        self.backbone = backbone or ZeroSteerBackbone()
        self.user_config = user_config
        cfg = self.backbone.config
        self.policy_head = WeightAllocatorHead(
            d_model=cfg.d_model,
            weights_output_mode=user_config.weights_output_mode,
            dropout=cfg.dropout,
            use_moe_allocator=cfg.use_moe_allocator,
            n_routed_experts=cfg.allocator_routed_experts,
            n_activated_experts=cfg.allocator_activated_experts,
            inter_dim=cfg.allocator_inter_dim,
            ffn_activation=cfg.ffn_activation,
            use_aux_loss_free=cfg.use_aux_loss_free_moe,
            moe_score_func=cfg.moe_score_func,
            moe_bias_update_speed=cfg.moe_bias_update_speed,
            moe_route_scale=cfg.moe_route_scale,
        )

    def forward(
        self,
        ohlcv_seq: torch.Tensor,
        z_mm: torch.Tensor | None = None,
        is_blackout: torch.Tensor | None = None,
        prev_weights: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        horizon: int | None = None,
        calendar_feats: torch.Tensor | None = None,
    ) -> ZeroSteerOutputs:
        """Compute multi-horizon portfolio weights and backbone embeddings in one forward pass.

        단일 순전파로 다기간 포트폴리오 비중 궤적과 백본 임베딩을 계산합니다.
        """
        eff_horizon = int(horizon if horizon is not None else self.user_config.portfolio_horizon)
        final_obs, final_traj = self.backbone.forward_backbone(
            ohlcv_seq=ohlcv_seq,
            z_mm=z_mm,
            is_blackout=is_blackout,
            prev_weights=prev_weights,
            horizon=eff_horizon,
            calendar_feats=calendar_feats,
        )
        weights_traj, logits_traj = self.policy_head(final_traj, valid_mask=valid_mask)
        return ZeroSteerOutputs(
            weights_long=weights_traj[:, 0, :],
            weights_long_trajectory=weights_traj,
            logits_long_trajectory=logits_traj,
            final_embs_obs=final_obs,
            final_embs_long_trajectory=final_traj,
        )

    def update_gate_bias(self, include_backbone: bool = True) -> None:
        """Update aux-loss-free MoE gate biases on policy head and optionally backbone.

        정책 헤드 및 선택적으로 백본의 Aux-Loss-Free MoE 게이트 바이어스를 갱신합니다.
        """
        if include_backbone:
            self.backbone.update_gate_bias()
        self.policy_head.update_gate_bias()

    def get_moe_load_stats(self) -> dict[str, Any]:
        """Return MoE routing load statistics across backbone layers and policy head.

        백본 레이어 및 정책 헤드의 MoE 라우팅 부하 통계를 반환합니다.
        """
        stats: dict[str, Any] = {"backbone": self.backbone.get_moe_load_stats()}
        head_stats = self.policy_head.get_load_stats()
        if head_stats:
            stats["policy_head"] = head_stats
        return stats

    def save_policy_head(self, path: str | Path) -> str:
        """Persist user-compiled policy head checkpoint with SHA-256 sibling manifest.

        사용자 컴파일 전용 정책 헤드 체크포인트와 SHA-256 동위 매니페스트를 저장합니다.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "user_config": self.user_config.model_dump(mode="json"),
            "backbone_config": self.backbone.config.model_dump(mode="json"),
            "policy_head_state_dict": self.policy_head.state_dict(),
        }
        torch.save(payload, out_path)
        sha256 = _compute_file_sha256(out_path)
        manifest = {
            "artifact_path": out_path.name,
            "sha256": sha256,
            "backbone_version": BACKBONE_ARTIFACT_VERSION,
            "has_policy_head": True,
            "weights_output_mode": self.user_config.weights_output_mode,
        }
        _manifest_path_for(out_path).write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return sha256

    def load_policy_head(
        self,
        path: str | Path,
        map_location: str | torch.device = "cpu",
        verify_sha256: bool = True,
    ) -> None:
        """Load compiled policy head weights after optional SHA-256 verification.

        선택적 SHA-256 무결성 검증 후 컴파일된 정책 헤드 가중치를 로드합니다.
        """
        in_path = Path(path)
        if verify_sha256:
            _verify_manifest_sha256(in_path)
        payload = torch.load(in_path, map_location=map_location, weights_only=True)
        self.policy_head.load_state_dict(payload["policy_head_state_dict"], strict=True)

    @staticmethod
    def blend_temporal_ensemble(
        trajectory_history: list[torch.Tensor],
        decay_beta: float = 0.5,
    ) -> torch.Tensor:
        """Blend historical overlapping action trajectories using exponential decay weights.

        과거 추론 궤적들과 당일 추론 비중을 지수 감쇠 가중치로 결합하는 시간축 앙상블입니다.
        """
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
