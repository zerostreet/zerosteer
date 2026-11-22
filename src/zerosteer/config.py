"""Model hyperparameters and user policy YAML loader for ZeroSteer.

ZeroSteer 모델 하이퍼파라미터 설정 및 사용자 정책 YAML 로더 모듈입니다.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from zerostreet_schema import HorizonMaskMode, UserPolicyConfig


class ZeroSteerConfig(BaseModel):
    """Hyperparameter configuration for ZeroSteer backbone and trajectory RL trainer.

    ZeroSteer 백본 및 궤적 RL 트레이너 하이퍼파라미터 설정입니다.
    """

    d_model: int = Field(default=128, ge=16)
    nhead: int = Field(default=8, ge=1)
    nlayers: int = Field(default=6, ge=1)
    n_dense_layers: int = Field(default=3, ge=0)
    n_routed_experts: int = Field(default=16, ge=2)
    n_activated_experts: int = Field(default=4, ge=1)
    expert_inter_dim: int = Field(default=256, ge=16)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)

    # 시계열 & 멀티모달 융합 설정
    timeseries_features: int = Field(default=6, ge=1)
    seq_len: int = Field(default=60, ge=4)
    supported_signal_dims: tuple[int, ...] = (32, 128, 256)
    signal_fusion_mode: Literal["film", "gated"] = "film"
    full_blackout_prob: float = Field(default=0.15, ge=0.0, le=1.0)

    # 다기간 행동 청킹(Multi-Horizon Action Chunking) 설정
    portfolio_horizon: int = Field(default=20, ge=1, le=64)
    max_horizon_num: int = Field(default=64, ge=1)
    horizon_mask_mode: HorizonMaskMode = HorizonMaskMode.BIDIRECTIONAL_PORTFOLIO
    obs_attend_p0: bool = True

    # Policy Head & Trajectory RL 안정화 하이퍼파라미터
    weights_output_mode: Literal["entmax15", "sparsemax", "softmax"] = "entmax15"
    ste_softmax_alpha: float = Field(default=0.10, ge=0.0, le=1.0)
    horizon_discount_gamma: float = Field(default=0.95, gt=0.0, le=1.0)
    fee_rate: float = Field(default=0.001, ge=0.0)
    quadratic_turnover_coeff: float = Field(default=0.005, ge=0.0)
    cvar_alpha: float = Field(default=0.15, gt=0.0, le=1.0)
    use_drifted_weights: bool = True
    step_reward_weight: float = Field(default=0.50, ge=0.0)
    trajectory_reward_weight: float = Field(default=0.50, ge=0.0)
    beta_trust_region: float = Field(default=0.02, ge=0.0)
    advantage_rms_floor: float = Field(default=0.05, gt=0.0)
    advantage_clip_val: float = Field(default=3.0, gt=0.0)
    max_weight_scale: float = Field(default=0.05, gt=0.0)
    temperature: float = Field(default=1.0, gt=0.0)


def load_user_policy_config(path: str | Path) -> UserPolicyConfig:
    """Load and validate a user-supplied Policy Head configuration from YAML.

    YAML 파일에서 사용자 지정 Policy Head 설정을 로드하고 검증합니다.
    """
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"User policy config not found: {yaml_path}")
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not raw or not isinstance(raw, dict):
        raise ValueError(f"User policy config is empty or invalid mapping: {yaml_path}")
    return UserPolicyConfig.model_validate(raw)
