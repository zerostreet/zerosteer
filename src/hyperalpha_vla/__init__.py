"""Public API for Model B (HyperAlpha-VLA): cross-asset relational backbone and trajectory RL.

Model B (HyperAlpha-VLA)의 Cross-Asset 관계형 백본 및 다기간 궤적 강화학습 공개 API입니다.
"""

from hyperalpha_vla.activations import Entmax15, Sparsemax, entmax15, sparsemax
from hyperalpha_vla.adapters import (
    DynamicEmbeddingAdapter,
    ObservationFusionLayer,
    TimeseriesEncoder,
)
from hyperalpha_vla.benchmark import (
    OOSBenchmarkReport,
    OOSMetrics,
    compute_oos_metrics,
    run_oos_comparison,
)
from hyperalpha_vla.config import HyperAlphaVLAConfig, load_user_policy_config
from hyperalpha_vla.model import (
    HyperAlphaVLA,
    HyperAlphaVLABackbone,
    HyperAlphaVLAOutputs,
    WeightAllocatorHead,
    build_horizon_attention_mask,
)
from hyperalpha_vla.moe import GeGLUFeedForward, MoETransformerBackbone, TopKMoELayer
from hyperalpha_vla.pretrain import CrossAssetMaskedRelationalPretrainer
from hyperalpha_vla.rl_trainer import DifferentiableTrajectoryRLTrainer, TrajectoryRLLoss

__version__ = "0.1.0"

__all__ = [
    "CrossAssetMaskedRelationalPretrainer",
    "DifferentiableTrajectoryRLTrainer",
    "DynamicEmbeddingAdapter",
    "Entmax15",
    "GeGLUFeedForward",
    "HyperAlphaVLA",
    "HyperAlphaVLABackbone",
    "HyperAlphaVLAConfig",
    "HyperAlphaVLAOutputs",
    "MoETransformerBackbone",
    "OOSBenchmarkReport",
    "OOSMetrics",
    "ObservationFusionLayer",
    "Sparsemax",
    "TimeseriesEncoder",
    "TopKMoELayer",
    "TrajectoryRLLoss",
    "WeightAllocatorHead",
    "__version__",
    "build_horizon_attention_mask",
    "compute_oos_metrics",
    "entmax15",
    "load_user_policy_config",
    "run_oos_comparison",
    "sparsemax",
]
