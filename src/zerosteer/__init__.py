"""Public API for ZeroSteer: cross-asset relational backbone and trajectory RL.

ZeroSteer의 Cross-Asset 관계형 백본 및 다기간 궤적 강화학습 공개 API입니다.
"""

from zerosteer.activations import Entmax15, Sparsemax, entmax15, sparsemax
from zerosteer.adapters import (
    DynamicEmbeddingAdapter,
    FinancialCalendarEncoder,
    ObservationFusionLayer,
    TimeseriesEncoder,
    extract_calendar_features,
)
from zerosteer.benchmark import (
    OOSBenchmarkReport,
    OOSMetrics,
    compute_oos_metrics,
    run_oos_comparison,
)
from zerosteer.config import ZeroSteerConfig, load_user_policy_config
from zerosteer.diagnostics import (
    ZeroSteerDiagnosticReport,
    compute_effective_rank,
    evaluate_backbone_representations,
)
from zerosteer.model import (
    WeightAllocatorHead,
    ZeroSteer,
    ZeroSteerBackbone,
    ZeroSteerOutputs,
    build_horizon_attention_mask,
)
from zerosteer.moe import (
    AuxLossFreeMoEGate,
    GeGLUFeedForward,
    MoETransformerBackbone,
    QKNormMultiheadAttention,
    RMSNorm,
    SwiGLUFeedForward,
    TopKMoELayer,
)
from zerosteer.pretrain import CrossAssetMaskedRelationalPretrainer, compute_sigreg_loss
from zerosteer.rl_trainer import DifferentiableTrajectoryRLTrainer, TrajectoryRLLoss

__version__ = "0.1.0"

__all__ = [
    "AuxLossFreeMoEGate",
    "CrossAssetMaskedRelationalPretrainer",
    "DifferentiableTrajectoryRLTrainer",
    "DynamicEmbeddingAdapter",
    "Entmax15",
    "FinancialCalendarEncoder",
    "GeGLUFeedForward",
    "MoETransformerBackbone",
    "OOSBenchmarkReport",
    "OOSMetrics",
    "ObservationFusionLayer",
    "QKNormMultiheadAttention",
    "RMSNorm",
    "Sparsemax",
    "SwiGLUFeedForward",
    "TimeseriesEncoder",
    "TopKMoELayer",
    "TrajectoryRLLoss",
    "WeightAllocatorHead",
    "ZeroSteer",
    "ZeroSteerBackbone",
    "ZeroSteerConfig",
    "ZeroSteerDiagnosticReport",
    "ZeroSteerOutputs",
    "__version__",
    "build_horizon_attention_mask",
    "compute_effective_rank",
    "compute_oos_metrics",
    "compute_sigreg_loss",
    "entmax15",
    "evaluate_backbone_representations",
    "extract_calendar_features",
    "load_user_policy_config",
    "run_oos_comparison",
    "sparsemax",
]
