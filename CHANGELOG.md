# Changelog

All notable changes to **`zerosteer`** are documented in this file.
The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

이 프로젝트의 주요 변경 이력은 Keep a Changelog 1.1.0 및 SemVer 2.0.0 규격을 따릅니다.

---

## [Unreleased]

## [0.2.0] - 2026-11-22

### Added
- **ZeroSteer v2 Architecture**:
  - `TimeseriesEncoder`: zero-std-safe RevIN (`unbiased=False` + `sqrt(var + 1e-5)` for `CASH=1.0` safety), multi-scale ModernTCN depthwise convolutions `(3, 7, 15)`, reversed recency positional encoding, and Window-Stats FiLM scale restoration.
  - `extract_calendar_features` & `FinancialCalendarEncoder`: deterministic 10d multi-periodic weekday/quarter/year harmonic conditioning.
  - `QKNormMultiheadAttention` & `SwiGLUFeedForward`: per-head Q/K `RMSNorm` with inverted boolean SDPA mask handling (`~attn_mask`) preventing future horizon (`P_1..P_{H-1}`) leakage.
  - `AuxLossFreeMoEGate` & MoE `WeightAllocatorHead`: DeepSeek-V3 bias-based load balancing outside autograd and Apple Silicon MPS-safe out-of-place `index_add`.
  - `compute_sigreg_loss` & `diagnostics.py` (`compute_effective_rank`, `ZeroSteerDiagnosticReport`, `evaluate_backbone_representations`) with CPU-safe `float64` SVD handling for MPS.
  - Hardened `TrajectoryRLLoss` blending trajectory metrics with discounted daily net return `step_adv`, `freeze_backbone=True` bitwise MoE buffer invariance (`include_backbone=False`), optional `lr_warmup_steps` cosine schedule, `model_frz_update_freq` periodic sync, SHA-256 sibling `.manifest.json` verification, and `ZeroSteerConfig.legacy_v1()`.
  - Apple Silicon MPS parity & full v2 test suite (`tests/test_zerosteer_v2_architecture.py`) and bilingual OpenWiki visual guide.
- **ZeroSteer v2 고도화**: Zero-Std 안전 RevIN·ModernTCN·Window-Stats FiLM, 10차원 금융 캘린더 인코더, QK-Norm SDPA 반전 마스크, Aux-Loss-Free MoE 및 MoE Policy Head, LeJEPA SIGReg, CPU `float64` SVD 표현 진단, Step+Trajectory 보상 결합 RL, SHA-256 매니페스트 무결성 검증 및 Apple Silicon MPS 정합성 검증을 반영했습니다.

## [0.1.0] - 2026-11-22

### Added
- Initial MIT release of **ZeroSteer**: Future-Label-Free Cross-Asset MoE Transformer backbone (`ZeroSteerBackbone`), `DynamicEmbeddingAdapter` (`0/32/128/256d` + blackout fallback), 0-byte Policy Head separation (`user_config=None -> ValueError`), `Entmax15`/`Sparsemax`, `CrossAssetMaskedRelationalPretrainer`, `DifferentiableTrajectoryRLTrainer`, and `run_oos_comparison`.
- **ZeroSteer** 초기 오픈소스 릴리스: Cross-Asset 관계형 MoE 백본, 동적 `z_mm` 어댑터, 정책 가중치 0바이트 분리 원칙, `Entmax15`/`Sparsemax`, 사전학습·궤적 RL 트레이너 및 OOS 비교 도구를 공개했습니다.
