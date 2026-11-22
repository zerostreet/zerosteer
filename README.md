# zerosteer (`Public · MIT · Layer 1`)

Open-source **ZeroSteer (v2)**: Cross-Asset Relational MoE Transformer backbone with Multi-Horizon Action Chunking (`H=20`), Zero-std-safe RevIN + ModernTCN multi-scale temporal encoder, harmonic `FinancialCalendarEncoder`, QK-Norm SDPA attention, DeepSeek-V3 Aux-Loss-Free MoE (`TopKMoELayer` + MoE `WeightAllocatorHead`), LeJEPA `compute_sigreg_loss`, SVD diagnostics, SHA-256 tamper-checked checkpoints, dynamic `ZeroLens` (`z_mm`) adapters (`0/32/128/256d`), and Differentiable Trajectory RL trainer.

오픈소스 **ZeroSteer (v2)**는 Zero-std 안전 RevIN + 다중 스케일 ModernTCN, 다주기 금융 캘린더 인코더, QK-Norm SDPA 어텐션, DeepSeek-V3 Aux-Loss-Free MoE(백본 및 할당기 헤드), LeJEPA SIGReg 정규화, SVD 표현 진단, SHA-256 무결성 매니페스트 및 미분 가능 궤적 RL 트레이너를 제공합니다.

---

## Business Role & Value Proposition / 비즈니스 역할 및 가치

### English

* **Open-Core Trust Magnet & Hugging Face Distribution (`ZeroSteer`)**: Developers and quantitative agents can inspect every line of ZeroStreet's portfolio control architecture openly under MIT, including `TimeseriesEncoder`, `FinancialCalendarEncoder`, `ObservationFusionLayer`, `QKNormMultiheadAttention`, `AuxLossFreeMoEGate`, sparse `Entmax15`/`Sparsemax`, `compute_sigreg_loss`, `evaluate_backbone_representations`, and `CrossAssetMaskedRelationalPretrainer`.
* **Zero Advisory Risk & 0 Bytes of Policy Weights**: `zerosteer` never ships company-trained stock-picking Policy Heads. Instantiating `ZeroSteer` or `DifferentiableTrajectoryRLTrainer` without an explicit `user_config` (`RLRewardWeights` + `target_universe`) raises `ValueError`. Both `ZeroSteerBackbone.save_backbone()` and `ZeroSteer.save_policy_head()` emit SHA-256 sibling `.manifest.json` files verified on load (`verify_sha256=True`).
* **Seamless Upgrade Path to AlphaSmith & ZeroLens**: Out-of-the-box `DynamicEmbeddingAdapter` runs in zero-external-signal OHLCV blackout mode (`is_blackout=True`) locally, while `run_oos_comparison()` lets teams benchmark how injecting paid `ZeroLens` `z_mm` multimodal embeddings (`32d/128d/256d`) improves out-of-sample Sharpe and drawdown—creating natural demand for `AlphaSmith` Serverless RL Foundry compilation.

### 한국어

* **오픈코어 신뢰 확보(Trust Magnet) 및 생태계 확산 엔진**: 개발자와 고객 에이전트가 ZeroStreet의 포트폴리오 제어 모델(`ZeroSteer v2`) 전체 구조—`TimeseriesEncoder`, `FinancialCalendarEncoder`, `ObservationFusionLayer`, `QKNormMultiheadAttention`, `AuxLossFreeMoEGate`, `compute_sigreg_loss`, `evaluate_backbone_representations`, `CrossAssetMaskedRelationalPretrainer`—를 MIT 오픈소스로 투명하게 검증하고 채택할 수 있습니다.
* **규제 안전성 및 정책 가중치 0바이트 원칙**: 회사가 학습한 종목별 매수/매도 Policy Head 가중치를 0바이트도 포함하지 않으며, 사용자가 명시적 `user_config`를 주지 않으면 `ZeroSteer`와 트레이너 생성 자체를 차단(`ValueError`)합니다. 체크포인트 저장·로드 시 동위 SHA-256 매니페스트(`.manifest.json`)로 변조 여부를 검증합니다.
* **AlphaSmith 유료 파운드리 업그레이드 퍼널**: 로컬 OHLCV 단독 모드(`is_blackout=True`)로 즉시 실행·검증해 본 뒤, `run_oos_comparison()`을 통해 비공개 `ZeroLens` 멀티모달 텐서(`32d/128d/256d`) 결합 시 성과 개선폭을 직접 측정하고 `AlphaSmith` Serverless Foundry 유료 구독으로 자연스럽게 전환하도록 유도합니다.

---

## Key Components / 핵심 모듈

1. **Backbone & Policy Split (`model.py`, `moe.py`, `adapters.py`)**: `ZeroSteerBackbone`, `MoETransformerBackbone`, `QKNormMultiheadAttention`, `SwiGLUFeedForward`, `AuxLossFreeMoEGate`, `TimeseriesEncoder`, `FinancialCalendarEncoder`, `extract_calendar_features`, `DynamicEmbeddingAdapter`, `ObservationFusionLayer`, `WeightAllocatorHead`, `ZeroSteer`
2. **Sparse Projections (`activations.py`)**: `Entmax15`, `Sparsemax`, `entmax15`, `sparsemax`
3. **Future-Label-Free Pretraining, SIGReg & Trajectory RL (`pretrain.py`, `rl_trainer.py`)**: `compute_sigreg_loss`, `CrossAssetMaskedRelationalPretrainer`, `TrajectoryRLLoss`, `DifferentiableTrajectoryRLTrainer`
4. **Representation Diagnostics, OOS Comparator & Config (`diagnostics.py`, `benchmark.py`, `config.py`)**: `compute_effective_rank`, `ZeroSteerDiagnosticReport`, `evaluate_backbone_representations`, `run_oos_comparison`, `compute_oos_metrics`, `ZeroSteerConfig` (`legacy_v1()`), `load_user_policy_config`

## Local Verification / 로컬 검증

```bash
uv sync --dev
uv run ruff check .
uv run pytest -q
```
