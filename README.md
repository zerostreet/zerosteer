# hyperalpha-vla (`Public · MIT · Layer 1`)

Open-source `Model B` (`HyperAlpha-VLA`): Cross-Asset Relational MoE Transformer backbone with Multi-Horizon Action Chunking (`H=20`), dynamic `z_mm` adapters (`0/32/128/256d`), and Differentiable Trajectory RL trainer.

오픈소스 `Model B`(`HyperAlpha-VLA`)는 Cross-Asset 관계형 MoE 백본, 다기간 행동 청킹(`H=20`), 가변 차원 동적 어댑터 및 미분 가능 궤적 RL 트레이너를 제공합니다.

---

## Business Role & Value Proposition / 비즈니스 역할 및 가치

### English

* **Open-Core Trust Magnet & Hugging Face Distribution (`Model B`)**: Developers and quantitative agents can inspect every line of ZeroStreet's portfolio allocation architecture openly under MIT, including `TimeseriesEncoder`, `ObservationFusionLayer`, sparse `Entmax15`/`Sparsemax`, and `CrossAssetMaskedRelationalPretrainer`.
* **Zero Advisory Risk & 0 Bytes of Policy Weights**: `hyperalpha-vla` never ships company-trained stock-picking Policy Heads. Instantiating `HyperAlphaVLA` or `DifferentiableTrajectoryRLTrainer` without an explicit `user_config` (`RLRewardWeights` + `target_universe`) raises `ValueError`.
* **Seamless Upgrade Path to AlphaSmith & zerostreet-models**: Out-of-the-box `DynamicEmbeddingAdapter` runs in zero-external-signal OHLCV blackout mode (`is_blackout=True`) locally, while `run_oos_comparison()` lets teams benchmark how injecting paid `z_mm` multimodal embeddings (`32d/128d/256d`) improves out-of-sample Sharpe and drawdown—creating natural demand for `AlphaSmith` Serverless RL Foundry compilation.

### 한국어

* **오픈코어 신뢰 확보(Trust Magnet) 및 생태계 확산 엔진**: 개발자와 고객 에이전트가 ZeroStreet의 포트폴리오 의사결정 모델(`Model B`) 전체 구조—`TimeseriesEncoder`, `ObservationFusionLayer`, 희소 `Entmax15`/`Sparsemax`, `CrossAssetMaskedRelationalPretrainer`—를 MIT 오픈소스로 투명하게 검증하고 채택할 수 있습니다.
* **규제 안전성 및 정책 가중치 0바이트 원칙**: 회사가 학습한 종목별 매수/매도 Policy Head 가중치를 0바이트도 포함하지 않으며, 사용자가 명시적 `user_config`를 주지 않으면 `HyperAlphaVLA`와 트레이너 생성 자체를 차단(`ValueError`)합니다.
* **AlphaSmith 유료 파운드리 업그레이드 퍼널**: 로컬 OHLCV 단독 모드(`is_blackout=True`)로 즉시 실행·검증해 본 뒤, `run_oos_comparison()`을 통해 비공개 `Model A` 멀티모달 텐서(`32d/128d/256d`) 결합 시 성과 개선폭을 직접 측정하고 `AlphaSmith` Serverless Foundry 유료 구독으로 자연스럽게 전환하도록 유도합니다.

---

## Key Components / 핵심 모듈

1. **Backbone & Policy Split (`model.py`, `moe.py`, `adapters.py`)**: `HyperAlphaVLABackbone`, `MoETransformerBackbone`, `DynamicEmbeddingAdapter`, `ObservationFusionLayer`, `WeightAllocatorHead`, `HyperAlphaVLA`
2. **Sparse Projections (`activations.py`)**: `Entmax15`, `Sparsemax`, `entmax15`, `sparsemax`
3. **Future-Label-Free Pretraining & Trajectory RL (`pretrain.py`, `rl_trainer.py`)**: `CrossAssetMaskedRelationalPretrainer`, `TrajectoryRLLoss`, `DifferentiableTrajectoryRLTrainer`
4. **OOS Comparator & Config (`benchmark.py`, `config.py`)**: `run_oos_comparison`, `compute_oos_metrics`, `HyperAlphaVLAConfig`, `load_user_policy_config`

## Local Verification / 로컬 검증

```bash
uv sync --dev
uv run ruff check .
uv run pytest -q
```
