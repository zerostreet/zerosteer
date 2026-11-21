# hyperalpha-vla (`Model B`)

**`HyperAlpha-VLA`**는 ZeroStreet의 오픈소스(MIT) **Cross-Asset 관계형 포트폴리오 모델(`Model B`) 및 미분 가능 궤적 강화학습(Differentiable Trajectory RL) 프레임워크**입니다.

## 핵심 아키텍처 및 규제 안전 경계

1. **Policy Head 0바이트 격리 (`user_config=None -> ValueError`)**:
   - 회사가 학습시킨 종목별 매수/매도 Policy 가중치는 **0바이트**만 포함되며, `HyperAlphaVLA(..., user_config=None)` 또는 `DifferentiableTrajectoryRLTrainer(..., user_config=None)` 호출 시 즉시 `ValueError`를 발생시킵니다.
   - 오직 사용자 에이전트가 직접 `UserPolicyConfig`를 지정해야만 `WeightAllocatorHead`가 초기화·컴파일됩니다.
2. **Future-Label-Free Cross-Asset 관계형 백본 (`HyperAlphaVLABackbone`)**:
   - `CrossAssetMaskedRelationalPretrainer`는 동일 시점 $t$ 내 자산 간 마스킹 복원만 수행하며 미래 수익률/변동성 라벨을 0% 사용합니다.
3. **가변 차원 어댑터 (`DynamicEmbeddingAdapter`)**:
   - `0 / 32 / 128 / 256` 차원의 `AlphaSmith` 멀티모달 임베딩(`z_mm`)을 수용하며, `is_blackout=True` 또는 미연결 시 학습된 `[MASK_SIGNAL]` 토큰으로 폴백하여 로컬 OHLCV 단독 모드에서도 즉시 작동합니다.
4. **다기간 Action Chunking (`H=20`) + `1.5-Entmax` + Dense-Sparse STE**:
   - 관측 융합(`ObservationFusionLayer`)은 배치당 1회만 연산하며, 비대칭 2D 어텐션 마스크(`build_horizon_attention_mask`)로 `obs_embs` 오염을 차단하고, `P_0`에만 직전 실제 보유 비중(`prev_weights`)을 앵커링합니다.

## 로컬 검증

```bash
uv sync --dev
uv run ruff check .
uv run pytest -q
```
