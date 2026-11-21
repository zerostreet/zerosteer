---
type: quickstart
title: hyperalpha-vla 빠른 시작 및 아키텍처 맵
description: Model B(HyperAlpha-VLA)의 Policy Head 0바이트 분리 경계, config.example.yaml 문법 예시, 설치·검증 방법 및 문서 라우팅 맵을 제공합니다.
tags: [quickstart, hyperalpha-vla, model-b, pytorch, compliance]
verified:
  - by: openwiki/0.7.2
    at: 2026-11-21T17:10:03.172Z
sources:
  - id: openwiki-source-873af29a56ffef7d51f59cb6
    resource: repo://config.example.yaml
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-503bbbc2e71356e464ab331f
    resource: repo://src/hyperalpha_vla/__init__.py
  - id: openwiki-source-6ae6f72eecc22448f38ce845
    resource: repo://src/hyperalpha_vla/config.py
generated: { by: "antigravity", at: "2026-11-21T17:10:03.172Z" }
---

# hyperalpha-vla 빠른 시작 및 아키텍처 맵

`hyperalpha-vla` (`v0.1.0`, MIT)는 ZeroStreet 생태계의 오픈소스 **Model B(Cross-Asset 관계형 포트폴리오 신경망 및 궤적 강화학습 프레임워크)**입니다. 자본시장법상 무인가 투자자문·일임업을 원천 회피하기 위해, 공개 모델에는 **Future-Label-Free 관계형 백본(`HyperAlphaVLABackbone`)만 포함**하고 종목별 비중 할당기(`WeightAllocatorHead`)는 반드시 사용자 에이전트가 직접 컴파일하도록 격리합니다.

---

## 1. 작업별 문서 라우팅 맵

| 탐색 주제 | 문서 경로 | 핵심 내용 |
| :--- | :--- | :--- |
| **모델 구조 & 어댑터** | [Model B 관계형 백본, 어댑터 및 다기간 Action Chunking 아키텍처](architecture/backbone-and-adapters.md) | `DynamicEmbeddingAdapter(0/32/128/256d + blackout)`, `ObservationFusionLayer`, `MoETransformerBackbone`, 비대칭 마스크, `1.5-Entmax` |
| **사전학습 & 궤적 RL** | [관계형 사전학습, 궤적 강화학습 컴파일 및 OOS 벤치마크](workflows/pretrain-and-trajectory-rl.md) | `CrossAssetMaskedRelationalPretrainer`, `TrajectoryRLLoss` (STE/드리프트/rms_softclip), `DifferentiableTrajectoryRLTrainer`, `run_oos_comparison` |

---

## 2. 법적 안전 경계 및 `config.example.yaml`

- **Policy Head 0바이트 원칙**: `HyperAlphaVLA`와 `DifferentiableTrajectoryRLTrainer`는 `user_config=None`일 때 즉시 `ValueError`를 발생시킵니다.
- **문법 참고용 가상 예시**: `config.example.yaml`은 가상 심볼 `["ASSET_A", "ASSET_B", "ASSET_C", "CASH"]`와 예시 A(저회전·방어형) 및 예시 B(수익·변동성 균형형) 문법만 제공하며 코드 기본값으로 자동 로드되지 않습니다. `load_user_policy_config()`로 사용자 YAML을 명시적으로 검증해 주입해야 합니다.

---

## 3. 설치 및 로컬 검증 절차

`pyproject.toml`은 `[tool.uv.sources]`를 통해 `../zerostreet-schema`를 editable 로컬 패키지로 연결하며, `torch`, `numpy`, `pydantic`, `pyyaml`, `loguru`에 의존합니다.

```bash
uv sync --dev
uv run ruff check .
uv run pytest -q
```
