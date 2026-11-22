---
type: quickstart
title: hyperalpha-vla Quickstart & Business Overview / hyperalpha-vla 빠른 시작 및 비즈니스 개요
description: Introduces the Model B open-core role, 0-byte policy weight boundary, config.example.yaml, validation commands, and OpenWiki map. Model B의 비즈니스 역할, Policy Head 0바이트 경계, config.example.yaml 예시, 검증 방법 및 문서 맵을 제공합니다.
tags:
  - quickstart
  - hyperalpha-vla
  - model-b
  - pytorch
  - compliance
  - business-context
verified:
  - by: openwiki/0.7.2
    at: 2026-11-22T03:52:50.876Z
sources:
  - id: openwiki-source-873af29a56ffef7d51f59cb6
    resource: repo://config.example.yaml
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-503bbbc2e71356e464ab331f
    resource: repo://src/hyperalpha_vla/__init__.py
  - id: openwiki-source-6ae6f72eecc22448f38ce845
    resource: repo://src/hyperalpha_vla/config.py
generated: { by: "antigravity", at: "2026-11-22T03:52:50.876Z" }
---

# hyperalpha-vla Quickstart & Business Overview / 빠른 시작 및 비즈니스 개요

`hyperalpha-vla` (`v0.1.0`, MIT) is the open-source **Model B (Cross-Asset Relational Portfolio Network & Trajectory RL Framework)** in the ZeroStreet ecosystem. To comply strictly with investment-advisory laws, public artifacts contain **only the Future-Label-Free relational backbone (`HyperAlphaVLABackbone`)** while customer agents compile and own individualized `WeightAllocatorHead` weights.

`hyperalpha-vla` (`v0.1.0`, MIT)는 ZeroStreet 생태계의 오픈소스 **Model B(Cross-Asset 관계형 포트폴리오 신경망 및 궤적 강화학습 프레임워크)**로, 공개 범위에는 순수 관계형 백본만 포함하고 종목별 정책 헤드는 반드시 사용자 에이전트가 직접 컴파일하도록 분리합니다.

---

## Business Role & Value Proposition / 비즈니스 역할 및 가치

### English

* **Open-Core Trust Magnet (`Model B`)**: Publishes ZeroStreet's core relational MoE Transformer, dynamic multimodal adapters, sparse `1.5-Entmax` / `Sparsemax` projections, and differentiable Trajectory RL compiler openly under MIT ([`README.md`](../README.md)).
* **Zero Advisory Risk & 0 Bytes of Policy Weights**: Calling `HyperAlphaVLA` or `DifferentiableTrajectoryRLTrainer` without explicit `user_config` raises `ValueError` immediately.
* **Commercial Funnel to AlphaSmith & zerostreet-models**: Runs standalone in local OHLCV blackout mode (`is_blackout=True`), while `run_oos_comparison()` proves the performance lift of injecting paid `Model A` `z_mm` embeddings (`32d/128d/256d`) and compiling user checkpoints via AlphaSmith Serverless Foundry.

### 한국어

* **오픈코어 신뢰 확보(Trust Magnet) 엔진**: 관계형 MoE 트랜스포머, 동적 어댑터, 희소 심플렉스 투영 및 미분 가능 궤적 RL 구조를 MIT 오픈소스로 공개해 개발자·에이전트의 신뢰와 채택을 극대화합니다.
* **규제 안전성 및 정책 가중치 0바이트**: `user_config=None`일 때 즉시 `ValueError`를 발생시켜 회사 기본 전략 배포를 차단합니다.
* **AlphaSmith 유료 전환 퍼널**: 로컬 OHLCV 단독 모드(`is_blackout=True`)로 먼저 검증한 뒤 `run_oos_comparison()`으로 `Model A` 텐서 결합 시 OOS 개선 효과를 수치로 확인하고 유료 파운드리로 전환하도록 설계되었습니다.

---

## 1. Documentation Routing Map / 작업별 문서 라우팅 맵

| Topic / 탐색 주제 | Document Path / 문서 경로 | Summary / 핵심 내용 |
| :--- | :--- | :--- |
| **Model & Adapters / 모델 구조 & 어댑터** | [Model B Relational Backbone, Adapters & Action Chunking](./architecture/backbone-and-adapters.md) | `DynamicEmbeddingAdapter(0/32/128/256d)`, `ObservationFusionLayer`, `MoETransformerBackbone`, asymmetric masks, `1.5-Entmax` |
| **Pretraining & Trajectory RL / 사전학습 & 궤적 RL** | [Relational Pretraining, Trajectory RL & OOS Benchmark](./workflows/pretrain-and-trajectory-rl.md) | `CrossAssetMaskedRelationalPretrainer`, `TrajectoryRLLoss`, `DifferentiableTrajectoryRLTrainer`, `run_oos_comparison` |

---

## 2. Regulatory Boundary & `config.example.yaml` / 법적 안전 경계 및 설정 예시

* **0-Byte Policy Weight Invariant / 정책 가중치 0바이트 원칙**: [`HyperAlphaVLA`](../src/hyperalpha_vla/model.py) and [`DifferentiableTrajectoryRLTrainer`](../src/hyperalpha_vla/rl_trainer.py) reject `user_config=None` at instantiation time.
* **Illustrative Config Syntax / 문법 참고용 가상 예시**: [`config.example.yaml`](../config.example.yaml) illustrates synthetic tickers (`ASSET_A`, `ASSET_B`, `ASSET_C`, `CASH`) for two reward profiles and is never auto-loaded as defaults; users must explicitly validate YAML via `load_user_policy_config()` ([`config.py`](../src/hyperalpha_vla/config.py)).

---

## 3. Installation & Local Verification / 설치 및 로컬 검증 절차

[`pyproject.toml`](../pyproject.toml) links `../zerostreet-schema` via `[tool.uv.sources]` alongside `torch`, `numpy`, `pydantic`, `pyyaml`, and `loguru` ([`__init__.py`](../src/hyperalpha_vla/__init__.py)).

```bash
uv sync --dev
uv run ruff check .
uv run pytest -q
```
