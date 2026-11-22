---
type: quickstart
title: zerosteer Quickstart & Business Overview / zerosteer 빠른 시작 및 비즈니스 개요
description: Introduces the ZeroSteer open-core role, 0-byte policy weight boundary, config.example.yaml, validation commands, and OpenWiki map. ZeroSteer의 비즈니스 역할, Policy Head 0바이트 경계, config.example.yaml 예시, 검증 방법 및 문서 맵을 제공합니다.
tags:
  - quickstart
  - zerosteer
  - pytorch
  - compliance
  - business-context
  - zerolens
verified:
  - by: openwiki/0.7.2
    at: 2026-11-22T09:34:13.721Z
sources:
  - id: openwiki-source-873af29a56ffef7d51f59cb6
    resource: repo://config.example.yaml
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-00bb995465adf73d06c1b44e
    resource: repo://src/zerosteer/__init__.py
  - id: openwiki-source-ee493b2cba1f439554f5d214
    resource: repo://src/zerosteer/config.py
generated: { by: "antigravity", at: "2026-11-22T09:34:13.721Z" }
---

# zerosteer Quickstart & Business Overview / zerosteer 빠른 시작 및 비즈니스 개요

## Business Role & Value Proposition / 비즈니스 역할 및 가치

### English

* **Open-Core Trust Magnet (`ZeroSteer`)**: `zerosteer` (`v0.1.0`, MIT) publishes ZeroStreet's Cross-Asset relational MoE Transformer backbone (`ZeroSteerBackbone`), dynamic `ZeroLens` (`z_mm`) adapters, sparse `Entmax15` / `Sparsemax` projections, and differentiable Trajectory RL compiler openly under MIT ([`README.md`](../README.md)) while holding **0 bytes** of vendor-trained stock-picking Policy Head weights.
* **Zero Advisory Risk & 0 Bytes of Policy Weights**: Calling `ZeroSteer` or `DifferentiableTrajectoryRLTrainer` without explicit `user_config` raises `ValueError` immediately; customer agents compile and own individualized `WeightAllocatorHead` weights.
* **Commercial Funnel to AlphaSmith & ZeroLens**: Runs standalone in local OHLCV blackout mode (`is_blackout=True`), while `run_oos_comparison()` proves the performance lift of injecting paid `ZeroLens` `z_mm` embeddings (`32d/128d/256d`) and compiling user checkpoints via AlphaSmith Serverless Foundry.

### 한국어

* **오픈코어 신뢰 확보(Trust Magnet) 제어 엔진**: `zerosteer`(`v0.1.0`, MIT)는 Cross-Asset 관계형 MoE 트랜스포머 백본(`ZeroSteerBackbone`), 동적 `ZeroLens` 어댑터, 희소 심플렉스 투영 및 미분 가능 궤적 RL 컴파일러를 MIT 오픈소스로 공개해 개발자·에이전트의 신뢰와 채택을 극대화하면서도 회사 학습 정책 가중치는 **0바이트**만 유지합니다.
* **규제 안전성 및 정책 가중치 0바이트**: `user_config=None`일 때 즉시 `ValueError`를 발생시켜 회사 기본 전략 배포를 차단하며, 모든 `WeightAllocatorHead` 가중치는 사용자 에이전트가 직접 컴파일·소유합니다.
* **AlphaSmith 유료 전환 퍼널**: 로컬 OHLCV 단독 모드(`is_blackout=True`)로 먼저 검증한 뒤 `run_oos_comparison()`으로 `ZeroLens` 텐서 결합 시 OOS 개선 효과를 수치로 확인하고 유료 파운드리로 전환하도록 설계되었습니다.

---

## 1. Documentation Routing Map / 작업별 문서 라우팅 맵

| Topic / 탐색 주제 | Document Path / 문서 경로 | Summary / 핵심 내용 |
| :--- | :--- | :--- |
| **Model & Adapters / 모델 구조 & 어댑터** | [ZeroSteer Relational Backbone, Adapters & Action Chunking](./architecture/backbone-and-adapters.md) | `DynamicEmbeddingAdapter(0/32/128/256d)`, `ObservationFusionLayer`, `MoETransformerBackbone`, asymmetric masks, `Entmax15` |
| **Pretraining & Trajectory RL / 사전학습 & 궤적 RL** | [Relational Pretraining, Trajectory RL & OOS Benchmark](./workflows/pretrain-and-trajectory-rl.md) | `CrossAssetMaskedRelationalPretrainer`, `TrajectoryRLLoss`, `DifferentiableTrajectoryRLTrainer`, `run_oos_comparison` |

---

## 2. Regulatory Boundary & `config.example.yaml` / 법적 안전 경계 및 설정 예시

* **0-Byte Policy Weight Invariant / 정책 가중치 0바이트 원칙**: [`ZeroSteer`](../src/zerosteer/model.py) and [`DifferentiableTrajectoryRLTrainer`](../src/zerosteer/rl_trainer.py) reject `user_config=None` at instantiation time.
* **Illustrative Config Syntax / 문법 참고용 가상 예시**: [`config.example.yaml`](../config.example.yaml) illustrates synthetic tickers (`ASSET_A`, `ASSET_B`, `ASSET_C`, `CASH`) for two reward profiles and is never auto-loaded as defaults; users must explicitly validate YAML via `load_user_policy_config()` ([`config.py`](../src/zerosteer/config.py)).

---

## 3. Installation & Local Verification / 설치 및 로컬 검증 절차

[`pyproject.toml`](../pyproject.toml) links `../zerostreet-schema` via `[tool.uv.sources]` alongside `torch`, `numpy`, `pydantic`, `pyyaml`, and `loguru` ([`__init__.py`](../src/zerosteer/__init__.py)).

```bash
uv sync --dev
uv run ruff check .
uv run pytest -q
```
