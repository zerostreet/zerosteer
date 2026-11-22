---
type: quickstart
title: ZeroSteer Overview & Quickstart / ZeroSteer 개요 및 빠른 시작
description: Introduces the ZeroSteer v2 open-core role, intuitive visual guide link, 0-byte policy weight boundary, SHA-256 checkpoint integrity, v2 public exports, config.example.yaml, and local/MPS verification. ZeroSteer v2의 오픈코어 비즈니스 역할, 직관 도식 가이드 연결, Policy Head 0바이트 경계, SHA-256 무결성 매니페스트, v2 공개 API 및 로컬/MPS 검증 방법을 안내합니다.
tags:
  - quickstart
  - zerosteer
  - zerosteer-v2
  - pytorch
  - compliance
  - business-context
  - zerolens
verified:
  - by: openwiki/0.7.2
    at: 2026-11-22T11:49:49.324Z
sources:
  - id: openwiki-source-873af29a56ffef7d51f59cb6
    resource: repo://config.example.yaml
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-00bb995465adf73d06c1b44e
    resource: repo://src/zerosteer/__init__.py
  - id: openwiki-source-ee493b2cba1f439554f5d214
    resource: repo://src/zerosteer/config.py
  - id: openwiki-source-3b787a5e57595f705c524ac4
    resource: repo://tests/test_zerosteer_v2_architecture.py
generated: { by: "antigravity", at: "2026-11-22T11:49:49.324Z" }
---

# ZeroSteer Overview & Quickstart / ZeroSteer 개요 및 빠른 시작

## Business Role & Value Proposition / 비즈니스 역할 및 가치

### English

* **Open-Core Trust Magnet (`ZeroSteer v2`)**: `zerosteer` (`v0.1.0`, MIT) publishes ZeroStreet's Cross-Asset relational MoE Transformer backbone (`ZeroSteerBackbone`), zero-std RevIN + ModernTCN `TimeseriesEncoder`, 10d `FinancialCalendarEncoder`, `QKNormMultiheadAttention`, DeepSeek-V3 `AuxLossFreeMoEGate`, sparse `Entmax15`/`Sparsemax`, LeJEPA `compute_sigreg_loss`, SVD `evaluate_backbone_representations`, and Differentiable Trajectory RL compiler openly under MIT ([`README.md`](../README.md)) while holding **0 bytes** of vendor-trained stock-picking Policy Head weights.
* **Zero Advisory Risk & Tamper-Checked Policy Split**: Calling `ZeroSteer` or `DifferentiableTrajectoryRLTrainer` without explicit `user_config` raises `ValueError` immediately; customer agents compile and own individualized MoE/MLP `WeightAllocatorHead` weights verified via SHA-256 sibling `.manifest.json` files.
* **Commercial Funnel to AlphaSmith & ZeroLens**: Runs standalone in local OHLCV blackout mode (`is_blackout=True`), while `run_oos_comparison()` proves the performance lift of injecting paid `ZeroLens` `z_mm` embeddings (`32d/128d/256d`) and compiling user checkpoints via AlphaSmith Serverless Foundry.

### 한국어

* **오픈코어 신뢰 확보(Trust Magnet) 제어 엔진 (`ZeroSteer v2`)**: `zerosteer`(`v0.1.0`, MIT)는 RevIN+ModernTCN `TimeseriesEncoder`, 10차원 `FinancialCalendarEncoder`, `QKNormMultiheadAttention`, DeepSeek-V3 `AuxLossFreeMoEGate`, 희소 심플렉스 투영, LeJEPA `compute_sigreg_loss`, SVD 표현 진단 및 미분 가능 궤적 RL 컴파일러 전체를 MIT 오픈소스로 공개([`README.md`](../README.md))하면서도 회사 학습 종목 선정 가중치는 **0바이트**만 유지합니다.
* **규제 안전성 및 SHA-256 무결성 정책 분리**: `user_config=None`일 때 즉시 `ValueError`를 발생시켜 회사 기본 전략 배포를 차단하며, 고객 에이전트가 직접 컴파일·소유한 정책 헤드는 동위 `.manifest.json` SHA-256 검증으로 보호됩니다.
* **AlphaSmith 유료 전환 퍼널**: 로컬 OHLCV 단독 모드(`is_blackout=True`)로 먼저 검증한 뒤 `run_oos_comparison()`으로 `ZeroLens` 텐서(`32d/128d/256d`) 결합 효과를 수치로 확인하고 유료 파운드리로 전환하도록 설계되었습니다.

> **Start Here for Intuition / 직관적 이해 먼저 보기**: Read the [ZeroSteer v2 Intuitive Visual Guide / ZeroSteer v2 직관적 이해 및 도식 가이드](./guides/intuitive-visual-guide.md) for the autonomous-driving analogy, a step-by-step 4-asset example (`AAPL, MSFT, NVDA, CASH`), and 5 clear architecture diagrams. / 자율주행 비유, 4종목 실전 예시(`AAPL·MSFT·NVDA·CASH`), 5가지 핵심 도식으로 ZeroSteer v2를 한눈에 이해하려면 [ZeroSteer v2 직관적 이해 및 도식 가이드](./guides/intuitive-visual-guide.md)부터 읽어보세요.

---

## 1. Documentation Routing Map / 작업별 문서 라우팅 맵

| Topic / 탐색 주제 | Document Path / 문서 경로 | Summary / 핵심 내용 |
| :--- | :--- | :--- |
| **Intuitive Visual Guide / 직관적 도식 가이드** | [ZeroSteer v2 Intuitive Visual Guide / ZeroSteer v2 직관적 이해 및 도식 가이드](./guides/intuitive-visual-guide.md) | Self-driving analogy, concrete `AAPL/MSFT/NVDA/CASH` data journey, inverted SDPA mask partition, why each v2 upgrade matters, and 3-phase lifecycle diagrams |
| **Backbone & Adapters / 백본 및 어댑터** | [Backbone, Adapters & Policy Split](./architecture/backbone-and-adapters.md) | `TimeseriesEncoder` (RevIN + ModernTCN + Window-Stats FiLM), `FinancialCalendarEncoder`, `DynamicEmbeddingAdapter`, `QKNormMultiheadAttention`, `AuxLossFreeMoEGate`, MoE `WeightAllocatorHead`, SHA-256 `.manifest.json`, `ZeroSteerConfig.legacy_v1()` |
| **Pretraining, Diagnostics & RL / 사전학습·진단·RL** | [Pretraining, SIGReg, Diagnostics & Trajectory RL](./workflows/pretrain-and-trajectory-rl.md) | `compute_sigreg_loss`, `CrossAssetMaskedRelationalPretrainer`, `compute_effective_rank`, `evaluate_backbone_representations`, `TrajectoryRLLoss` (step+trajectory blending), `DifferentiableTrajectoryRLTrainer`, `run_oos_comparison` |

---

## 2. Regulatory Boundary, v2 Public API & `config.example.yaml` / 법적 안전 경계, v2 공개 API 및 설정 예시

* **0-Byte Policy Weight Invariant / 정책 가중치 0바이트 원칙**: [`ZeroSteer`](../src/zerosteer/model.py) and [`DifferentiableTrajectoryRLTrainer`](../src/zerosteer/rl_trainer.py) reject `user_config=None` at instantiation time, and checkpoint loaders verify sibling `<stem>.manifest.json` SHA-256 digests when `verify_sha256=True`. (`ZeroSteer`와 트레이너는 `user_config=None`을 즉시 거부하며 체크포인트 로더는 동위 `.manifest.json` SHA-256 해시를 검증합니다.)
* **Top-Level v2 Exports (`__init__.py`) / v2 최상위 공개 API**: [`__init__.py`](../src/zerosteer/__init__.py) exports `ZeroSteerConfig` (`legacy_v1()`), `QKNormMultiheadAttention`, `SwiGLUFeedForward`, `AuxLossFreeMoEGate`, `FinancialCalendarEncoder`, `extract_calendar_features`, `compute_sigreg_loss`, `compute_effective_rank`, `ZeroSteerDiagnosticReport`, and `evaluate_backbone_representations` alongside core v1 modules. (`__init__.py`는 기존 API와 함께 QK-Norm, SwiGLU, Aux-Loss-Free MoE, 캘린더 인코더, SIGReg 및 SVD 진단 도구를 모두 공개합니다.)
* **Illustrative Config Syntax / 문법 참고용 가상 예시**: [`config.example.yaml`](../config.example.yaml) illustrates synthetic tickers (`ASSET_A`, `ASSET_B`, `ASSET_C`, `CASH`) and is never auto-loaded as defaults; users must explicitly validate YAML via `load_user_policy_config()` ([`config.py`](../src/zerosteer/config.py)). (`config.example.yaml`은 자동 로드되지 않으며 반드시 `load_user_policy_config()`로 명시적 검증 후 사용해야 합니다.)

---

## 3. Installation & Local Verification / 설치 및 로컬 검증 절차

[`pyproject.toml`](../pyproject.toml) links editable `../zerostreet-schema` via `[tool.uv.sources]` alongside `torch>=2.2.0`, `numpy>=1.26.0`, `pydantic>=2.7.0`, `pyyaml>=6.0.1`, and `loguru>=0.7.2`, and [`tests/test_zerosteer_v2_architecture.py`](../tests/test_zerosteer_v2_architecture.py) verifies CPU and Apple Silicon MPS parity without additional external packages.

[`pyproject.toml`](../pyproject.toml)은 추가 외부 패키지 없이 `../zerostreet-schema` 편집 가능 링크와 핵심 PyTorch/Pydantic 의존성을 연결하며, `tests/test_zerosteer_v2_architecture.py`를 통해 CPU 및 Apple Silicon MPS 정합성까지 로컬 검증할 수 있습니다.

```bash
uv sync --dev
uv run ruff check .
uv run pytest -v
```
