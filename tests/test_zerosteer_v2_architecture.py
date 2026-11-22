"""Comprehensive tests for ZeroSteer v2 architecture upgrades and Apple Silicon MPS parity.

ZeroSteer v2 아키텍처 고도화(RevIN, QK-Norm, Aux-Loss-Free MoE, SIGReg, 진단, 무결성) 및 MPS 패리티 테스트입니다.
"""

from datetime import date, datetime
from pathlib import Path

import pytest
import torch
from zerostreet_schema import HorizonMaskMode, RLRewardWeights, UserPolicyConfig

from zerosteer import (
    CrossAssetMaskedRelationalPretrainer,
    DifferentiableTrajectoryRLTrainer,
    FinancialCalendarEncoder,
    MoETransformerBackbone,
    ObservationFusionLayer,
    TimeseriesEncoder,
    TopKMoELayer,
    TrajectoryRLLoss,
    WeightAllocatorHead,
    ZeroSteer,
    ZeroSteerBackbone,
    ZeroSteerConfig,
    build_horizon_attention_mask,
    compute_effective_rank,
    compute_sigreg_loss,
    evaluate_backbone_representations,
    extract_calendar_features,
)


def _make_user_config(
    *,
    horizon: int = 4,
    mode: HorizonMaskMode = HorizonMaskMode.BIDIRECTIONAL_PORTFOLIO,
) -> UserPolicyConfig:
    return UserPolicyConfig(
        target_universe=["AAPL", "MSFT", "NVDA", "CASH"],
        portfolio_horizon=horizon,
        horizon_mask_mode=mode,
        weights_output_mode="entmax15",
        max_single_weight=0.60,
        rl_reward_weights=RLRewardWeights(return_20=0.5, sharpe_20=0.5),
    )


def _make_v2_config(**overrides: object) -> ZeroSteerConfig:
    defaults: dict[str, object] = {
        "d_model": 32,
        "nhead": 4,
        "nlayers": 3,
        "n_dense_layers": 1,
        "n_routed_experts": 4,
        "n_activated_experts": 2,
        "expert_inter_dim": 64,
        "dropout": 0.0,
        "portfolio_horizon": 4,
        "allocator_routed_experts": 4,
        "allocator_activated_experts": 2,
        "allocator_inter_dim": 48,
    }
    defaults.update(overrides)
    return ZeroSteerConfig(**defaults)  # type: ignore[arg-type]


def test_qknorm_sdpa_mask_prevents_future_horizon_leakage() -> None:
    """Verify bool attention mask inversion blocks P_1..P_{H-1} leakage under SDPA.

    SDPA bool 마스크 반전 처리가 P_1..P_{H-1}의 obs/과거 토큰 오염을 차단하는지 검증합니다.
    """
    torch.manual_seed(7)
    obs_num, asset_num, horizon, d_model = 3, 3, 4, 32

    # 1. Single-layer direct attention: BIDIRECTIONAL_PORTFOLIO + obs_attend_p0=True blocks P_1..P_{H-1} -> obs
    single_layer = MoETransformerBackbone(
        d_model=d_model,
        nhead=4,
        nlayers=1,
        n_dense_layers=1,
        dropout=0.0,
        use_qk_norm=True,
        ffn_activation="swiglu",
    ).eval()
    seq1 = torch.randn(2, obs_num + horizon * asset_num, d_model)
    seq_future = seq1.clone()
    seq_future[:, obs_num + asset_num :, :] += 15.0
    mask_bi = build_horizon_attention_mask(
        obs_num,
        asset_num,
        horizon,
        mode=HorizonMaskMode.BIDIRECTIONAL_PORTFOLIO,
        obs_attend_p0=True,
    )
    out1 = single_layer(seq1, attn_mask=mask_bi)
    out_future = single_layer(seq_future, attn_mask=mask_bi)
    assert torch.allclose(out1[:, :obs_num, :], out_future[:, :obs_num, :], atol=1e-6)

    # Multi-layer stack with obs_attend_p0=False guarantees full causal isolation across all layers.
    backbone = MoETransformerBackbone(
        d_model=d_model,
        nhead=4,
        nlayers=3,
        n_dense_layers=1,
        n_routed_experts=4,
        n_activated_experts=2,
        inter_dim=64,
        dropout=0.0,
        use_qk_norm=True,
        ffn_activation="swiglu",
        use_aux_loss_free=True,
    ).eval()

    # 2. CAUSAL_PORTFOLIO with obs_attend_p0=False: obs, P_0, P_1 must ignore P_2, P_3 across 3 layers
    seq_p23 = seq1.clone()
    seq_p23[:, obs_num + 2 * asset_num :, :] += 15.0
    mask_causal = build_horizon_attention_mask(
        obs_num,
        asset_num,
        horizon,
        mode=HorizonMaskMode.CAUSAL_PORTFOLIO,
        obs_attend_p0=False,
    )
    out_c1 = backbone(seq1, attn_mask=mask_causal)
    out_c2 = backbone(seq_p23, attn_mask=mask_causal)
    assert torch.allclose(
        out_c1[:, : obs_num + 2 * asset_num, :],
        out_c2[:, : obs_num + 2 * asset_num, :],
        atol=1e-6,
    )

    # 3. BACKWARD_DP with obs_attend_p0=False: P_2, P_3 must ignore P_0, P_1 across 3 layers
    seq_p01 = seq1.clone()
    seq_p01[:, obs_num : obs_num + 2 * asset_num, :] += 15.0
    mask_dp = build_horizon_attention_mask(
        obs_num,
        asset_num,
        horizon,
        mode=HorizonMaskMode.BACKWARD_DP,
        obs_attend_p0=False,
    )
    out_d1 = backbone(seq1, attn_mask=mask_dp)
    out_d2 = backbone(seq_p01, attn_mask=mask_dp)
    assert torch.allclose(
        out_d1[:, obs_num + 2 * asset_num :, :],
        out_d2[:, obs_num + 2 * asset_num :, :],
        atol=1e-6,
    )


def test_revin_zero_std_cash_safety_and_window_stats_film() -> None:
    """Ensure constant CASH assets never produce NaNs and Window-Stats FiLM preserves scale.

    변동성이 0인 CASH 자산에서도 NaN이 발생하지 않고 Window-Stats FiLM이 스케일을 구분함을 검증합니다.
    """
    torch.manual_seed(11)
    encoder = TimeseriesEncoder(
        in_features=6,
        d_model=32,
        dropout=0.0,
        use_revin_multiscale=True,
        temporal_kernel_sizes=(3, 7, 15),
    ).eval()

    # Asset 0: constant CASH (=1.0), Asset 1: constant shifted level (=50.0), Asset 2: volatile
    ohlcv = torch.randn(2, 3, 24, 6, requires_grad=True)
    with torch.no_grad():
        patched = ohlcv.clone()
        patched[:, 0, :, :] = 1.0
        patched[:, 1, :, :] = 50.0
    patched.requires_grad_(True)

    emb = encoder(patched)
    assert torch.isfinite(emb).all()
    assert not torch.allclose(emb[:, 0, :], emb[:, 1, :], atol=1e-4)

    emb.sum().backward()
    assert patched.grad is not None
    assert torch.isfinite(patched.grad).all()

    # Short sequence (< max kernel size 15) still preserves shape via length slicing.
    short_out = encoder(torch.randn(1, 2, 10, 6))
    assert short_out.shape == (1, 2, 32)
    assert torch.isfinite(short_out).all()


def test_calendar_encoder_and_observation_fusion() -> None:
    """Verify 10d deterministic calendar features condition ObservationFusionLayer cleanly.

    결정론적 10차원 금융 캘린더 피처 추출 및 ObservationFusionLayer 조건화 동작을 검증합니다.
    """
    feats = extract_calendar_features(
        [
            date(2025, 1, 6),  # Monday Q1
            datetime(2025, 12, 19, 15, 30),  # Friday Q4
            "2026-06-15T09:00:00Z",  # ISO Z timestamp
        ]
    )
    assert feats.shape == (3, 10)
    assert torch.all(feats <= 1.0) and torch.all(feats >= -1.0)
    assert not torch.allclose(feats[0], feats[1], atol=1e-3)

    cal_enc = FinancialCalendarEncoder(d_model=32)
    encoded_cal = cal_enc(feats)
    assert encoded_cal.shape == (3, 1, 32)

    fusion = ObservationFusionLayer(
        in_features=6,
        d_model=32,
        dropout=0.0,
        use_calendar_encoder=True,
    ).eval()
    ohlcv = torch.randn(2, 4, 20, 6)
    z_mm = torch.randn(2, 4, 128)
    obs_no_cal = fusion(ohlcv, z_mm=z_mm, calendar_feats=None)
    obs_with_cal = fusion(ohlcv, z_mm=z_mm, calendar_feats=feats[:2])
    assert obs_no_cal.shape == (2, 4, 32)
    assert not torch.allclose(obs_no_cal, obs_with_cal, atol=1e-5)


def test_aux_loss_free_moe_balancing_and_moe_allocator_head() -> None:
    """Test deepseek-v3 bias balancing recovers diversity and MoE Policy Head emits simplex weights.

    Aux-Loss-Free 바이어스 보정이 전문가 다양성을 회복하고 MoE 할당기 헤드가 심플렉스를 만족함을 검증합니다.
    """
    torch.manual_seed(23)
    moe = TopKMoELayer(
        d_model=16,
        inter_dim=32,
        n_routed_experts=4,
        n_activated_experts=1,
        dropout=0.0,
        use_aux_loss_free=True,
        score_func="sigmoid",
        bias_update_speed=0.25,
    ).train()

    with torch.no_grad():
        moe.gate.gate_network.weight.normal_(mean=0.0, std=0.05)  # type: ignore[union-attr]
        moe.gate.bias.copy_(torch.tensor([1.5, -1.5, -1.5, -1.5]))  # type: ignore[union-attr]

    x = torch.randn(8, 12, 16)
    _ = moe(x)
    initial_counts = list(moe.get_load_stats()["last_load_counts"])
    assert initial_counts[0] == 96

    for _ in range(12):
        _ = moe(x)
        moe.update_gate_bias()

    stats = moe.get_load_stats()
    assert stats["gate_bias"][0] < 1.5  # type: ignore[operator]
    assert sum(1 for c in stats["last_load_counts"] if c > 0) > 1

    head = WeightAllocatorHead(
        d_model=32,
        weights_output_mode="entmax15",
        dropout=0.0,
        use_moe_allocator=True,
        n_routed_experts=4,
        n_activated_experts=2,
        inter_dim=48,
    )
    embs = torch.randn(2, 4, 5, 32)
    valid_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    weights, logits = head(embs, valid_mask=valid_mask)
    assert weights.shape == (2, 4, 5)
    assert logits.shape == (2, 4, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-5)
    assert torch.allclose(weights[0, :, 3:], torch.zeros(4, 2), atol=1e-6)


def test_frozen_backbone_invariance_and_step_trajectory_rl_loss(tmp_path: Path) -> None:
    """Confirm freeze_backbone=True keeps ALL backbone parameters and MoE buffers bitwise invariant.

    freeze_backbone=True 컴파일 시 백본 가중치와 MoE 버퍼가 비트 단위로 불변임을 검증합니다.
    """
    torch.manual_seed(31)
    cfg = _make_v2_config(model_frz_update_freq=2, lr_warmup_steps=2)
    backbone = ZeroSteerBackbone(cfg)
    user_cfg = _make_user_config(horizon=4)

    pre_params = {k: v.detach().clone() for k, v in backbone.named_parameters()}
    pre_buffers = {k: v.detach().clone() for k, v in backbone.named_buffers()}

    batches = [
        {
            "ohlcv_seq": torch.randn(2, 4, 24, 6),
            "z_mm": torch.randn(2, 4, 128),
            "is_blackout": torch.zeros(2, 4, dtype=torch.bool),
            "calendar_feats": torch.randn(2, 10),
            "prev_weights": torch.full((2, 4), 0.25),
            "valid_mask": torch.ones(2, 4, dtype=torch.bool),
            "future_returns": torch.randn(2, 4, 4) * 0.01,
            "horizon_mask": torch.ones(2, 4),
            "bm_returns": torch.randn(2, 4) * 0.005,
        }
        for _ in range(3)
    ]

    trainer = DifferentiableTrajectoryRLTrainer(
        backbone=backbone,
        user_config=user_cfg,
        freeze_backbone=True,
        lr=1e-3,
    )
    pre_head = {k: v.detach().clone() for k, v in trainer.model.policy_head.state_dict().items()}

    save_path = tmp_path / "compiled_policy.pt"
    result = trainer.compile_policy(batches=batches, steps=4, save_path=save_path)
    assert "checkpoint_sha256" in result
    assert save_path.exists()
    assert (tmp_path / "compiled_policy.manifest.json").exists()

    for name, param in backbone.named_parameters():
        assert torch.equal(pre_params[name], param), f"Backbone param {name} changed!"
    for name, buf in backbone.named_buffers():
        assert torch.equal(pre_buffers[name], buf), f"Backbone buffer {name} changed!"

    post_head = trainer.model.policy_head.state_dict()
    assert any(not torch.equal(pre_head[k], post_head[k]) for k in pre_head)

    # Verify standalone RL loss exposes both trajectory_advantage and step_advantage
    loss_fn = TrajectoryRLLoss(cfg)
    w_a = torch.softmax(torch.randn(2, 4, 4, requires_grad=True), dim=-1)
    w_b = torch.softmax(torch.randn(2, 4, 4), dim=-1)
    losses = loss_fn.compute_loss(
        reward_weights=user_cfg.rl_reward_weights,
        return_data=batches[0]["future_returns"],  # type: ignore[arg-type]
        weights_traj=w_a,
        weights_frz=w_b,
    )
    assert {"total_loss", "policy_loss", "trust_penalty", "mean_advantage", "trajectory_advantage", "step_advantage"} <= set(
        losses.keys()
    )


def test_sigreg_diagnostics_sha256_tamper_and_legacy_v1(tmp_path: Path) -> None:
    """Verify SIGReg anti-collapse, SVD diagnostics, SHA-256 tamper detection, and legacy_v1 preset.

    SIGReg 붕괴 방지, SVD 진단, SHA-256 변조 탐지 및 legacy_v1 어블레이션 프리셋을 검증합니다.
    """
    gen = torch.Generator(device="cpu").manual_seed(99)
    iso = torch.randn(256, 16, generator=gen)
    collapsed = torch.randn(1, 16, generator=gen).expand(256, 16)
    assert compute_sigreg_loss(iso, num_slices=32, generator=gen) < compute_sigreg_loss(
        collapsed, num_slices=32, generator=gen
    )

    eff_iso, norm_iso = compute_effective_rank(iso)
    eff_col, norm_col = compute_effective_rank(collapsed)
    assert eff_iso > eff_col
    assert norm_iso > norm_col

    cfg = _make_v2_config()
    backbone = ZeroSteerBackbone(cfg)
    pretrainer = CrossAssetMaskedRelationalPretrainer(backbone, mask_ratio=0.3, lr=1e-3)
    batches = [
        {
            "ohlcv_seq": torch.randn(2, 4, 24, 6),
            "z_mm": torch.randn(2, 4, 128),
            "is_blackout": torch.zeros(2, 4, dtype=torch.bool),
            "calendar_feats": torch.randn(2, 10),
        }
        for _ in range(2)
    ]
    ckpt_path = tmp_path / "backbone_v2.pt"
    history = pretrainer.train_steps(batches, save_path=ckpt_path)
    assert "sigreg_loss" in history[-1]
    assert ckpt_path.exists()
    assert (tmp_path / "backbone_v2.manifest.json").exists()

    report = evaluate_backbone_representations(
        backbone,
        batches[0]["ohlcv_seq"],  # type: ignore[arg-type]
        z_mm=batches[0]["z_mm"],
        calendar_feats=batches[0]["calendar_feats"],
    )
    assert report.num_tokens == 8
    assert report.effective_rank > 1.0
    assert report.moe_load_stats

    loaded = ZeroSteerBackbone.load_backbone(ckpt_path, verify_sha256=True)
    assert isinstance(loaded, ZeroSteerBackbone)

    # Corrupt 1 byte in backbone checkpoint -> verify_sha256=True must raise ValueError
    raw_bytes = bytearray(ckpt_path.read_bytes())
    raw_bytes[-1] ^= 0xFF
    ckpt_path.write_bytes(bytes(raw_bytes))
    with pytest.raises(ValueError, match="SHA-256"):
        ZeroSteerBackbone.load_backbone(ckpt_path, verify_sha256=True)

    # Verify policy head save/load & tamper rejection
    user_cfg = _make_user_config(horizon=4)
    model = ZeroSteer(backbone=backbone, user_config=user_cfg)
    policy_path = tmp_path / "policy_v2.pt"
    sha_pol = model.save_policy_head(policy_path)
    assert len(sha_pol) == 64
    model.load_policy_head(policy_path, verify_sha256=True)

    pol_bytes = bytearray(policy_path.read_bytes())
    pol_bytes[-1] ^= 0xFF
    policy_path.write_bytes(bytes(pol_bytes))
    with pytest.raises(ValueError, match="SHA-256"):
        model.load_policy_head(policy_path, verify_sha256=True)

    # Verify legacy_v1 ablation preset runs cleanly end-to-end
    v1_cfg = ZeroSteerConfig.legacy_v1(d_model=32, nhead=4, nlayers=2, n_dense_layers=1, portfolio_horizon=4)
    v1_model = ZeroSteer(backbone=ZeroSteerBackbone(v1_cfg), user_config=user_cfg).eval()
    v1_out = v1_model(ohlcv_seq=torch.randn(2, 4, 24, 6), z_mm=torch.randn(2, 4, 32))
    assert v1_out.weights_long_trajectory.shape == (2, 4, 4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple Silicon MPS unavailable")
def test_mps_forward_backward_parity() -> None:
    """Validate CPU vs Apple Silicon MPS forward parity and MPS backward/SIGReg/diagnostics execution.

    CPU와 Apple Silicon MPS 간 순전파 정합성(atol=1e-4) 및 MPS 역전파·SIGReg·진단 실행을 검증합니다.
    """
    import copy

    torch.manual_seed(123)
    cfg = _make_v2_config()
    user_cfg = _make_user_config(horizon=4)
    cpu_model = ZeroSteer(backbone=ZeroSteerBackbone(cfg), user_config=user_cfg).eval()
    mps_model = copy.deepcopy(cpu_model).to("mps").eval()

    ohlcv_cpu = torch.randn(2, 4, 24, 6)
    z_cpu = torch.randn(2, 4, 128)
    cal_cpu = torch.randn(2, 10)
    prev_cpu = torch.full((2, 4), 0.25)
    valid_cpu = torch.ones(2, 4, dtype=torch.bool)

    with torch.no_grad():
        out_cpu = cpu_model(
            ohlcv_seq=ohlcv_cpu,
            z_mm=z_cpu,
            prev_weights=prev_cpu,
            valid_mask=valid_cpu,
            calendar_feats=cal_cpu,
        )
        out_mps = mps_model(
            ohlcv_seq=ohlcv_cpu.to("mps"),
            z_mm=z_cpu.to("mps"),
            prev_weights=prev_cpu.to("mps"),
            valid_mask=valid_cpu.to("mps"),
            calendar_feats=cal_cpu.to("mps"),
        )

    assert torch.allclose(
        out_cpu.weights_long_trajectory,
        out_mps.weights_long_trajectory.cpu(),
        atol=1e-4,
    )
    assert torch.allclose(
        out_cpu.final_embs_obs,
        out_mps.final_embs_obs.cpu(),
        atol=1e-4,
    )

    # Verify MPS backward + gate bias update + SIGReg + CPU-safe SVD diagnostics
    mps_model.train()
    train_out = mps_model(
        ohlcv_seq=ohlcv_cpu.to("mps"),
        z_mm=z_cpu.to("mps"),
        prev_weights=prev_cpu.to("mps"),
        valid_mask=valid_cpu.to("mps"),
        calendar_feats=cal_cpu.to("mps"),
    )
    loss = train_out.weights_long_trajectory.sum() + compute_sigreg_loss(
        train_out.final_embs_obs.reshape(-1, cfg.d_model)
    )
    loss.backward()
    mps_model.update_gate_bias(include_backbone=True)
    report = evaluate_backbone_representations(
        mps_model.backbone,
        ohlcv_cpu.to("mps"),
        z_mm=z_cpu.to("mps"),
        calendar_feats=cal_cpu.to("mps"),
    )
    assert report.effective_rank > 1.0
