from pathlib import Path

import pytest
import torch
from zerostreet_schema import HorizonMaskMode, RLRewardWeights, UserPolicyConfig

from hyperalpha_vla import (
    DifferentiableTrajectoryRLTrainer,
    HyperAlphaVLA,
    HyperAlphaVLABackbone,
    HyperAlphaVLAConfig,
    build_horizon_attention_mask,
    load_user_policy_config,
)


def _small_config() -> HyperAlphaVLAConfig:
    return HyperAlphaVLAConfig(
        d_model=32,
        nhead=4,
        nlayers=2,
        n_dense_layers=1,
        n_routed_experts=4,
        n_activated_experts=2,
        expert_inter_dim=64,
        seq_len=16,
        timeseries_features=6,
        portfolio_horizon=4,
        max_horizon_num=8,
    )


def _sample_user_config() -> UserPolicyConfig:
    return UserPolicyConfig(
        target_universe=["ASSET_A", "ASSET_B", "ASSET_C", "CASH"],
        rl_reward_weights=RLRewardWeights(sortino_20=1.0, cvar_20=0.5),
        portfolio_horizon=4,
    )


def test_policy_head_isolation_requires_user_config():
    backbone = HyperAlphaVLABackbone(_small_config())
    with pytest.raises(ValueError, match="user_config is required"):
        HyperAlphaVLA(backbone=backbone, user_config=None)

    with pytest.raises(ValueError, match="user_config is required"):
        DifferentiableTrajectoryRLTrainer(backbone=backbone, user_config=None)


def test_load_config_example_yaml_and_save_load_backbone(tmp_path: Path):
    example_yaml = Path(__file__).resolve().parents[1] / "config.example.yaml"
    user_cfg = load_user_policy_config(example_yaml)
    assert len(user_cfg.target_universe) == 4

    backbone = HyperAlphaVLABackbone(_small_config())
    assert torch.all(backbone.horizon_encodings.weight[0] == 0.0)

    ckpt_path = tmp_path / "backbone.pt"
    backbone.save_backbone(ckpt_path)
    loaded = HyperAlphaVLABackbone.load_backbone(ckpt_path)
    assert not any("policy_head" in k for k in loaded.state_dict())


def test_dynamic_embedding_dims_blackout_and_valid_mask():
    cfg = _small_config()
    backbone = HyperAlphaVLABackbone(cfg)
    model = HyperAlphaVLA(backbone=backbone, user_config=_sample_user_config())
    model.eval()

    bsz, n_assets, seq_len, n_feat = 2, 4, cfg.seq_len, cfg.timeseries_features
    ohlcv = torch.randn(bsz, n_assets, seq_len, n_feat)
    prev_w = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1]])
    valid_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

    # 1) Full blackout (z_mm=None, OHLCV-only mode)
    out_none = model(ohlcv, z_mm=None, prev_weights=prev_w, valid_mask=valid_mask)
    assert out_none.weights_long.shape == (bsz, n_assets)
    assert out_none.weights_long_trajectory.shape == (bsz, 4, n_assets)
    assert torch.allclose(out_none.weights_long.sum(dim=-1), torch.ones(bsz), atol=1e-4)
    # Asset 2 masked out in sample 0 across all horizon steps
    assert torch.all(out_none.weights_long_trajectory[0, :, 2] == 0.0)

    # 2) Variable MRL dimensions (32, 128, 256) with partial asset blackout
    is_blk = torch.tensor([[False, True, False, False], [False, False, False, False]])
    for dim in (32, 128, 256):
        z_mm = torch.randn(bsz, n_assets, dim)
        out_mm = model(ohlcv, z_mm=z_mm, is_blackout=is_blk, valid_mask=valid_mask)
        assert torch.allclose(out_mm.weights_long.sum(dim=-1), torch.ones(bsz), atol=1e-4)


def test_horizon_attention_mask_modes_and_temporal_ensemble():
    mask_bi = build_horizon_attention_mask(
        obs_num=4, step_asset_num=4, horizon=3, mode=HorizonMaskMode.BIDIRECTIONAL_PORTFOLIO
    )
    assert mask_bi is not None
    # obs_embs (0..3) cannot attend to P_1..P_2 (8..15)
    assert mask_bi[:4, 8:].all()
    # obs_embs can attend to P_0 (4..7) when obs_attend_p0=True
    assert not mask_bi[:4, 4:8].any()

    mask_causal = build_horizon_attention_mask(
        obs_num=4, step_asset_num=4, horizon=3, mode=HorizonMaskMode.CAUSAL_PORTFOLIO
    )
    assert mask_causal is not None
    # P_0 (4..7) cannot attend to P_1 (8..11)
    assert mask_causal[4:8, 8:].all()

    mask_bw = build_horizon_attention_mask(
        obs_num=4, step_asset_num=4, horizon=3, mode=HorizonMaskMode.BACKWARD_DP
    )
    assert mask_bw is not None
    # P_1 (8..11) cannot attend to P_0 (4..7)
    assert mask_bw[8:12, 4:8].all()

    trajs = [torch.softmax(torch.randn(2, 4, 4), dim=-1) for _ in range(3)]
    blended = HyperAlphaVLA.blend_temporal_ensemble(trajs, decay_beta=0.5)
    assert blended.shape == (2, 4)
    assert torch.allclose(blended.sum(dim=-1), torch.ones(2), atol=1e-5)
