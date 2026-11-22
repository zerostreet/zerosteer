from pathlib import Path

import torch
from zerostreet_schema import RLRewardWeights, UserPolicyConfig

from zerosteer import (
    CrossAssetMaskedRelationalPretrainer,
    DifferentiableTrajectoryRLTrainer,
    TrajectoryRLLoss,
    ZeroSteerBackbone,
    ZeroSteerConfig,
    run_oos_comparison,
)


def _tiny_cfg() -> ZeroSteerConfig:
    return ZeroSteerConfig(
        d_model=32,
        nhead=4,
        nlayers=2,
        n_dense_layers=1,
        n_routed_experts=4,
        n_activated_experts=2,
        expert_inter_dim=64,
        seq_len=12,
        timeseries_features=6,
        portfolio_horizon=4,
        max_horizon_num=8,
    )


def test_cross_asset_masked_relational_pretrainer_future_label_free(tmp_path: Path):
    torch.manual_seed(42)
    cfg = _tiny_cfg()
    backbone = ZeroSteerBackbone(cfg)
    pretrainer = CrossAssetMaskedRelationalPretrainer(backbone=backbone, mask_ratio=0.5, lr=1e-3)

    batches = [
        {
            "ohlcv_seq": torch.randn(3, 4, cfg.seq_len, cfg.timeseries_features),
            "z_mm": torch.randn(3, 4, 32),
        }
        for _ in range(3)
    ]
    ckpt = tmp_path / "pretrained_backbone.pt"
    history = pretrainer.train_steps(batches, save_path=ckpt)
    assert len(history) == 3
    assert ckpt.exists()
    assert all("relational_loss" in m for m in history)


def test_trajectory_rl_trainer_freezes_backbone_and_compiles_policy(tmp_path: Path):
    torch.manual_seed(7)
    cfg = _tiny_cfg()
    backbone = ZeroSteerBackbone(cfg)
    before_backbone_sum = sum(float(p.sum().item()) for p in backbone.parameters())

    user_cfg = UserPolicyConfig(
        target_universe=["ASSET_A", "ASSET_B", "ASSET_C", "CASH"],
        rl_reward_weights=RLRewardWeights(
            excess_return_20=0.8,
            sortino_20=0.6,
            cvar_20=1.2,
            mdd_20=0.5,
            max_weight_limit=0.2,
            quadratic_turnover_coeff=0.05,
        ),
        portfolio_horizon=4,
        max_single_weight=0.5,
    )
    trainer = DifferentiableTrajectoryRLTrainer(
        backbone=backbone,
        user_config=user_cfg,
        freeze_backbone=True,
        lr=2e-3,
    )

    batches = [
        {
            "ohlcv_seq": torch.randn(4, 4, cfg.seq_len, cfg.timeseries_features),
            "z_mm": torch.randn(4, 4, 128),
            "prev_weights": torch.full((4, 4), 0.25),
            "future_returns": 0.01 * torch.randn(4, 4, 4),
            "horizon_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0]] * 3 + [[1.0, 1.0, 0.0, 0.0]]),
            "bm_returns": 0.002 * torch.randn(4, 4),
        }
        for _ in range(3)
    ]
    policy_path = tmp_path / "user_policy_ckpt.pt"
    result = trainer.compile_policy(batches=batches, steps=4, save_path=policy_path)
    assert policy_path.exists()
    assert len(str(result["checkpoint_sha256"])) == 64

    after_backbone_sum = sum(float(p.sum().item()) for p in trainer.model.backbone.parameters())
    assert abs(after_backbone_sum - before_backbone_sum) < 1e-6

    # Also verify STE gradient flows even to low-logit elements & OOS comparison works
    loss_fn = TrajectoryRLLoss(config=cfg)
    logits = torch.tensor([[[4.0, -6.0, -7.0, -8.0]] * 4], requires_grad=True)
    from zerosteer.activations import entmax15

    w_ent = entmax15(logits, dim=-1)
    assert (w_ent[0, 0, 1:] == 0.0).all()
    sim_loss = loss_fn.compute_loss(
        reward_weights=user_cfg.rl_reward_weights,
        return_data=0.01 * torch.randn(1, 4, 4),
        weights_traj=w_ent,
        weights_frz=w_ent.detach(),
        logits_traj=logits,
    )
    sim_loss["total_loss"].backward()
    assert logits.grad is not None and torch.abs(logits.grad[..., -1]).sum().item() > 0.0

    report = run_oos_comparison(
        model=trainer.model,
        ohlcv_seq=torch.randn(10, 4, cfg.seq_len, cfg.timeseries_features),
        z_mm=torch.randn(10, 4, 128),
        future_day1_returns=0.01 * torch.randn(10, 4),
    )
    assert isinstance(report.sharpe_delta, float)
