"""미분 가능 20일 가상 투자 궤적 강화학습(Trajectory RL) 손실 및 컴파일 트레이너 모듈."""

import copy
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from zerostreet_schema import RLRewardWeights, UserPolicyConfig

from hyperalpha_vla.config import HyperAlphaVLAConfig
from hyperalpha_vla.model import HyperAlphaVLA, HyperAlphaVLABackbone


class TrajectoryRLLoss:
    """드리프트 사후 비중, L1/L2 거래비용, Dense-Sparse STE, rms_softclip 기반 Trajectory RL 손실기."""

    def __init__(self, config: HyperAlphaVLAConfig | None = None, eps: float = 1e-6) -> None:
        self.config = config or HyperAlphaVLAConfig()
        self.eps = eps

    def _get_effective_horizon_mask(
        self,
        batch_size: int,
        window: int,
        device: torch.device,
        dtype: torch.dtype,
        horizon_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if horizon_mask is not None:
            h_mask = horizon_mask.to(device=device, dtype=dtype)
            if h_mask.dim() == 1:
                h_mask = h_mask.unsqueeze(0).expand(batch_size, -1)
            if h_mask.shape[1] >= window:
                mask_slice = h_mask[:, :window]
            else:
                pad = torch.zeros((batch_size, window - h_mask.shape[1]), device=device, dtype=dtype)
                mask_slice = torch.cat([h_mask, pad], dim=1)
        else:
            mask_slice = torch.ones((batch_size, window), device=device, dtype=dtype)

        valid_k = mask_slice.sum(dim=1).clamp(min=1.0)
        sample_conf = torch.sqrt(valid_k / float(max(1, window)))
        return mask_slice, valid_k, sample_conf

    def simulate_trajectory(
        self,
        return_data: torch.Tensor,
        weights_traj: torch.Tensor,
        prev_weights: torch.Tensor | None = None,
        horizon_mask: torch.Tensor | None = None,
        quadratic_turnover_coeff: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """H일 가상 계좌 운용(드리프트 사후 비중 + L1 수수료 + L2 시장충격 차감)을 수행합니다."""
        w_3d = weights_traj.unsqueeze(1) if weights_traj.dim() == 2 else weights_traj
        rets_3d = return_data.unsqueeze(1) if return_data.dim() == 2 else return_data
        bsz, h_len, _n_assets = w_3d.shape
        window = max(1, min(h_len, rets_3d.shape[1]))

        rets_slice = rets_3d[:, :window, :]
        weights_slice = w_3d[:, :window, :]
        mask_slice, valid_k, sample_conf = self._get_effective_horizon_mask(
            batch_size=bsz,
            window=window,
            device=weights_slice.device,
            dtype=weights_slice.dtype,
            horizon_mask=horizon_mask,
        )

        if prev_weights is not None:
            p_w0 = prev_weights.to(device=weights_slice.device, dtype=weights_slice.dtype)
            if p_w0.dim() == 3:
                p_w0 = p_w0[:, 0, :]
        else:
            p_w0 = weights_slice[:, 0, :].detach()

        if window > 1:
            if self.config.use_drifted_weights:
                growth_prev = weights_slice[:, :-1, :] * (
                    1.0 + rets_slice[:, :-1, :]
                ).clamp(min=1e-4)
                drifted_prev = growth_prev / growth_prev.sum(dim=-1, keepdim=True).clamp(
                    min=self.eps
                )
            else:
                drifted_prev = weights_slice[:, :-1, :]
            pre_trade_weights = torch.cat([p_w0.unsqueeze(1), drifted_prev], dim=1)
        else:
            pre_trade_weights = p_w0.unsqueeze(1)

        weight_deltas = weights_slice - pre_trade_weights
        daily_l1_to = torch.abs(weight_deltas).sum(dim=-1)
        daily_l2_impact = (weight_deltas**2).sum(dim=-1)
        daily_gross_ret = (weights_slice * rets_slice).sum(dim=-1)

        q_coeff = (
            float(quadratic_turnover_coeff)
            if quadratic_turnover_coeff is not None
            else self.config.quadratic_turnover_coeff
        )
        daily_net_ret = (
            daily_gross_ret - self.config.fee_rate * daily_l1_to - q_coeff * daily_l2_impact
        )
        return {
            "daily_net_ret": daily_net_ret,
            "daily_gross_ret": daily_gross_ret,
            "daily_l1_to": daily_l1_to,
            "daily_l2_impact": daily_l2_impact,
            "mask_slice": mask_slice,
            "valid_k": valid_k,
            "sample_conf": sample_conf,
            "weights_slice": weights_slice,
        }

    def _discounted_weights(self, mask_slice: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(mask_slice.shape[1], device=mask_slice.device, dtype=mask_slice.dtype)
        disc = (self.config.horizon_discount_gamma**steps).unsqueeze(0) * mask_slice
        return disc / disc.sum(dim=1, keepdim=True).clamp(min=self.eps)

    def _normalize_metric(self, delta: torch.Tensor) -> torch.Tensor:
        if delta.shape[0] <= 1:
            scale = torch.ones_like(delta)
        else:
            scale = delta.detach().std(dim=0, unbiased=False, keepdim=True).clamp(min=self.eps)
        z = delta / scale
        return 0.75 * torch.tanh(z) + 0.25 * torch.asinh(z)

    def _compute_metrics(
        self,
        sim: dict[str, torch.Tensor],
        bm_returns: torch.Tensor | None,
        max_single_weight: float,
    ) -> dict[str, torch.Tensor]:
        net_ret = sim["daily_net_ret"]
        mask = sim["mask_slice"]
        valid_k = sim["valid_k"]
        weights_slice = sim["weights_slice"]

        masked_net = net_ret * mask
        geo_ret = torch.exp(
            torch.log((1.0 + masked_net).clamp(min=1e-4)).sum(dim=1) / valid_k
        ) - 1.0
        mean_ret = masked_net.sum(dim=1) / valid_k

        var_ret = (((masked_net - mean_ret.unsqueeze(1)) ** 2) * mask).sum(dim=1) / valid_k
        std_ret = torch.sqrt(var_ret + self.eps)
        sharpe = torch.tanh((mean_ret / std_ret) * math.sqrt(250.0) / 3.0)

        downside = torch.clamp(masked_net, max=0.0)
        down_std = torch.sqrt(((downside**2) * mask).sum(dim=1) / valid_k + self.eps)
        sortino = torch.tanh((mean_ret / down_std) * math.sqrt(250.0) / 4.0)

        pos_days = ((masked_net > 0).float() * mask).sum(dim=1) / valid_k
        neg_days = ((masked_net < 0).float() * mask).sum(dim=1) / valid_k
        fip_return = geo_ret * (1.0 + pos_days - neg_days)

        # Differentiable CVaR (lower tail average loss, negated so higher is better)
        losses = -masked_net
        tail_k = max(1, math.ceil(self.config.cvar_alpha * net_ret.shape[1]))
        worst_losses, _ = torch.topk(losses, k=tail_k, dim=1)
        cvar_reward = -F.softplus(worst_losses.mean(dim=1), beta=20.0)

        # Differentiable Compounded MDD (negated so smaller drawdown is higher reward)
        cum_wealth = torch.cumprod((1.0 + masked_net).clamp(min=1e-4), dim=1)
        running_peak = torch.cummax(cum_wealth, dim=1).values.clamp(min=self.eps)
        drawdowns = (running_peak - cum_wealth) / running_peak
        mdd_reward = -drawdowns.max(dim=1).values

        # Max single asset weight constraint hinge penalty
        excess_w = torch.clamp(weights_slice - max_single_weight, min=0.0)
        hinge = (excess_w / self.config.max_weight_scale) + (
            excess_w / self.config.max_weight_scale
        ) ** 2
        disc_w = self._discounted_weights(mask)
        max_w_reward = -(hinge.sum(dim=-1) * disc_w).sum(dim=1)

        if bm_returns is not None:
            bm_slice = bm_returns[:, : net_ret.shape[1]].to(
                device=net_ret.device, dtype=net_ret.dtype
            )
            active_ret = (net_ret - bm_slice) * mask
            active_mean = active_ret.sum(dim=1) / valid_k
            active_std = torch.sqrt(
                (((active_ret - active_mean.unsqueeze(1)) ** 2) * mask).sum(dim=1) / valid_k
                + self.eps
            )
            active_down = torch.sqrt(
                ((torch.clamp(active_ret, max=0.0) ** 2) * mask).sum(dim=1) / valid_k + self.eps
            )
            ir = torch.tanh((active_mean / active_std) * math.sqrt(250.0) / 3.0)
            active_sortino = torch.tanh((active_mean / active_down) * math.sqrt(250.0) / 4.0)
            excess_ret = active_mean
        else:
            ir = sharpe
            active_sortino = sortino
            excess_ret = geo_ret

        return {
            "return_20": geo_ret,
            "excess_return_20": excess_ret,
            "fip_return_20": fip_return,
            "sortino_20": sortino,
            "sharpe_20": sharpe,
            "information_ratio_20": ir,
            "active_sortino_20": active_sortino,
            "cvar_20": cvar_reward,
            "mdd_20": mdd_reward,
            "vol_20": -std_ret,
            "max_weight_limit": max_w_reward,
        }

    def compute_loss(
        self,
        reward_weights: RLRewardWeights,
        return_data: torch.Tensor,
        weights_traj: torch.Tensor,
        weights_frz: torch.Tensor,
        logits_traj: torch.Tensor | None = None,
        prev_weights: torch.Tensor | None = None,
        horizon_mask: torch.Tensor | None = None,
        bm_returns: torch.Tensor | None = None,
        max_single_weight: float = 0.60,
    ) -> dict[str, torch.Tensor]:
        """Dense-Sparse STE와 rms_softclip 어드밴티지를 결합해 최종 RL 손실을 계산합니다."""
        effective_w = weights_traj
        if (
            logits_traj is not None
            and self.config.ste_softmax_alpha > 0.0
            and effective_w.requires_grad
        ):
            dense_probs = (logits_traj / max(self.config.temperature, 1e-4)).softmax(dim=-1)
            effective_w = effective_w + self.config.ste_softmax_alpha * (
                dense_probs - dense_probs.detach()
            )

        q_coeff = (
            reward_weights.quadratic_turnover_coeff
            if reward_weights.quadratic_turnover_coeff > 0.0
            else self.config.quadratic_turnover_coeff
        )
        sim_policy = self.simulate_trajectory(
            return_data=return_data,
            weights_traj=effective_w,
            prev_weights=prev_weights,
            horizon_mask=horizon_mask,
            quadratic_turnover_coeff=q_coeff,
        )
        sim_frz = self.simulate_trajectory(
            return_data=return_data,
            weights_traj=weights_frz.detach(),
            prev_weights=prev_weights,
            horizon_mask=horizon_mask,
            quadratic_turnover_coeff=q_coeff,
        )

        m_pol = self._compute_metrics(sim_policy, bm_returns, max_single_weight)
        m_frz = self._compute_metrics(sim_frz, bm_returns, max_single_weight)

        weight_map = {
            "return_20": reward_weights.return_20,
            "excess_return_20": reward_weights.excess_return_20,
            "fip_return_20": reward_weights.fip_return_20,
            "sortino_20": reward_weights.sortino_20,
            "sharpe_20": reward_weights.sharpe_20,
            "information_ratio_20": reward_weights.information_ratio_20,
            "active_sortino_20": reward_weights.active_sortino_20,
            "cvar_20": reward_weights.cvar_20,
            "mdd_20": reward_weights.mdd_20,
            "vol_20": reward_weights.vol_20,
            "max_weight_limit": reward_weights.max_weight_limit,
        }

        combined_adv = torch.zeros_like(sim_policy["sample_conf"])
        for name, weight_val in weight_map.items():
            if weight_val <= 0.0:
                continue
            delta = m_pol[name] - m_frz[name].detach()
            if self.config.step_reward_weight + self.config.trajectory_reward_weight > 0:
                norm_delta = self._normalize_metric(delta)
                rms = torch.sqrt(torch.mean(norm_delta**2) + self.eps).clamp(
                    min=self.config.advantage_rms_floor
                )
                clip_c = self.config.advantage_clip_val
                adv_m = clip_c * torch.tanh(norm_delta / (clip_c * rms))
                combined_adv = combined_adv + weight_val * adv_m * sim_policy["sample_conf"]

        policy_loss = -combined_adv.mean()

        disc_w = self._discounted_weights(sim_policy["mask_slice"])
        step_l1 = torch.abs(sim_policy["weights_slice"] - sim_frz["weights_slice"]).sum(dim=-1)
        trust_penalty = (step_l1 * disc_w).sum(dim=1).mean()
        total_loss = policy_loss + self.config.beta_trust_region * trust_penalty

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "trust_penalty": trust_penalty,
            "mean_advantage": combined_adv.mean().detach(),
        }


class DifferentiableTrajectoryRLTrainer:
    """유저별 Policy Head 컴파일(On-Demand Trajectory RL) 전용 트레이너."""

    def __init__(
        self,
        backbone: HyperAlphaVLABackbone | None = None,
        user_config: UserPolicyConfig | None = None,
        freeze_backbone: bool = True,
        lr: float = 1e-3,
    ) -> None:
        if user_config is None:
            raise ValueError(
                "user_config is required to instantiate DifferentiableTrajectoryRLTrainer. "
                "Policy compilation must always originate from the user agent."
            )
        self.user_config = user_config
        self.model = HyperAlphaVLA(backbone=backbone, user_config=user_config)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
        self.loss_fn = TrajectoryRLLoss(config=self.model.backbone.config)
        self.frozen_model = copy.deepcopy(self.model).eval()
        for param in self.frozen_model.parameters():
            param.requires_grad = False

    def compile_policy(
        self,
        batches: list[dict[str, torch.Tensor | None]],
        steps: int | None = None,
        save_path: str | Path | None = None,
    ) -> dict[str, str | float]:
        """주어진 배치에 대해 사용자 전용 Policy Head를 컴파일하고 체크포인트 SHA-256을 반환합니다."""
        if not batches:
            raise ValueError("batches must contain at least one training batch.")

        self.model.train()
        if self.freeze_backbone:
            self.model.backbone.eval()

        total_steps = int(steps if steps is not None else len(batches))
        last_loss = 0.0
        last_adv = 0.0

        for step_idx in range(total_steps):
            batch = batches[step_idx % len(batches)]
            self.optimizer.zero_grad()

            ohlcv_seq = batch["ohlcv_seq"]
            z_mm = batch.get("z_mm")
            is_blackout = batch.get("is_blackout")
            prev_weights = batch.get("prev_weights")
            valid_mask = batch.get("valid_mask")
            future_returns = batch["future_returns"]
            horizon_mask = batch.get("horizon_mask")
            bm_returns = batch.get("bm_returns")

            assert ohlcv_seq is not None and future_returns is not None
            outputs = self.model(
                ohlcv_seq=ohlcv_seq,
                z_mm=z_mm,
                is_blackout=is_blackout,
                prev_weights=prev_weights,
                valid_mask=valid_mask,
                horizon=self.user_config.portfolio_horizon,
            )
            with torch.no_grad():
                frz_outputs = self.frozen_model(
                    ohlcv_seq=ohlcv_seq,
                    z_mm=z_mm,
                    is_blackout=is_blackout,
                    prev_weights=prev_weights,
                    valid_mask=valid_mask,
                    horizon=self.user_config.portfolio_horizon,
                )

            losses = self.loss_fn.compute_loss(
                reward_weights=self.user_config.rl_reward_weights,
                return_data=future_returns,
                weights_traj=outputs.weights_long_trajectory,
                weights_frz=frz_outputs.weights_long_trajectory,
                logits_traj=outputs.logits_long_trajectory,
                prev_weights=prev_weights,
                horizon_mask=horizon_mask,
                bm_returns=bm_returns,
                max_single_weight=self.user_config.max_single_weight,
            )
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.policy_head.parameters(), max_norm=1.0)
            self.optimizer.step()

            last_loss = float(losses["total_loss"].item())
            last_adv = float(losses["mean_advantage"].item())
            logger.debug("Foundry compile step={} loss={:.6f}", step_idx, last_loss)

        hasher = hashlib.sha256()
        for key in sorted(self.model.policy_head.state_dict().keys()):
            tensor_bytes = self.model.policy_head.state_dict()[key].detach().cpu().numpy().tobytes()
            hasher.update(key.encode("utf-8"))
            hasher.update(tensor_bytes)
        ckpt_sha256 = hasher.hexdigest()

        if save_path is not None:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "user_config": self.user_config.model_dump(mode="json"),
                    "policy_head_state_dict": self.model.policy_head.state_dict(),
                    "checkpoint_sha256": ckpt_sha256,
                },
                out,
            )

        return {
            "checkpoint_sha256": ckpt_sha256,
            "final_loss": last_loss,
            "mean_advantage": last_adv,
            "steps_completed": float(total_steps),
        }
