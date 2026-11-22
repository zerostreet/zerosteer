"""Future-Label-Free cross-asset masked relational pretraining module.

Future-Label-Free Cross-Asset 마스킹 관계형 백본 사전학습 모듈입니다.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

from hyperalpha_vla.model import HyperAlphaVLABackbone


class CrossAssetMaskedRelationalPretrainer:
    """Reconstructs masked intra-timestamp asset tokens via cross-asset attention.

    Zero advisory labels: uses 0% future return, ranking, volatility, or crash labels.

    동일 시점(t) 내 마스킹된 자산 토큰을 나머지 종목/지수 컨텍스트로 복원하는 사전학습기입니다.
    미래 수익률, 랭킹, 변동성/폭락 라벨을 0% 사용합니다(Future-Label-Free).
    """

    def __init__(
        self,
        backbone: HyperAlphaVLABackbone,
        mask_ratio: float = 0.25,
        lr: float = 1e-3,
    ) -> None:
        self.backbone = backbone
        self.mask_ratio = mask_ratio
        d_model = backbone.config.d_model
        self.rel_mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.rel_mask_token, mean=0.0, std=0.02)
        self.recon_head = nn.Linear(d_model, d_model)
        self.optimizer = torch.optim.AdamW(
            [*self.backbone.parameters(), self.rel_mask_token, *self.recon_head.parameters()],
            lr=lr,
            weight_decay=0.01,
        )

    def compute_relational_loss(
        self,
        ohlcv_seq: torch.Tensor,
        z_mm: torch.Tensor | None = None,
        is_blackout: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Mask random same-timestamp observation tokens and compute MSE + cosine loss.

        동일 시점 관측 토큰 일부를 마스킹하고 교차 어텐션으로 복원 손실을 계산합니다.
        """
        obs_embs = self.backbone.encode_observations(
            ohlcv_seq=ohlcv_seq, z_mm=z_mm, is_blackout=is_blackout
        )
        bsz, n_assets, _ = obs_embs.shape
        mask = torch.rand((bsz, n_assets), device=obs_embs.device) < self.mask_ratio
        # Ensure at least one asset is masked per sample for stable gradient signal
        none_masked = ~mask.any(dim=1)
        if none_masked.any():
            mask[none_masked, 0] = True

        target = obs_embs.detach()
        masked_obs = torch.where(
            mask.unsqueeze(-1),
            self.rel_mask_token.to(device=obs_embs.device, dtype=obs_embs.dtype).expand_as(
                obs_embs
            ),
            obs_embs,
        )
        encoded = self.backbone.transformer(masked_obs)
        pred = self.recon_head(encoded)

        pred_masked = pred[mask]
        target_masked = target[mask]
        mse_loss = F.mse_loss(pred_masked, target_masked)
        cos_sim = F.cosine_similarity(pred_masked, target_masked, dim=-1).mean()
        loss = mse_loss + 0.1 * (1.0 - cos_sim)
        return loss, {
            "relational_loss": float(loss.item()),
            "mse_loss": float(mse_loss.item()),
            "cosine_alignment": float(cos_sim.item()),
        }

    def train_steps(
        self,
        batches: list[dict[str, torch.Tensor | None]],
        save_path: str | Path | None = None,
    ) -> list[dict[str, float]]:
        """Execute Future-Label-Free pretraining steps and optionally save pure backbone.

        Future-Label-Free 관계형 사전학습을 실행하고 순수 백본만 선택적으로 저장합니다.
        """
        self.backbone.train()
        history: list[dict[str, float]] = []
        for step_idx, batch in enumerate(batches):
            self.optimizer.zero_grad()
            loss, metrics = self.compute_relational_loss(
                ohlcv_seq=batch["ohlcv_seq"],  # type: ignore[arg-type]
                z_mm=batch.get("z_mm"),
                is_blackout=batch.get("is_blackout"),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=1.0)
            self.optimizer.step()
            metrics["step"] = float(step_idx)
            history.append(metrics)
            logger.debug("Pretrain step={} metrics={}", step_idx, metrics)

        if save_path is not None:
            self.backbone.save_backbone(save_path)
        return history
