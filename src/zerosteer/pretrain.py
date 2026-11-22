"""Future-Label-Free cross-asset masked relational pretraining and SIGReg regularization.

Future-Label-Free Cross-Asset 마스킹 관계형 백본 사전학습 및 SIGReg 정규화 모듈입니다.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

from zerosteer.model import ZeroSteerBackbone


def compute_sigreg_loss(
    z: torch.Tensor,
    *,
    num_slices: int = 64,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """LeJEPA Sketched Isotropic Gaussian Regularization (SIGReg) without complex FFTs.

    Projects (N, D) embeddings onto random unit directions and integrates weighted MSE against
    the standard normal characteristic function phi(t) = exp(-t^2 / 2) via trapezoidal rule.
    Uses real Euler cos/sin so execution is uniform across CPU, CUDA, and Apple Silicon MPS.

    복소수 FFT 없이 무작위 단위 방향 투영과 사다리꼴 적분을 통해 임베딩 분포를
    등방성 표준 정규 분포로 정규화하는 LeJEPA SIGReg 손실 함수입니다(MPS/CUDA/CPU 호환).
    """
    if z.ndim != 2 or z.shape[0] < 4 or z.shape[1] == 0:
        return z.sum() * 0.0

    dim = z.shape[1]
    raw_dir = (
        torch.randn((dim, num_slices), dtype=torch.float32, device="cpu", generator=generator).to(
            device=z.device, dtype=z.dtype
        )
        if generator is not None
        else torch.randn((dim, num_slices), dtype=z.dtype, device=z.device)
    )
    proj_dirs = F.normalize(raw_dir, p=2, dim=0)
    t = torch.linspace(-5.0, 5.0, 17, dtype=z.dtype, device=z.device)
    gcf = torch.exp(-0.5 * t.square())
    args = (z @ proj_dirs).unsqueeze(-1) * t.view(1, 1, -1)
    ecf_re = torch.cos(args).mean(dim=0)
    ecf_im = torch.sin(args).mean(dim=0)
    diff_sq = (ecf_re - gcf.unsqueeze(0)).square() + ecf_im.square()
    weighted_err = diff_sq * gcf.unsqueeze(0)
    return torch.trapezoid(weighted_err, t, dim=-1).mean()


class CrossAssetMaskedRelationalPretrainer:
    """Reconstructs masked intra-timestamp asset tokens via cross-asset attention + SIGReg.

    Zero advisory labels: uses 0% future return, ranking, volatility, or crash labels.

    동일 시점(t) 내 마스킹된 자산 토큰을 나머지 종목/지수 컨텍스트로 복원하는 사전학습기입니다.
    미래 수익률, 랭킹, 변동성/폭락 라벨을 0% 사용합니다(Future-Label-Free).
    """

    def __init__(
        self,
        backbone: ZeroSteerBackbone,
        mask_ratio: float = 0.25,
        lr: float = 1e-3,
        *,
        sigreg_weight: float | None = None,
        sigreg_num_slices: int | None = None,
    ) -> None:
        self.backbone = backbone
        self.mask_ratio = mask_ratio
        self.sigreg_weight = float(
            backbone.config.sigreg_weight if sigreg_weight is None else sigreg_weight
        )
        self.sigreg_num_slices = int(
            backbone.config.sigreg_num_slices if sigreg_num_slices is None else sigreg_num_slices
        )

        d_model = backbone.config.d_model
        backbone_device = next(backbone.parameters()).device
        self.rel_mask_token = nn.Parameter(
            torch.zeros(1, 1, d_model, device=backbone_device)
        )
        nn.init.normal_(self.rel_mask_token, mean=0.0, std=0.02)
        self.recon_head = nn.Linear(d_model, d_model).to(device=backbone_device)
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
        calendar_feats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Mask random same-timestamp observation tokens and compute MSE + cosine + SIGReg loss.

        동일 시점 관측 토큰 일부를 마스킹하고 교차 어텐션 복원 손실 및 SIGReg 정규화 항을 계산합니다.
        """
        obs_embs = self.backbone.encode_observations(
            ohlcv_seq=ohlcv_seq,
            z_mm=z_mm,
            is_blackout=is_blackout,
            calendar_feats=calendar_feats,
        )
        bsz, n_assets, d_model = obs_embs.shape
        mask = torch.rand((bsz, n_assets), device=obs_embs.device) < self.mask_ratio
        # Ensure at least one asset is masked per sample for stable gradient signal.
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
        sigreg_loss = compute_sigreg_loss(
            encoded.reshape(-1, d_model),
            num_slices=self.sigreg_num_slices,
        )
        loss = mse_loss + 0.1 * (1.0 - cos_sim) + self.sigreg_weight * sigreg_loss
        return loss, {
            "relational_loss": float(loss.item()),
            "mse_loss": float(mse_loss.item()),
            "cosine_alignment": float(cos_sim.item()),
            "sigreg_loss": float(sigreg_loss.item()),
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
        trainable_params = [
            *self.backbone.parameters(),
            self.rel_mask_token,
            *self.recon_head.parameters(),
        ]
        for step_idx, batch in enumerate(batches):
            self.optimizer.zero_grad()
            loss, metrics = self.compute_relational_loss(
                ohlcv_seq=batch["ohlcv_seq"],  # type: ignore[arg-type]
                z_mm=batch.get("z_mm"),
                is_blackout=batch.get("is_blackout"),
                calendar_feats=batch.get("calendar_feats"),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            self.optimizer.step()
            self.backbone.update_gate_bias()
            metrics["step"] = float(step_idx)
            history.append(metrics)
            logger.debug("Pretrain step={} metrics={}", step_idx, metrics)

        if save_path is not None:
            self.backbone.save_backbone(save_path)
        return history
