"""Representation collapse, isotropy, and MoE routing diagnostics for ZeroSteer.

ZeroSteer 백본의 표현 붕괴, 등방성 및 MoE 라우팅 균형 진단 모듈입니다.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from zerosteer.model import ZeroSteerBackbone
from zerosteer.pretrain import compute_sigreg_loss


@dataclass(frozen=True)
class ZeroSteerDiagnosticReport:
    """Summary metrics for ZeroSteer representation health and expert balance.

    ZeroSteer 표현 건전성 및 전문가 라우팅 균형 진단 리포트입니다.
    """

    effective_rank: float
    normalized_rank: float
    mean_cosine_similarity: float
    sigreg_loss: float
    embedding_std: float
    num_tokens: int
    moe_load_stats: dict[str, dict[str, list[float] | list[int]]]


def compute_effective_rank(embeddings: torch.Tensor, eps: float = 1e-12) -> tuple[float, float]:
    """Compute SVD Shannon effective rank and normalized isotropy ratio on CPU float64.

    MPS/CUDA 정밀도 제약을 회피하기 위해 CPU float64 SVD로 유효 랭크와 정규화 랭크 비율을 계산합니다.
    Returns `(effective_rank, normalized_rank)` where `normalized_rank = effective_rank / min(N, D)`.
    """
    if embeddings.ndim < 2 or embeddings.shape[-1] == 0:
        raise ValueError("embeddings must have at least 2 dimensions and non-zero feature size.")

    flat = embeddings.reshape(-1, embeddings.shape[-1])
    n_tokens, dim = flat.shape
    max_rank = min(n_tokens, dim)
    if max_rank < 2:
        return 1.0, 1.0

    # Move to CPU before casting to float64 because Apple Silicon MPS has no Double type.
    centered = flat.detach().cpu().to(dtype=torch.float64)
    centered = centered - centered.mean(dim=0, keepdim=True)
    singular_vals = torch.linalg.svdvals(centered)
    total = singular_vals.sum()
    if total <= eps:
        return 1.0, 1.0 / float(max_rank)

    probs = (singular_vals / total).clamp(min=eps)
    entropy = -(probs * torch.log(probs)).sum()
    effective_rank = float(torch.exp(entropy).item())
    normalized_rank = min(max(effective_rank / float(max_rank), 0.0), 1.0)
    return effective_rank, normalized_rank


@torch.no_grad()
def evaluate_backbone_representations(
    backbone: ZeroSteerBackbone,
    ohlcv_seq: torch.Tensor,
    z_mm: torch.Tensor | None = None,
    is_blackout: torch.Tensor | None = None,
    calendar_feats: torch.Tensor | None = None,
    *,
    num_slices: int = 64,
    seed: int | None = 42,
) -> ZeroSteerDiagnosticReport:
    """Evaluate backbone observation embeddings for rank collapse, cosine collapse, and SIGReg.

    백본 관측 임베딩의 랭크 붕괴, 코사인 붕괴 및 SIGReg 등방성 점수를 비학습 모드로 진단합니다.
    """
    was_training = backbone.training
    backbone.eval()
    try:
        final_obs, _ = backbone.forward_backbone(
            ohlcv_seq=ohlcv_seq,
            z_mm=z_mm,
            is_blackout=is_blackout,
            calendar_feats=calendar_feats,
        )
    finally:
        backbone.train(was_training)

    flat = final_obs.reshape(-1, final_obs.shape[-1]).detach().float()
    eff_rank, norm_rank = compute_effective_rank(flat)

    n_tokens = flat.shape[0]
    if n_tokens >= 2:
        normed = F.normalize(flat, p=2, dim=-1)
        sim_matrix = normed @ normed.T
        off_diag_mask = ~torch.eye(n_tokens, dtype=torch.bool, device=sim_matrix.device)
        mean_cos = float(sim_matrix[off_diag_mask].mean().item())
    else:
        mean_cos = 1.0

    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    sigreg = float(compute_sigreg_loss(flat, num_slices=num_slices, generator=generator).item())
    emb_std = float(flat.std(unbiased=False).item())

    return ZeroSteerDiagnosticReport(
        effective_rank=eff_rank,
        normalized_rank=norm_rank,
        mean_cosine_similarity=mean_cos,
        sigreg_loss=sigreg,
        embedding_std=emb_std,
        num_tokens=int(n_tokens),
        moe_load_stats=backbone.get_moe_load_stats(),
    )
