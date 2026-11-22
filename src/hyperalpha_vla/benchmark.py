"""Out-of-sample comparator: OHLCV blackout mode vs. Model A (z_mm) + Model B combined.

OHLCV 단독(Blackout) 모드 대 Model A(z_mm)+Model B 결합 Out-of-Sample 벤치마크 비교기입니다.
"""

from dataclasses import dataclass

import numpy as np
import torch

from hyperalpha_vla.model import HyperAlphaVLA


@dataclass
class OOSMetrics:
    """Out-of-sample portfolio performance metrics container.

    Out-of-Sample 포트폴리오 성과 지표 컨테이너입니다.
    """

    total_return_pct: float
    annualized_sharpe: float
    annualized_sortino: float
    max_drawdown_pct: float
    cvar_15_pct: float


@dataclass
class OOSBenchmarkReport:
    """Benchmark comparison report between OHLCV-only and combined Model A+B modes.

    OHLCV 단독 모드 대비 Model A+B 결합 모드 벤치마크 비교 리포트입니다.
    """

    ohlcv_only: OOSMetrics
    multimodal_combined: OOSMetrics
    sharpe_delta: float
    mdd_improvement_pct: float


def compute_oos_metrics(daily_returns: np.ndarray, cvar_alpha: float = 0.15) -> OOSMetrics:
    """Compute OOS return, Sharpe, Sortino, MDD, and tail CVaR from daily returns.

    일별 순수익률 벡터에서 OOS 수익률·Sharpe·Sortino·MDD·CVaR 지표를 계산합니다.
    """
    rets = np.asarray(daily_returns, dtype=np.float64)
    if rets.size == 0:
        return OOSMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    wealth = np.cumprod(1.0 + rets)
    total_ret = (wealth[-1] - 1.0) * 100.0
    peaks = np.maximum.accumulate(np.maximum(wealth, 1e-8))
    drawdowns = (wealth - peaks) / peaks
    mdd_pct = float(np.min(drawdowns) * 100.0)

    mean_r = float(np.mean(rets))
    std_r = max(float(np.std(rets)), 1e-8)
    downside = np.minimum(rets, 0.0)
    down_std = max(float(np.sqrt(np.mean(downside**2))), 1e-8)

    sharpe = (mean_r / std_r) * np.sqrt(250.0)
    sortino = (mean_r / down_std) * np.sqrt(250.0)

    tail_k = max(1, int(np.ceil(cvar_alpha * rets.size)))
    worst_losses = np.sort(rets)[:tail_k]
    cvar_pct = float(np.mean(worst_losses) * 100.0)

    return OOSMetrics(
        total_return_pct=round(float(total_ret), 4),
        annualized_sharpe=round(float(sharpe), 4),
        annualized_sortino=round(float(sortino), 4),
        max_drawdown_pct=round(float(mdd_pct), 4),
        cvar_15_pct=round(float(cvar_pct), 4),
    )


@torch.no_grad()
def run_oos_comparison(
    model: HyperAlphaVLA,
    ohlcv_seq: torch.Tensor,
    z_mm: torch.Tensor,
    future_day1_returns: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> OOSBenchmarkReport:
    """Compare OHLCV blackout vs. multimodal z_mm performance on identical policy weights.

    동일 백본·정책 모델에 대해 OHLCV 단독(is_blackout=True) 대 멀티모달(z_mm) OOS를 비교합니다.
    """
    model.eval()
    bsz, n_assets = ohlcv_seq.shape[:2]
    all_blackout = torch.ones((bsz, n_assets), dtype=torch.bool, device=ohlcv_seq.device)
    no_blackout = torch.zeros((bsz, n_assets), dtype=torch.bool, device=ohlcv_seq.device)

    out_blackout = model(
        ohlcv_seq=ohlcv_seq,
        z_mm=None,
        is_blackout=all_blackout,
        valid_mask=valid_mask,
    )
    out_mm = model(
        ohlcv_seq=ohlcv_seq,
        z_mm=z_mm,
        is_blackout=no_blackout,
        valid_mask=valid_mask,
    )

    ret_np = future_day1_returns.detach().cpu().numpy()
    w_blk = out_blackout.weights_long.detach().cpu().numpy()
    w_mm = out_mm.weights_long.detach().cpu().numpy()

    pnl_blk = np.sum(w_blk * ret_np, axis=-1)
    pnl_mm = np.sum(w_mm * ret_np, axis=-1)

    m_blk = compute_oos_metrics(pnl_blk)
    m_mm = compute_oos_metrics(pnl_mm)
    return OOSBenchmarkReport(
        ohlcv_only=m_blk,
        multimodal_combined=m_mm,
        sharpe_delta=round(m_mm.annualized_sharpe - m_blk.annualized_sharpe, 4),
        mdd_improvement_pct=round(m_mm.max_drawdown_pct - m_blk.max_drawdown_pct, 4),
    )
