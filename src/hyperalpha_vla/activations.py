"""Sparse probability simplex projections (1.5-Entmax and Sparsemax).

희소 확률 심플렉스 투영 활성화 함수(1.5-Entmax 및 Sparsemax) 모듈입니다.

References:
    - Martins & Astudillo (ICML 2016): "From Softmax to Sparsemax"
    - Peters, Niculae & Martins (ACL 2019): "Sparse Sequence-to-Sequence Models"
"""

import torch
from torch import nn


class Entmax15Function(torch.autograd.Function):
    """1.5-Entmax (Tsallis alpha=1.5) exact Euclidean simplex projection.

    Tsallis alpha=1.5 기반의 정확한 1.5-Entmax 유클리드 심플렉스 투영 연산자입니다.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Compute forward 1.5-Entmax probability simplex projection.

        순전파 1.5-Entmax 확률 심플렉스 투영을 계산합니다.
        """
        ctx.dim = dim
        input_dtype = x.dtype
        x_fp32 = torch.nan_to_num(x.float(), nan=-1e4, neginf=-1e4)
        x_max = x_fp32.max(dim=dim, keepdim=True).values
        x_shifted = torch.clamp(x_fp32 - x_max, min=-1e4)

        x_sorted, _ = torch.sort(x_shifted, dim=dim, descending=True)
        k = torch.arange(1, x_fp32.shape[dim] + 1, device=x.device, dtype=torch.float32)
        shape = [1] * x.ndim
        shape[dim] = -1
        k = k.view(shape)

        cumsum_x = torch.cumsum(x_sorted, dim=dim)
        cumsum_x2 = torch.cumsum(x_sorted**2, dim=dim)

        mean = cumsum_x / k
        variance = cumsum_x2 / k - mean**2
        delta = 4.0 / k - variance
        valid_tau = delta >= 0
        tau_cand = mean - torch.sqrt(torch.clamp(delta, min=0.0))

        is_active = (tau_cand < x_sorted) & valid_tau
        k_star = torch.clamp(is_active.sum(dim=dim, keepdim=True), min=1)
        tau = tau_cand.gather(dim, k_star - 1)

        p = torch.clamp((x_shifted - tau) / 2.0, min=0.0) ** 2
        ctx.save_for_backward(p)
        return p.to(dtype=input_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute exact gradients of 1.5-Entmax simplex projection.

        1.5-Entmax 심플렉스 투영의 해석적 역전파 기울기를 계산합니다.
        """
        (p,) = ctx.saved_tensors
        dim = ctx.dim
        grad_dtype = grad_output.dtype
        p_fp32 = p.float()
        grad_fp32 = grad_output.float()

        sqrt_p = torch.sqrt(torch.clamp(p_fp32, min=0.0))
        v = sqrt_p * grad_fp32
        sum_v = v.sum(dim=dim, keepdim=True)
        sum_sqrt_p = torch.clamp(sqrt_p.sum(dim=dim, keepdim=True), min=1e-12)

        grad_input = v - sqrt_p * (sum_v / sum_sqrt_p)
        return grad_input.to(dtype=grad_dtype), None


class SparsemaxFunction(torch.autograd.Function):
    """Sparsemax (alpha=2.0) exact Euclidean simplex projection.

    Sparsemax (alpha=2.0) 기반의 정확한 유클리드 심플렉스 투영 연산자입니다.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Compute forward Sparsemax probability simplex projection.

        순전파 Sparsemax 확률 심플렉스 투영을 계산합니다.
        """
        ctx.dim = dim
        input_dtype = x.dtype
        x_fp32 = torch.nan_to_num(x.float(), nan=-1e4, neginf=-1e4)
        x_max = x_fp32.max(dim=dim, keepdim=True).values
        x_shifted = torch.clamp(x_fp32 - x_max, min=-1e4)

        x_sorted, _ = torch.sort(x_shifted, dim=dim, descending=True)
        k = torch.arange(1, x_fp32.shape[dim] + 1, device=x.device, dtype=torch.float32)
        shape = [1] * x.ndim
        shape[dim] = -1
        k = k.view(shape)

        cumsum_x = torch.cumsum(x_sorted, dim=dim)
        is_active = (1.0 + k * x_sorted) > cumsum_x
        k_star = torch.clamp(is_active.sum(dim=dim, keepdim=True), min=1)

        cumsum_k_star = cumsum_x.gather(dim, k_star - 1)
        tau = (cumsum_k_star - 1.0) / k_star.to(dtype=torch.float32)

        p = torch.clamp(x_shifted - tau, min=0.0)
        ctx.save_for_backward(p)
        return p.to(dtype=input_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute exact support-restricted gradients of Sparsemax.

        Sparsemax 활성화 지원 집합에 대한 역전파 기울기를 계산합니다.
        """
        (p,) = ctx.saved_tensors
        dim = ctx.dim
        grad_dtype = grad_output.dtype
        p_fp32 = p.float()
        grad_fp32 = grad_output.float()

        support = (p_fp32 > 0).to(dtype=torch.float32)
        v = grad_fp32 * support
        support_count = torch.clamp(support.sum(dim=dim, keepdim=True), min=1.0)
        grad_input = v - support * (v.sum(dim=dim, keepdim=True) / support_count)
        return grad_input.to(dtype=grad_dtype), None


def entmax15(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute 1.5-Entmax probability distribution along the target dimension.

    지정된 차원에 대해 1.5-Entmax 희소 확률 분포를 계산합니다.
    """
    return Entmax15Function.apply(x, dim)


def sparsemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute Sparsemax probability distribution along the target dimension.

    지정된 차원에 대해 Sparsemax 희소 확률 분포를 계산합니다.
    """
    return SparsemaxFunction.apply(x, dim)


class Entmax15(nn.Module):
    """PyTorch module wrapper for 1.5-Entmax activation.

    1.5-Entmax 활성화 레이어 모듈입니다.
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 1.5-Entmax activation along self.dim.

        설정된 차원(self.dim)으로 1.5-Entmax 활성화를 적용합니다.
        """
        return entmax15(x, dim=self.dim)


class Sparsemax(nn.Module):
    """PyTorch module wrapper for Sparsemax activation.

    Sparsemax 활성화 레이어 모듈입니다.
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Sparsemax activation along self.dim.

        설정된 차원(self.dim)으로 Sparsemax 활성화를 적용합니다.
        """
        return sparsemax(x, dim=self.dim)
