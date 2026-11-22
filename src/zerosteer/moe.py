"""Cross-Asset relational MoE Transformer backbone (Dense + Shared/Routed Experts).

Cross-Asset 관계형 MoE 트랜스포머 백본 모듈(Dense + Shared/Routed Experts)입니다.
"""

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization layer.

    Root Mean Square 레이어 정규화(RMSNorm) 모듈입니다.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization along the last dimension.

        마지막 차원에 대해 RMS 정규화를 적용합니다.
        """
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class GeGLUFeedForward(nn.Module):
    """GeGLU-gated feed-forward network block.

    GeGLU 게이트 기반 피드포워드 네트워크 블록입니다.
    """

    def __init__(self, d_model: int, inter_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, inter_dim * 2)
        self.out = nn.Linear(inter_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform hidden states through GeGLU gating and output projection.

        GeGLU 게이팅 및 출력 투영을 통해 은닉 표현을 변환합니다.
        """
        val, gate = self.proj(x).chunk(2, dim=-1)
        hidden = val * F.gelu(gate)
        return self.out(self.drop(hidden))


class TopKMoELayer(nn.Module):
    """Mixture-of-Experts layer combining a shared expert with Top-K routed experts.

    공유 전문가(Shared Expert)와 Top-K 라우팅 전문가가 결합된 MoE 레이어입니다.
    """

    def __init__(
        self,
        d_model: int,
        inter_dim: int,
        n_routed_experts: int = 16,
        n_activated_experts: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = min(n_activated_experts, n_routed_experts)
        self.gate = nn.Linear(d_model, n_routed_experts, bias=False)
        self.shared_expert = GeGLUFeedForward(d_model, inter_dim, dropout=dropout)
        self.experts = nn.ModuleList(
            [GeGLUFeedForward(d_model, inter_dim, dropout=dropout) for _ in range(n_routed_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route tokens across top-k experts and combine with the shared expert output.

        토큰을 Top-K 전문가에 분배한 뒤 공유 전문가 출력과 합산합니다.
        """
        bsz, seq_len, dim = x.shape
        flat_x = x.reshape(-1, dim)

        logits = self.gate(flat_x)
        topk_vals, topk_idx = torch.topk(logits, k=self.n_activated_experts, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)

        routed_out = torch.zeros_like(flat_x)
        for expert_idx, expert in enumerate(self.experts):
            mask = topk_idx == expert_idx
            if not mask.any():
                continue
            token_indices, slot_indices = mask.nonzero(as_tuple=True)
            selected_x = flat_x[token_indices]
            expert_out = expert(selected_x)
            weights = topk_weights[token_indices, slot_indices].unsqueeze(-1)
            routed_out.index_add_(0, token_indices, expert_out * weights)

        shared_out = self.shared_expert(flat_x)
        return (shared_out + routed_out).view(bsz, seq_len, dim)


class MoETransformerBlock(nn.Module):
    """Hybrid Dense/MoE transformer block supporting asymmetric attention masks.

    비대칭 어텐션 마스크를 지원하는 Dense/MoE 하이브리드 트랜스포머 블록입니다.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        inter_dim: int,
        use_moe: bool,
        n_routed_experts: int = 16,
        n_activated_experts: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn: nn.Module = (
            TopKMoELayer(
                d_model=d_model,
                inter_dim=inter_dim,
                n_routed_experts=n_routed_experts,
                n_activated_experts=n_activated_experts,
                dropout=dropout,
            )
            if use_moe
            else GeGLUFeedForward(d_model=d_model, inter_dim=inter_dim, dropout=dropout)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply pre-norm self-attention with optional attn_mask and FFN residual block.

        선택적 어텐션 마스크와 잔차 FFN(Dense 또는 MoE) 블록을 적용합니다.
        """
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class MoETransformerBackbone(nn.Module):
    """Cross-asset relational backbone stacking initial Dense layers and MoE layers.

    초기 Dense 레이어와 후속 MoE 레이어를 적층한 Cross-Asset 관계형 백본입니다.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        nlayers: int = 6,
        n_dense_layers: int = 3,
        n_routed_experts: int = 16,
        n_activated_experts: int = 4,
        inter_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MoETransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    inter_dim=inter_dim,
                    use_moe=(layer_idx >= n_dense_layers),
                    n_routed_experts=n_routed_experts,
                    n_activated_experts=n_activated_experts,
                    dropout=dropout,
                )
                for layer_idx in range(nlayers)
            ]
        )
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run stacked Dense/MoE blocks with final RMSNorm over the token sequence.

        적층된 Dense/MoE 블록과 최종 RMSNorm을 입력 토큰 시퀀스에 적용합니다.
        """
        hidden = x
        for layer in self.layers:
            hidden = layer(hidden, attn_mask=attn_mask)
        return self.norm(hidden)
