"""Cross-Asset 관계형 MoE Transformer 백본 모듈 (Dense + Shared/Routed Experts)."""

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class GeGLUFeedForward(nn.Module):
    """GeGLU 게이트 기반 피드포워드 네트워크."""

    def __init__(self, d_model: int, inter_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, inter_dim * 2)
        self.out = nn.Linear(inter_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val, gate = self.proj(x).chunk(2, dim=-1)
        hidden = val * F.gelu(gate)
        return self.out(self.drop(hidden))


class TopKMoELayer(nn.Module):
    """Shared Expert와 Top-K Routed Experts가 결합된 Mixture-of-Experts 레이어."""

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
    """비대칭 어텐션 마스크를 지원하는 Dense/MoE 하이브리드 트랜스포머 블록."""

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
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class MoETransformerBackbone(nn.Module):
    """초기 Dense 레이어와 후속 MoE 레이어를 적층한 Cross-Asset 관계형 백본."""

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
        hidden = x
        for layer in self.layers:
            hidden = layer(hidden, attn_mask=attn_mask)
        return self.norm(hidden)
