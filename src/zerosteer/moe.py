"""Cross-Asset relational MoE Transformer backbone (Dense + Shared/Routed Experts).

Cross-Asset 관계형 MoE 트랜스포머 백본 모듈(Dense + Shared/Routed Experts)입니다.
"""

from typing import Literal

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

    def __init__(
        self,
        d_model: int,
        inter_dim: int,
        dropout: float = 0.1,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        output_dim = d_model if out_dim is None else out_dim
        self.proj = nn.Linear(d_model, inter_dim * 2)
        self.out = nn.Linear(inter_dim, output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform hidden states through GeGLU gating and output projection.

        GeGLU 게이팅 및 출력 투영을 통해 은닉 표현을 변환합니다.
        """
        val, gate = self.proj(x).chunk(2, dim=-1)
        hidden = val * F.gelu(gate)
        return self.out(self.drop(hidden))


class SwiGLUFeedForward(nn.Module):
    """SwiGLU-gated feed-forward network block without bias terms.

    바이어스 없는 SwiGLU 게이트 기반 피드포워드 네트워크 블록입니다.
    """

    def __init__(
        self,
        d_model: int,
        inter_dim: int,
        dropout: float = 0.1,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        output_dim = d_model if out_dim is None else out_dim
        self.w1 = nn.Linear(d_model, inter_dim, bias=False)
        self.w3 = nn.Linear(d_model, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, output_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform hidden states through SiLU-gated linear units.

        SiLU 게이트 선형 유닛(SwiGLU)을 통해 은닉 표현을 변환합니다.
        """
        hidden = F.silu(self.w1(x)) * self.w3(x)
        return self.w2(self.drop(hidden))


def _make_ffn(
    d_model: int,
    inter_dim: int,
    activation: Literal["swiglu", "geglu"],
    dropout: float,
    out_dim: int | None = None,
) -> nn.Module:
    """Instantiate SwiGLU or GeGLU feed-forward network uniformly.

    SwiGLU 또는 GeGLU 피드포워드 네트워크를 일관되게 생성합니다.
    """
    if activation == "swiglu":
        return SwiGLUFeedForward(d_model=d_model, inter_dim=inter_dim, dropout=dropout, out_dim=out_dim)
    return GeGLUFeedForward(d_model=d_model, inter_dim=inter_dim, dropout=dropout, out_dim=out_dim)


class QKNormMultiheadAttention(nn.Module):
    """Multihead self-attention with per-head QK RMSNorm and SDPA mask polarity handling.

    헤드별 Q/K RMSNorm과 SDPA 마스크 극성 변환을 지원하는 멀티헤드 셀프 어텐션입니다.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        self.use_qk_norm = use_qk_norm

        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm: nn.Module = RMSNorm(self.head_dim) if use_qk_norm else nn.Identity()
        self.k_norm: nn.Module = RMSNorm(self.head_dim) if use_qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply Q/K-normalized scaled dot-product self-attention.

        헤드별 Q/K 정규화가 적용된 스케일드 닷프로덕트 셀프 어텐션을 수행합니다.
        Note: `attn_mask` with bool dtype uses nn.MultiheadAttention convention (True = blocked)
        and is inverted to SDPA convention (True = attend).
        """
        bsz, seq_len, _ = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = self.q_norm(q.view(bsz, seq_len, self.nhead, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(bsz, seq_len, self.nhead, self.head_dim)).transpose(1, 2)
        v = v.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        sdpa_mask: torch.Tensor | None = None
        if attn_mask is not None:
            # nn.MultiheadAttention boolean masks mark True=blocked; PyTorch SDPA marks True=attend.
            sdpa_mask = ~attn_mask if attn_mask.dtype == torch.bool else attn_mask
            if sdpa_mask.ndim == 3 and sdpa_mask.shape[0] == bsz:
                sdpa_mask = sdpa_mask.unsqueeze(1)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(attn_out)


class AuxLossFreeMoEGate(nn.Module):
    """DeepSeek-V3 style auxiliary-loss-free top-k MoE router with adaptive bias balancing.

    보조 손실 없이 적응형 바이어스 조절로 부하를 균형화하는 DeepSeek-V3 방식 Top-K MoE 게이트입니다.
    """

    bias: torch.Tensor
    expert_loads: torch.Tensor
    last_load_counts: torch.Tensor

    def __init__(
        self,
        d_model: int,
        n_routed_experts: int,
        n_activated_experts: int,
        *,
        score_func: Literal["sigmoid", "softmax"] = "sigmoid",
        bias_update_speed: float = 0.01,
        route_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = min(n_activated_experts, n_routed_experts)
        self.score_func = score_func
        self.bias_update_speed = bias_update_speed
        self.route_scale = route_scale

        self.gate_network = nn.Linear(d_model, n_routed_experts, bias=False)
        nn.init.normal_(self.gate_network.weight, mean=0.0, std=0.02)

        self.register_buffer("bias", torch.zeros(n_routed_experts))
        self.register_buffer("expert_loads", torch.zeros(n_routed_experts))
        self.register_buffer("last_load_counts", torch.zeros(n_routed_experts, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute normalized top-k gating weights and selected expert indices.

        정규화된 Top-K 게이팅 가중치와 선택된 전문가 인덱스를 계산합니다.
        """
        logits = self.gate_network(x)
        if self.score_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            scores = logits.softmax(dim=-1, dtype=torch.float32).to(logits.dtype)

        selection_scores = scores + self.bias.to(dtype=scores.dtype)
        _, topk_idx = torch.topk(selection_scores, k=self.n_activated_experts, dim=-1)

        weights = scores.gather(1, topk_idx)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        weights = weights * self.route_scale

        if self.training:
            with torch.no_grad():
                counts = torch.bincount(topk_idx.reshape(-1), minlength=self.n_routed_experts)
                self.expert_loads.add_(counts.to(dtype=self.expert_loads.dtype))
                self.last_load_counts.copy_(counts)

        return weights, topk_idx

    @torch.no_grad()
    def update_bias(self) -> None:
        """Adjust routing bias outside autograd to equalize expert utilization.

        자동 미분 그래프 외부에서 라우팅 바이어스를 조정하여 전문가 활용도를 균등화합니다.
        """
        if self.bias_update_speed <= 0.0:
            self.expert_loads.zero_()
            return
        total_load = self.expert_loads.sum()
        if total_load <= 0.0:
            return
        avg_load = total_load / float(self.n_routed_experts)
        diff = self.expert_loads - avg_load
        self.bias.sub_(self.bias_update_speed * torch.sign(diff))
        self.bias.clamp_(-2.0, 2.0)
        self.expert_loads.zero_()


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
        *,
        out_dim: int | None = None,
        ffn_activation: Literal["swiglu", "geglu"] = "swiglu",
        use_aux_loss_free: bool = True,
        score_func: Literal["sigmoid", "softmax"] = "sigmoid",
        bias_update_speed: float = 0.01,
        route_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.out_dim = d_model if out_dim is None else out_dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = min(n_activated_experts, n_routed_experts)
        self.use_aux_loss_free = use_aux_loss_free

        if use_aux_loss_free:
            self.gate: nn.Module = AuxLossFreeMoEGate(
                d_model=d_model,
                n_routed_experts=n_routed_experts,
                n_activated_experts=self.n_activated_experts,
                score_func=score_func,
                bias_update_speed=bias_update_speed,
                route_scale=route_scale,
            )
        else:
            self.gate = nn.Linear(d_model, n_routed_experts, bias=False)

        self.shared_expert = _make_ffn(
            d_model=d_model,
            inter_dim=inter_dim,
            activation=ffn_activation,
            dropout=dropout,
            out_dim=self.out_dim,
        )
        self.experts = nn.ModuleList(
            [
                _make_ffn(
                    d_model=d_model,
                    inter_dim=inter_dim,
                    activation=ffn_activation,
                    dropout=dropout,
                    out_dim=self.out_dim,
                )
                for _ in range(n_routed_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route tokens across top-k experts and combine with the shared expert output.

        토큰을 Top-K 전문가에 분배한 뒤 공유 전문가 출력과 합산합니다.
        """
        orig_shape = x.shape[:-1]
        dim = x.shape[-1]
        flat_x = x.reshape(-1, dim)

        if isinstance(self.gate, AuxLossFreeMoEGate):
            topk_weights, topk_idx = self.gate(flat_x)
        else:
            logits = self.gate(flat_x)
            topk_vals, topk_idx = torch.topk(logits, k=self.n_activated_experts, dim=-1)
            topk_weights = F.softmax(topk_vals, dim=-1)

        routed_out = flat_x.new_zeros((flat_x.shape[0], self.out_dim))
        for expert_idx, expert in enumerate(self.experts):
            mask = topk_idx == expert_idx
            if not mask.any():
                continue
            token_indices, slot_indices = mask.nonzero(as_tuple=True)
            selected_x = flat_x[token_indices]
            expert_out = expert(selected_x)
            weights = topk_weights[token_indices, slot_indices].unsqueeze(-1)
            # Out-of-place index_add preserves autograd and avoids Apple Silicon MPS aliasing issues.
            routed_out = routed_out.index_add(
                0,
                token_indices,
                (expert_out * weights).to(dtype=routed_out.dtype),
            )

        shared_out = self.shared_expert(flat_x)
        return (shared_out + routed_out).reshape(*orig_shape, self.out_dim)

    def update_gate_bias(self) -> None:
        """Update aux-loss-free router bias if enabled.

        Aux-Loss-Free 라우터가 활성화된 경우 바이어스를 갱신합니다.
        """
        if isinstance(self.gate, AuxLossFreeMoEGate):
            self.gate.update_bias()

    def get_load_stats(self) -> dict[str, list[float] | list[int]]:
        """Return CPU-safe expert load counts and bias vectors for monitoring.

        모니터링용 CPU 기반 전문가 부하 카운트와 바이어스 벡터를 반환합니다.
        """
        if not isinstance(self.gate, AuxLossFreeMoEGate):
            return {}
        return {
            "last_load_counts": [
                int(v) for v in self.gate.last_load_counts.to("cpu", dtype=torch.long).tolist()
            ],
            "gate_bias": [float(v) for v in self.gate.bias.to("cpu", dtype=torch.float32).tolist()],
        }


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
        *,
        use_qk_norm: bool = True,
        ffn_activation: Literal["swiglu", "geglu"] = "swiglu",
        use_aux_loss_free: bool = True,
        moe_score_func: Literal["sigmoid", "softmax"] = "sigmoid",
        moe_bias_update_speed: float = 0.01,
        moe_route_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = QKNormMultiheadAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn: nn.Module = (
            TopKMoELayer(
                d_model=d_model,
                inter_dim=inter_dim,
                n_routed_experts=n_routed_experts,
                n_activated_experts=n_activated_experts,
                dropout=dropout,
                ffn_activation=ffn_activation,
                use_aux_loss_free=use_aux_loss_free,
                score_func=moe_score_func,
                bias_update_speed=moe_bias_update_speed,
                route_scale=moe_route_scale,
            )
            if use_moe
            else _make_ffn(
                d_model=d_model,
                inter_dim=inter_dim,
                activation=ffn_activation,
                dropout=dropout,
            )
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply pre-norm self-attention with optional attn_mask and FFN residual block.

        선택적 어텐션 마스크와 잔차 FFN(Dense 또는 MoE) 블록을 적용합니다.
        """
        normed = self.norm1(x)
        attn_out = self.attn(normed, attn_mask=attn_mask)
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
        *,
        use_qk_norm: bool = True,
        ffn_activation: Literal["swiglu", "geglu"] = "swiglu",
        use_aux_loss_free: bool = True,
        moe_score_func: Literal["sigmoid", "softmax"] = "sigmoid",
        moe_bias_update_speed: float = 0.01,
        moe_route_scale: float = 1.0,
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
                    use_qk_norm=use_qk_norm,
                    ffn_activation=ffn_activation,
                    use_aux_loss_free=use_aux_loss_free,
                    moe_score_func=moe_score_func,
                    moe_bias_update_speed=moe_bias_update_speed,
                    moe_route_scale=moe_route_scale,
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

    def update_gate_bias(self) -> None:
        """Update aux-loss-free gate biases across all MoE transformer layers.

        모든 MoE 트랜스포머 레이어의 Aux-Loss-Free 게이트 바이어스를 갱신합니다.
        """
        for layer in self.layers:
            if isinstance(layer.ffn, TopKMoELayer):
                layer.ffn.update_gate_bias()

    def get_moe_load_stats(self) -> dict[str, dict[str, list[float] | list[int]]]:
        """Collect per-layer expert load statistics from all MoE blocks.

        모든 MoE 블록으로부터 레이어별 전문가 부하 통계를 수집합니다.
        """
        stats: dict[str, dict[str, list[float] | list[int]]] = {}
        for idx, layer in enumerate(self.layers):
            if isinstance(layer.ffn, TopKMoELayer):
                layer_stats = layer.ffn.get_load_stats()
                if layer_stats:
                    stats[f"layer_{idx}"] = layer_stats
        return stats
