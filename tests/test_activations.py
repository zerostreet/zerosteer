import torch

from hyperalpha_vla.activations import Entmax15, Sparsemax, entmax15, sparsemax


def test_entmax15_and_sparsemax_simplex_and_sparsity():
    logits = torch.tensor([[3.0, 1.0, -5.0, -8.0], [0.5, 0.5, -10.0, -10.0]], requires_grad=True)
    p_ent = entmax15(logits, dim=-1)
    p_sp = sparsemax(logits, dim=-1)

    assert torch.allclose(p_ent.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(p_sp.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert (p_ent >= 0).all()
    assert (p_sp >= 0).all()
    # Low-logit tail elements must be truncated to exact 0.0
    assert (p_ent[:, -1] == 0.0).all()
    assert (p_sp[:, -1] == 0.0).all()

    loss = (p_ent[:, 0] - p_ent[:, 1]).sum() + p_sp[:, 0].sum()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_activation_modules_run():
    x = torch.randn(4, 5)
    assert torch.allclose(Entmax15()(x).sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(Sparsemax()(x).sum(dim=-1), torch.ones(4), atol=1e-5)
