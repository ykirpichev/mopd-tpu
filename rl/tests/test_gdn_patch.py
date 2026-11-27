"""CPU checks that the Gated DeltaNet speed patches are exact and actually cut dispatched ops."""
import os
import sys

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from qrl import gdn_patch  # noqa: E402
from tests.test_cpu import tiny_model  # noqa: E402,F401

transformers_gdn = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
REFERENCE = transformers_gdn.torch_chunk_gated_delta_rule


class OpCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n += 1
        return func(*args, **(kwargs or {}))


def _inputs(seq=700, heads=4, dim=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, seq, heads, dim, generator=g)
    k = torch.randn(1, seq, heads, dim, generator=g)
    v = torch.randn(1, seq, heads, dim, generator=g)
    # Realistic gates: log-decay in (-softplus range) and beta in (0, 1).
    gate = -torch.nn.functional.softplus(torch.randn(1, seq, heads, generator=g)) * 0.5
    beta = torch.sigmoid(torch.randn(1, seq, heads, generator=g))
    return [t.requires_grad_(True) for t in (q, k, v, gate, beta)]


def _run(fn, inputs, **kw):
    out, _ = fn(*inputs, use_qk_l2norm_in_kernel=True, **kw)
    w = torch.randn_like(out, generator=torch.Generator().manual_seed(1))
    grads = torch.autograd.grad((out * w).sum(), inputs)
    return out.detach(), [x.detach() for x in grads]


@pytest.mark.parametrize("chunk,solve", [(64, "triangular"), (128, "triangular"), (256, "triangular"),
                                         (64, "neumann"), (256, "neumann")])
def test_chunk_rule_matches_reference(chunk, solve):
    ref_out, ref_grads = _run(REFERENCE, _inputs())
    out, grads = _run(gdn_patch.chunk_gated_delta_rule, _inputs(), chunk_size=chunk, solve=solve)
    scale = ref_out.abs().max()
    assert (out - ref_out).abs().max() / scale < 2e-4
    for a, b in zip(grads, ref_grads):
        assert (a - b).abs().max() / (b.abs().max() + 1e-12) < 2e-3


def test_neumann_inverse_is_exact():
    torch.manual_seed(0)
    a = torch.randn(3, 64, 64) * 0.3
    inv = gdn_patch.unit_lower_inverse_neumann(a)
    full = torch.eye(64) + a.tril(-1)
    assert torch.allclose(full @ inv, torch.eye(64).expand(3, 64, 64), atol=1e-4)


def test_contiguous_grad_layout():
    x = torch.randn(2, 16, 32).transpose(1, 2).requires_grad_(True)  # [2, 32, 16] with transposed strides
    y = gdn_patch.contiguous_grad(x)
    (g,) = torch.autograd.grad((y.transpose(1, 2).contiguous() * 2).sum(), x)
    assert torch.equal(g, torch.full_like(g, 2.0))


def test_bigger_chunks_cut_dispatched_ops():
    counts = {}
    for chunk in (64, 256):
        c = OpCounter()
        with c:
            _run(gdn_patch.chunk_gated_delta_rule, _inputs(seq=2048), chunk_size=chunk)
        counts[chunk] = c.n
    assert counts[256] < 0.5 * counts[64], counts


def test_model_level_patch(tiny_model):  # noqa: F811
    from transformers import AutoModelForCausalLM

    from qrl.learner import token_logprobs

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(tiny_model, dtype=torch.float32)
    ids = torch.randint(0, 1000, (2, 300))
    ref = token_logprobs(model, ids[:, :-1], ids[:, 1:])
    orig = transformers_gdn.torch_chunk_gated_delta_rule
    calls = []
    try:
        desc = gdn_patch.apply(chunk_size=128, solve="neumann", contiguous_grads=True)
        patched = transformers_gdn.torch_chunk_gated_delta_rule

        def spy(*a, **k):
            calls.append(1)
            return patched(*a, **k)

        transformers_gdn.torch_chunk_gated_delta_rule = spy
        out = token_logprobs(model, ids[:, :-1], ids[:, 1:])
        out.sum().backward()
    finally:
        transformers_gdn.torch_chunk_gated_delta_rule = orig
    assert "chunk_size=128" in desc and calls
    assert torch.allclose(out, ref, atol=1e-4)
