"""Learner-side speed patches for Qwen3.5's Gated DeltaNet torch fallback (no FLA kernels on TPU).

The fallback scans the sequence in Python over chunks of 64 tokens, so an 8k-token sequence issues
128 iterations x ~6 ops per layer in forward, again in backward and again under recompute. In eager
TorchTPU that dispatch overhead dominates. Three knobs, all exact up to fp32 rounding:

- chunk_size: fewer, larger chunks (the chunked delta rule is exact for any chunk size).
- solve="neumann": replace the unit-triangular solves with log2(chunk) matmuls,
  (I + N)^-1 = prod_j (I + (-N)^(2^j)) for strictly-lower (nilpotent) N.
- contiguous_grads: the fallback transposes beta/g to [B, H, S]; their gradients come back with
  [B, S, H] shape but [S*H, 1, S] strides, which torch.compile's linear-backward decomposition
  cannot .view() (the `view(): cannot create a view of shape [4096, 16]` error). A no-op autograd
  function makes those gradients contiguous.
"""
import functools
import math

import torch
import torch.nn.functional as F


class _ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.contiguous()


def contiguous_grad(x):
    return _ContiguousGrad.apply(x) if x.requires_grad else x


def unit_lower_inverse_neumann(a):
    """Inverse of I + strict_lower(a) via repeated squaring; a: [..., C, C]."""
    c = a.shape[-1]
    n = -a.tril(-1)
    inv = torch.eye(c, dtype=a.dtype, device=a.device) + n
    p = n
    for _ in range(max(0, math.ceil(math.log2(c)) - 1)):
        p = p @ p
        inv = inv + inv @ p
    return inv


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None,
                           output_final_state=False, use_qk_l2norm_in_kernel=False, solve="triangular", **kwargs):
    """Same math and signature as transformers' torch_chunk_gated_delta_rule (5.19), with a selectable solver."""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    query = query * query.shape[-1] ** -0.5
    pad = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad)) for x in (beta, decay))
    num_chunks = (sequence_length + pad) // chunk_size
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                                  for x in (query, key, k_beta, v_beta)]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum = decay.cumsum(dim=3)
    pair = (cum.unsqueeze(4) - cum.unsqueeze(3)).masked_fill(upper, float("-inf")).exp()
    ut = (k_beta @ key.transpose(-1, -2)) * pair
    intra = (query @ key.transpose(-1, -2)) * pair
    dk = k_beta * cum.exp().unsqueeze(-1)
    if solve == "neumann":
        inv = unit_lower_inverse_neumann(ut)
        new_values, k_cumdecay = inv @ v_beta, inv @ dk
    else:
        new_values = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
        k_cumdecay = torch.linalg.solve_triangular(ut, dk, upper=False, unitriangular=True)
    state = (torch.zeros((batch_size, num_v_heads, k_head_dim, v_head_dim), dtype=new_values.dtype,
                         device=new_values.device) if initial_state is None else initial_state.to(new_values))
    query = query * cum.exp().unsqueeze(-1)
    key = key * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]
    outs = []
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        outs.append(query[:, :, i] @ state + intra[:, :, i] @ v_new)
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
    out = torch.stack(outs, dim=2).reshape(batch_size, num_v_heads, -1, v_head_dim)[:, :, :sequence_length]
    out = out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return out, (state if output_final_state else None)


def apply(chunk_size=None, solve=None, contiguous_grads=False):
    """Patch transformers' Qwen3.5 module in place. Returns a description of what was patched."""
    if not chunk_size and not solve and not contiguous_grads:
        return "gdn: unpatched"
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    impl = functools.partial(chunk_gated_delta_rule, chunk_size=chunk_size or 64, solve=solve or "triangular")

    @functools.wraps(m.torch_chunk_gated_delta_rule)
    def patched(query, key, value, g, beta, **kw):
        kw.pop("chunk_size", None)
        if contiguous_grads:
            g, beta = contiguous_grad(g), contiguous_grad(beta)
        return impl(query, key, value, g, beta, **kw)

    m.torch_chunk_gated_delta_rule = patched
    return f"gdn: chunk_size={chunk_size or 64} solve={solve or 'triangular'} contiguous_grads={contiguous_grads}"
