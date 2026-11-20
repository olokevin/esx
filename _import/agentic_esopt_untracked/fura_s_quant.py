"""4-bit quantization of the FuRA-S small cores (simulated: values are stored dequantized).

With the residual pre-transform the training start is exact whatever the core
precision, so quantization only bends the update directions: the quantized core
is no longer exactly orthogonal. The quantizer therefore picks, per group, the
clipping ratio that minimises the reconstruction error (a small RTN search, as in
AWQ/HQQ-style PTQ), and reports how far ``Vh Vh^T`` is from the identity.
"""

from __future__ import annotations

import torch

RATIOS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)


@torch.no_grad()
def quantize_core_int4(vh: torch.Tensor, group_size: int = 32, bits: int = 4) -> torch.Tensor:
    """Symmetric per-group INT4 fake-quantization of ``vh`` (n_blk, b, b) along its last dim."""
    n_blk, k, b = vh.shape
    g = group_size if b % group_size == 0 else b
    x = vh.float().reshape(n_blk, k, b // g, g)
    qmax = 2 ** (bits - 1) - 1
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    best, best_err = None, None
    for ratio in RATIOS:
        scale = amax * ratio / qmax
        deq = (x / scale).round().clamp(-qmax - 1, qmax) * scale
        err = (deq - x).pow(2).sum(dim=-1, keepdim=True)
        if best is None:
            best, best_err = deq, err
        else:
            take = err < best_err
            best = torch.where(take, deq, best)
            best_err = torch.where(take, err, best_err)
    return best.reshape(n_blk, k, b).to(vh.dtype)


@torch.no_grad()
def orthogonality_error(vh: torch.Tensor) -> float:
    gram = torch.bmm(vh.float(), vh.float().transpose(1, 2))
    eye = torch.eye(gram.shape[-1], device=gram.device)
    return float((gram - eye).abs().max())
