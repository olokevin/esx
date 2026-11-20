"""FuRA-S with the FuRA-S paper's block layout, shared by the vLLM ES worker (workers/fura_worker.py), the BP trainer
(methods/bp.py) and the merges (workers/merge_worker.py).

Every attention / MLP projection W (out x in) is split into n_blk input blocks of width b, each factored as
W[:, blk_j] = U_j diag(S_j) Vh_j (full rank, frames frozen). The trained residual c (n_blk, r) moves the block
singular values: W = W0 + sum_j U_j diag(c_j) Vh_j. o_proj is blocked per attention head (b = head_dim); every other
projection by closest_factor_pair(in). ES perturbs c with noise |S0| z; BP trains t with c = |S0| t.

vLLM fuses q/k/v and gate/up into one weight; ``HF_FUSED`` lists the HF projections of each fused weight in row order,
which is how BP checkpoints (HF names) map onto the worker's segments.
"""
from __future__ import annotations

from typing import Tuple

import torch

from algorithms.es.structured_es import closest_factor_pair

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
HF_FUSED = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "gate_up_proj": ("gate_proj", "up_proj"),
            "o_proj": ("o_proj",), "down_proj": ("down_proj",)}


def block_layout(proj: str, in_features: int, head_dim: int) -> Tuple[int, int]:
    """(n_blk, b) of a projection named ``proj`` (HF or vLLM name)."""
    if proj == "o_proj":
        return in_features // head_dim, head_dim
    return closest_factor_pair(in_features)


def block_svd(w: torch.Tensor, n_blk: int, b: int):
    """SVD of the input blocks of w (rows x n_blk*b): u (n_blk, rows, r) in w's dtype, s (n_blk, r), vh (n_blk, r, b)
    in fp32."""
    rows = w.shape[0]
    wb = w.float().reshape(rows, n_blk, b).permute(1, 0, 2)
    u, s, vh = torch.linalg.svd(wb, full_matrices=False)
    return u.to(w.dtype).contiguous(), s, vh.float().contiguous()


def block_delta(u: torch.Tensor, c: torch.Tensor, vh: torch.Tensor) -> torch.Tensor:
    """sum_j U_j diag(c_j) Vh_j as a (rows, n_blk*b) fp32 matrix."""
    w = torch.bmm(u.float() * c.unsqueeze(1), vh)  # (n_blk, rows, b)
    n_blk, rows, b = w.shape
    return w.permute(1, 0, 2).reshape(rows, n_blk * b)
