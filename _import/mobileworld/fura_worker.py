"""param_es_worker plus ``fura_s_paper``: FuRA-S with the FuRA-S paper's block layout (FuRA-S/scripts/rl/make_fura_cores.py).

``fura_s_rel`` in param_es_worker SVDs vLLM's fused weights as one matrix. The paper instead factors every projection
separately and blocks o_proj per attention head:

  qkv_proj      -> q, k, v rows, each  W[:, blk_j] = U_j diag(S_j) Vh_j  over closest_factor_pair(in) input blocks
  gate_up_proj  -> gate, up rows, each as above
  o_proj        -> one input block per attention head (n = heads, b = head_dim)
  down_proj     -> closest_factor_pair(in) input blocks
  full rank (r = min(rows, b)), frames frozen, only S trained, noise eps = |S0| z (the paper's D = sigma_rel |S0|).

Written as a residual on the exact base weight, W = W0 + sum_seg U diag(c) Vh, so c = 0 renders W0 bit-exactly (the paper
rebuilds W from fp32 cores, equal up to SVD round-off). With sigma = sigma_rel and kappa = alpha / sigma = 1 the
perturbation and update are the paper's: S +- sigma_rel |S0| z,  S += alpha sigma_rel |S0| sum_i c_i z_i.
Loaded with ``--worker-extension-cls fura_worker.WorkerExtension`` (launch/start_es_server.sh); every other scope behaves
as in param_es_worker (scripts/motivation/param_es_worker.py). The block layout and SVD are mwt.fura's.

``bp_load_weights(path)`` loads HF-named tensors from a safetensors file through the model's own ``load_weights`` (which
maps HF names onto vLLM's fused layers): the weight sync of the BP trainer (methods/bp.py). Do not mix it with pes_* state.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

import param_es_worker as pew
from mwt.fura import block_delta, block_layout, block_svd

PAPER = "fura_s_paper"


class _PaperWeight(pew._Weight):
    """One fused vLLM weight split into the paper's per-projection, per-block SVD segments."""

    def __init__(self, name: str, p: torch.nn.Parameter, segments: List[Tuple[int, int, int, int]]):
        # segments: (row_start, row_end, n_blk, b) with n_blk * b == in_features
        self.name, self.p, self.scope, self.tid, self.bank = name, p, PAPER, pew._tid(name), None
        self.w0 = p.data.clone()
        self.theta: Dict[str, torch.Tensor] = {}
        self.D: Dict[str, torch.Tensor] = {}
        self.segs = []
        for i, (r0, r1, n_blk, b) in enumerate(segments):
            u, s, vh = block_svd(p.data[r0:r1], n_blk, b)
            key = f"c{i}"
            self.theta[key] = torch.zeros_like(s)
            self.D[key] = s.abs()
            self.segs.append((r0, r1, n_blk, b, u, vh, key))
        self.theta0 = {k: v.clone() for k, v in self.theta.items()}

    def phi(self, th, glob=None) -> torch.Tensor:
        out = torch.zeros(self.p.shape, dtype=torch.float32, device=self.p.device)
        for r0, r1, n_blk, b, u, vh, key in self.segs:
            out[r0:r1] = block_delta(u, th[key], vh)
        return out


def _segments(name: str, p: torch.nn.Parameter, mods: dict) -> List[Tuple[int, int, int, int]]:
    out_f, in_f = p.shape
    prefix = name.rsplit(".", 2)[0]  # ...layers.i.self_attn / ...layers.i.mlp
    mod = mods.get(name.rsplit(".", 1)[0])
    if name.endswith("o_proj.weight"):
        n_blk, b = block_layout("o_proj", in_f, mods[f"{prefix}.qkv_proj"].head_size)
        return [(0, out_f, n_blk, b)]
    n_blk, b = block_layout(name.rsplit(".", 2)[1], in_f, 0)
    sizes = list(getattr(mod, "output_sizes", None) or [out_f])
    assert sum(sizes) == out_f, (name, sizes, out_f)
    segs, r0 = [], 0
    for s in sizes:
        segs.append((r0, r0 + s, n_blk, b))
        r0 += s
    return segs


class WorkerExtension(pew.WorkerExtension):
    @torch.no_grad()
    def bp_load_weights(self, path: str) -> int:
        from safetensors.torch import load_file

        sd = load_file(path, device="cpu")
        loaded = pew._find_model(self).load_weights(sd.items())
        torch.cuda.synchronize()
        return len(loaded) if loaded is not None else len(sd)

    @torch.no_grad()
    def pes_init(self, scope: str, rank: int = 16, init_seed: int = 0, uni_d: int = 313344) -> dict:
        if scope != PAPER:
            return super().pes_init(scope, rank, init_seed, uni_d)
        model = pew._find_model(self)
        params = [(n, p) for n, p in model.named_parameters()
                  if p.ndim == 2 and ".layers." in n and not any(s in n for s in pew.SKIP)]
        mods = dict(model.named_modules())
        self._scope, self._bank = scope, None
        self._pes = [_PaperWeight(n, p, _segments(n, p, mods)) for n, p in params]
        torch.cuda.empty_cache()
        dim = sum(w.dim() for w in self._pes)
        segs = {w.name: [(r0, r1, n, b) for r0, r1, n, b, *_ in w.segs] for w in self._pes[:4]}
        return {"weights": len(self._pes), "dim": dim, "names": [w.name for w in self._pes][:12], "segments": segs}
