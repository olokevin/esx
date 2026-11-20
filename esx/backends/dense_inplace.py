"""Full-parameter ES (MeZO / ES-at-scale style) on a dense vLLM model, one member at a time.

INT4 weights cannot take a Gaussian perturbation in place, so full ES runs on the
dequantized INT4 checkpoint stored densely in bf16 (the same function as the INT4
base up to GEMM rounding). Every member costs a full write of all target weights,
``W = W_user + sigma * eps(seed)``, and a restore from the user's backup copy; the
population cannot share a forward pass. Updates are applied to the bf16 backup
with seeded stochastic rounding (an ES step is often below one bf16 ulp).

Memory: model (bf16) + backup of the target weights (bf16) + a CPU copy of the base.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from esx.backends.batched_es import stochastic_round
from esx.backends.fura_native import ATTN_SUFFIXES, MLP_SUFFIXES, SKIP_KEYWORDS, stable_tensor_id

_MASK64 = 0xFFFFFFFFFFFFFFFF


class DenseInplaceES:
    def __init__(self, model, targets=MLP_SUFFIXES + ATTN_SUFFIXES):
        self.params: List[tuple] = []  # (name, tensor_id, parameter)
        for name, module in model.named_modules():
            if any(k in name for k in SKIP_KEYWORDS) or name.split(".")[-1] not in targets:
                continue
            w = getattr(module, "weight", None)
            if isinstance(w, torch.nn.Parameter) and w.ndim == 2 and torch.is_floating_point(w):
                self.params.append((name + ".weight", stable_tensor_id(name + ".weight"), w))
        if not self.params:
            raise RuntimeError("no dense weights found")
        self.base_cpu = [p.data.detach().to("cpu").pin_memory() for _, _, p in self.params]
        self.backup = [p.data.detach().clone() for _, _, p in self.params]
        self.perturbed = False

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for _, _, p in self.params))

    def _noise(self, i: int, seed: int) -> torch.Tensor:
        _, tid, p = self.params[i]
        gen = torch.Generator(device=p.device)
        gen.manual_seed((int(seed) ^ tid) & _MASK64)
        return torch.randn(p.shape, generator=gen, dtype=torch.float32, device=p.device)

    @torch.no_grad()
    def reset_to_base(self) -> None:
        for (_, _, p), b, cpu in zip(self.params, self.backup, self.base_cpu):
            b.copy_(cpu, non_blocking=True)
            p.data.copy_(b)
        self.perturbed = False

    @torch.no_grad()
    def perturb(self, seed: int, sigma: float) -> None:
        for i, ((_, _, p), b) in enumerate(zip(self.params, self.backup)):
            p.data.copy_((b.float() + float(sigma) * self._noise(i, seed)).to(p.dtype))
        self.perturbed = True

    @torch.no_grad()
    def restore(self) -> None:
        for (_, _, p), b in zip(self.params, self.backup):
            p.data.copy_(b)
        self.perturbed = False

    @torch.no_grad()
    def update(self, seeds: Sequence[int], weights: Sequence[float], eta: float, round_seed: int) -> None:
        assert not self.perturbed
        scale = float(eta) / float(len(seeds))
        for i, ((_, tid, p), b) in enumerate(zip(self.params, self.backup)):
            total = b.float()
            for seed, w in zip(seeds, weights):
                total.add_(self._noise(i, seed), alpha=scale * float(w))
            gen = torch.Generator(device=p.device)
            gen.manual_seed((int(round_seed) ^ tid) & _MASK64)
            b.copy_(stochastic_round(total, b.dtype, gen))
            p.data.copy_(b)

    def info(self) -> Dict[str, object]:
        return {"n_tensors": len(self.params), "n_params": self.n_params,
                "backup_bytes": int(sum(b.numel() * b.element_size() for b in self.backup))}


class IsoInplaceES(DenseInplaceES):
    """Spectrum-preserving members on a dense vLLM model, one member at a time (ISO / POET style).

    ``W <- C_L W C_R^T`` with ``C_L``, ``C_R`` exactly orthogonal Cayley transforms of block-diagonal
    skew generators laid over a seed-dependent random permutation (a "stochastic primitive": every
    member rotates a different random set of coordinate blocks). Singular values of every weight are
    preserved exactly; only the singular frames U, V move. ``sigma`` is the relative displacement
    ``||dW||_F / ||W||_F`` (entries of the skew generator ~ N(0, 1/b)), shared ``sigma / sqrt(2)``
    per side when ``sides == "both"``. Left rotations never mix the fused q/k/v or gate/up segments.
    Uses the same helpers (and so the same draws for a given generator state) as ``structured_es``.
    """

    def __init__(self, model, targets=MLP_SUFFIXES + ATTN_SUFFIXES, block_size: int = 128, sides: str = "both"):
        super().__init__(model, targets)
        from esx.backends.structured_es import _block_size, _output_segments

        assert sides in ("both", "right", "left")
        self.sides = sides
        segs = _output_segments(model)
        self.layout = []
        for _, _, p in self.params:
            m, n = p.shape
            seg = segs.get(id(p), [m])
            self.layout.append((seg, _block_size(seg, block_size), _block_size([n], block_size)))

    @torch.no_grad()
    def perturb(self, seed: int, sigma: float) -> None:
        from esx.backends.structured_es import _cayley, _rotate_columns, _rotate_rows, _skew

        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for i, ((_, tid, p), b) in enumerate(zip(self.params, self.backup)):
                seg, bl, br = self.layout[i]
                gen = torch.Generator(device=p.device)
                gen.manual_seed((int(seed) ^ tid) & _MASK64)
                m, n = p.shape
                parts, off = [], 0
                for s in seg:
                    parts.append(torch.randperm(s, generator=gen, device=p.device) + off)
                    off += s
                perm_l = torch.cat(parts)
                perm_r = torch.randperm(n, generator=gen, device=p.device)
                scale = float(sigma) * (0.5 ** 0.5 if self.sides == "both" else 1.0)
                c_l = _cayley(_skew(m // bl, bl, gen, p.device), scale)
                c_r = _cayley(_skew(n // br, br, gen, p.device), scale)
                w = b.float()
                if self.sides in ("both", "left"):
                    w = _rotate_rows(w, perm_l, c_l)
                if self.sides in ("both", "right"):
                    w = _rotate_columns(w, perm_r, c_r)
                p.data.copy_(w.to(p.dtype))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        self.perturbed = True

    def update(self, *a, **k):
        raise NotImplementedError("IsoInplaceES is a sampler only")
