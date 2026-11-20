"""Per-sequence residual-stream steering for vLLM (``SlotEngine.slot_component``): every sequence (slot) carries
its own fixed latent ``z ~ N(0, I_k)``; at decoder layer L each of its tokens gets

    x_L <- x_L + alpha * ||x_L|| * u_slot,      u_slot = unit(U_L^T z)

where x_L = hidden + residual is the residual stream after layer L and U_L (k x d) spans the subspace
(``rand``: random orthonormal; a loaded basis, e.g. the top principal directions of x_L). The latent is persistent
over the whole trajectory (the slot stays with the job), unlike token-level temperature noise. Slot 0 = base.
Cost: one gather + add per token per steered layer.

``capture=True`` accumulates sum and second moment of x_L over slot-0 tokens (every ``stride``-th token) so
that a PCA basis can be built from base-policy states: ``cov(L)``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch


def decoder_layers(model) -> torch.nn.ModuleList:
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.endswith("layers"):
            if best is None or len(mod) > len(best):
                best = mod
    assert best is not None, "no decoder ModuleList found"
    return best


class SteerSlots:
    def __init__(self, model, layers: Sequence[int], basis: Optional[Dict[int, torch.Tensor]] = None, k: int = 16,
                 capture_layers: Sequence[int] = (), stride: int = 4):
        self.blocks = decoder_layers(model)
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.layers = list(layers)
        self.k = k
        self.basis = {}
        for L in self.layers:
            if basis is not None and L in basis:
                U = basis[L][:k].to(self.device, torch.float32)
            else:
                g = torch.Generator(device="cpu").manual_seed(1000 + L)
                U = torch.linalg.qr(torch.randn(self._width(), k, generator=g))[0].T
            self.basis[L] = U.to(self.device, torch.float32)  # k x d, rows orthonormal
        self.vec: Dict[int, Optional[torch.Tensor]] = {L: None for L in self.layers}
        self.bias: Dict[int, torch.Tensor] = {}  # L -> (d,) fixed offset added to every steered slot (directed sampler)
        self.slots: Optional[torch.Tensor] = None
        self.capture_layers = list(capture_layers)
        self.stride = stride
        self.mom = {L: None for L in self.capture_layers}
        self.handles = [self.blocks[L].register_forward_hook(self._hook(L))
                        for L in sorted(set(self.layers) | set(self.capture_layers))]

    def _width(self) -> int:
        norm = getattr(self.blocks[0], "input_layernorm", None)
        if norm is not None and getattr(norm, "weight", None) is not None:
            return norm.weight.shape[0]
        return max(m.weight.shape[0] for m in self.blocks[0].modules()
                   if getattr(m, "weight", None) is not None and m.weight.dim() == 1)

    # -------------------------------------------------------------- slots
    def set_token_slots(self, slots: Optional[torch.Tensor]) -> None:
        self.slots = slots

    def set_slots(self, entries: Sequence[tuple]) -> None:
        """entries[i] = (seed, alpha) of slot i + 1."""
        for L in self.layers:
            U = self.basis[L]
            V = torch.zeros(len(entries) + 1, U.shape[1], device=self.device, dtype=torch.float32)
            for i, (seed, alpha) in enumerate(entries):
                V[i + 1] = alpha * self.unit(seed, L) if alpha else 0.0
                if L in self.bias:
                    V[i + 1] += self.bias[L]
            self.vec[L] = V

    def set_slots_z(self, zs: Sequence[torch.Tensor], alpha: float) -> None:
        """Slot i + 1 gets the explicit latent zs[i] (k,), unnormalised: delta = alpha * ||x|| * (U^T z) / sqrt(k),
        so ||delta|| / ||x|| ~ alpha for z ~ N(0, I) and the latent's length matters (adaptive samplers)."""
        for L in self.layers:
            U = self.basis[L]
            Z = torch.stack([torch.as_tensor(z, dtype=torch.float32) for z in zs]).to(self.device)
            V = torch.zeros(len(zs) + 1, U.shape[1], device=self.device, dtype=torch.float32)
            V[1:] = alpha * (Z[:, : U.shape[0]] @ U) / U.shape[0] ** 0.5
            self.vec[L] = V

    def set_bias(self, dirs: Dict[int, torch.Tensor], beta: float) -> None:
        """Every steered slot gets beta * unit(dirs[L]) on top of its random part (relative to ||x||)."""
        self.bias = {L: beta * (d.flatten().to(self.device, torch.float32) / d.norm()) for L, d in dirs.items()
                     if L in self.layers}

    def unit(self, seed: int, L: int) -> torch.Tensor:
        """unit(U_L^T z) for the latent z of ``seed`` (shared with HFSteer)."""
        U = self.basis[L]
        g = torch.Generator(device="cpu").manual_seed(int(seed) * 131 + L)
        u = torch.randn(U.shape[0], generator=g).to(self.device) @ U
        return u / u.norm()

    def clear_slots(self) -> None:
        self.vec = {L: None for L in self.layers}

    # -------------------------------------------------------------- hook
    def _hook(self, L):
        def hook(mod, inputs, out):
            if not isinstance(out, tuple) or len(out) != 2:
                return None
            h, res = out
            if L in self.mom:
                x = (h + res) if res is not None else h
                xs = x
                if self.slots is not None:
                    xs = x[: self.slots.shape[0]][self.slots == 0]
                xs = xs[:: self.stride].float()
                if xs.shape[0]:
                    m = self.mom[L]
                    s1, s2 = xs.sum(0), xs.T @ xs
                    self.mom[L] = [s1, s2, xs.shape[0]] if m is None else [m[0] + s1, m[1] + s2, m[2] + xs.shape[0]]
            V = self.vec.get(L)
            if V is None or self.slots is None:
                return None
            n = self.slots.shape[0]
            x = (h[:n] + res[:n]) if res is not None else h[:n]
            delta = V[self.slots] * x.float().norm(dim=-1, keepdim=True)
            h = h.clone()
            h[:n] += delta.to(h.dtype)
            return (h, res)
        return hook

    def cov(self, L: int):
        s1, s2, n = self.mom[L]
        mu = s1 / n
        return mu, s2 / n - torch.outer(mu, mu), n

    def pca(self, L: int, k: int) -> torch.Tensor:
        _, C, _ = self.cov(L)
        w, V = torch.linalg.eigh(C)
        return V[:, torch.argsort(w, descending=True)[:k]].T.contiguous()  # k x d


class MultiSlots:
    """Several slot components behind one ``SlotEngine.slot_component``."""

    def __init__(self, comps):
        self.comps = list(comps)

    @property
    def route_all(self):
        return any(getattr(c, "route_all", False) for c in self.comps)

    def set_token_slots(self, slots):
        for c in self.comps:
            c.set_token_slots(slots)


class HFSteer:
    """The same steering on a HuggingFace model for backprop: row b of the batch gets alpha * ||x|| * u(seed_b)
    added to the output of decoder layer L at every position (as vLLM does for every token of its slot)."""

    def __init__(self, hf_model, layers: Sequence[int], ref: SteerSlots):
        self.blocks = decoder_layers(hf_model)
        self.layers, self.ref = list(layers), ref
        self.U: Dict[int, Optional[torch.Tensor]] = {L: None for L in self.layers}
        self.handles = [self.blocks[L].register_forward_hook(self._hook(L)) for L in self.layers]

    def set(self, seeds: Optional[Sequence[Optional[int]]], alpha: float) -> None:
        if seeds is None:
            self.U = {L: None for L in self.layers}
            return
        for L in self.layers:
            rows = [torch.zeros(self.ref.basis[L].shape[1], device=self.ref.device) if sd is None
                    else alpha * self.ref.unit(sd, L) for sd in seeds]
            self.U[L] = torch.stack(rows)  # B x d

    def _hook(self, L):
        def hook(mod, inputs, out):
            U = self.U[L]
            if U is None:
                return None
            x = out[0] if isinstance(out, tuple) else out
            x = x + (U[:, None, :] * x.float().norm(dim=-1, keepdim=True)).to(x.dtype)
            return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
        return hook
