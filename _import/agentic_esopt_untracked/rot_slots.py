"""Batched (slot-routed) weight perturbations for vLLM: every sequence in a batch can carry its own member.

Like ``FuraNative`` this wraps each linear's ``quant_method.apply`` and reads the per-token slot vector that
``SlotEngine`` sets before every forward pass. Slot 0 is the unmodified base (bit-identical). Modes:

* ``add``   ``y <- y + sigma * rms(y) * eps_m``: an activation-scaled steering vector per member and layer.
* ``out``   ``y <- C_m y``: block-wise rotation of the output (``W -> C W``, ISO with only U moved).
* ``in``    ``x <- C_m^T x``: block-wise rotation of the input (``W -> W C``, only V moved).
* ``both``  both sides (``W -> C_L W C_R^T``, ISO with U and V moved).

The rotations are spectrum-preserving and kept small on purpose: ``C`` is a product of ``stages`` layers of
disjoint 2x2 Givens rotations (block size 2) over fixed random pairings of the coordinates (per tensor and stage;
pairs never cross the fused q/k/v or gate/up segments of the output; one side of each pair is contiguous or strided
so the kernel's loads coalesce, see ``_pairing``). A member is a vector of ``stages * d / 2``
angles per side, theta ~ N(0, s^2) with s = sigma / sqrt(stages * n_sides), so ``||dW||_F / ||W||_F ~ sigma`` as
for ``IsoInplaceES``. One Triton kernel per stage applies the per-token rotation, so many members share a batch
at the cost of a few elementwise passes over the activations.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from algorithms.es.fura_native import ATTN_SUFFIXES, MLP_SUFFIXES, SKIP_KEYWORDS, stable_tensor_id

_MASK64 = 0xFFFFFFFFFFFFFFFF

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _givens_kernel(X, Y, SLOT, I0, I1, TH, d, P, BLOCK: tl.constexpr):
        t = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = offs < P
        s = tl.load(SLOT + t)
        i0 = tl.load(I0 + offs, mask=m, other=0)
        i1 = tl.load(I1 + offs, mask=m, other=0)
        row = t.to(tl.int64) * d
        a = tl.load(X + row + i0, mask=m, other=0.0).to(tl.float32)
        b = tl.load(X + row + i1, mask=m, other=0.0).to(tl.float32)
        th = tl.load(TH + s * P + offs, mask=m, other=0.0)
        c = tl.where(th == 0.0, 1.0, tl.cos(th))  # exact identity for zero angles (the base slot)
        sn = tl.where(th == 0.0, 0.0, tl.sin(th))
        tl.store(Y + row + i0, (c * a - sn * b).to(Y.dtype.element_ty), mask=m)
        tl.store(Y + row + i1, (sn * a + c * b).to(Y.dtype.element_ty), mask=m)

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


def givens_apply(x: torch.Tensor, slots: torch.Tensor, i0: torch.Tensor, i1: torch.Tensor,
                 theta: torch.Tensor) -> torch.Tensor:
    """One stage: ``y[t, i0[p]], y[t, i1[p]] = R(theta[slot_t, p]) (x[t, i0[p]], x[t, i1[p]])``.

    ``i0`` / ``i1`` int32, ``theta`` (n_slots, P) fp32 contiguous."""
    x = x.contiguous()
    T, d = x.shape
    P = i0.shape[0]
    if _HAS_TRITON and x.is_cuda:
        y = torch.empty_like(x)
        BLOCK = 1024
        _givens_kernel[(T, triton.cdiv(P, BLOCK))](x, y, slots, i0, i1, theta, d, P, BLOCK=BLOCK)
        return y
    a, b = x[:, i0.long()].float(), x[:, i1.long()].float()
    th = theta[slots]
    c, s = torch.cos(th), torch.sin(th)
    y = torch.empty_like(x)
    y[:, i0.long()] = (c * a - s * b).to(x.dtype)
    y[:, i1.long()] = (s * a + c * b).to(x.dtype)
    return y


def _pairing(n: int, segments: Sequence[int], gen: torch.Generator, device, stage: int = 0):
    """Disjoint pairs inside each segment, half of every pair on a contiguous / strided set so the loads coalesce:
    even stages pair the first half of a segment with a random permutation of its second half, odd stages pair the
    even coordinates with a random permutation of the odd ones. Returns int32 (i0, i1) of length n // 2."""
    a, b, off = [], [], 0
    for seg in segments:
        h = seg // 2
        if stage % 2 == 0:
            a.append(torch.arange(h) + off)
            b.append(torch.randperm(h, generator=gen) + off + h)
        else:
            a.append(torch.arange(0, 2 * h, 2) + off)
            b.append(2 * torch.randperm(h, generator=gen) + 1 + off)
        off += seg
    return (torch.cat(a).to(device=device, dtype=torch.int32), torch.cat(b).to(device=device, dtype=torch.int32))


class _Layer:
    def __init__(self, name, module, out_f, in_f, segments, stages):
        self.name, self.module = name, module
        self.out_f, self.in_f = out_f, in_f
        self.tid = stable_tensor_id(name + ".weight")
        dev = module.weight.device
        g = torch.Generator().manual_seed(self.tid & 0x7FFFFFFFFFFFFFFF)
        self.pin = [_pairing(in_f, [in_f], g, dev, st) for st in range(stages)]
        self.pout = [_pairing(out_f, segments, g, dev, st) for st in range(stages)]
        self.tab = {}  # side -> list per stage of (cos, sin) tables (n_slots, P) fp32 ; "add" -> (n_slots, out)


class _Method:
    def __init__(self, original, owner: "RotSlots", layer: _Layer):
        self.original, self.owner, self.layer = original, owner, layer

    def apply(self, module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        o, L = self.owner, self.layer
        slots = o.token_slots(x.shape[0] if x.dim() == 2 else x.reshape(-1, x.shape[-1]).shape[0])
        if slots is None or not L.tab:
            return self.original.apply(module, x, bias)
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        if "in" in L.tab:
            for (i0, i1), th in zip(L.pin, L.tab["in"]):
                x2 = givens_apply(x2, slots, i0, i1, th)
        y = self.original.apply(module, x2, bias)
        if "out" in L.tab:
            for (i0, i1), th in zip(L.pout, L.tab["out"]):
                y = givens_apply(y, slots, i0, i1, th)
        if "add" in L.tab:
            rms = y.float().pow(2).mean(-1, keepdim=True).sqrt()
            y = (y.float() + rms * L.tab["add"].index_select(0, slots)).to(y.dtype)
        return y.reshape(*lead, y.shape[-1])

    def __getattr__(self, name):
        return getattr(self.original, name)


class RotSlots:
    def __init__(self, model, mode: str = "both", stages: int = 2, targets=MLP_SUFFIXES + ATTN_SUFFIXES):
        from algorithms.es.structured_es import _output_segments

        assert mode in ("add", "out", "in", "both")
        self.mode, self.stages = mode, int(stages)
        self.layers: List[_Layer] = []
        self._slots: Optional[torch.Tensor] = None
        self.enabled = True
        segs = _output_segments(model)
        for module in model.modules():  # tensor parallel: fused segments of the local shard (global sizes / tp)
            w, sizes, tp = getattr(module, "weight", None), getattr(module, "output_sizes", None), getattr(module, "tp_size", 1)
            if w is not None and sizes and len(sizes) > 1 and tp > 1 and id(w) not in segs \
                    and getattr(w, "ndim", 0) == 2 and sum(sizes) == tp * int(w.shape[0]):
                segs[id(w)] = [int(x) // tp for x in sizes]
        for name, module in model.named_modules():
            if not hasattr(module, "quant_method") or module.quant_method is None:
                continue
            if any(k in name for k in SKIP_KEYWORDS) or name.split(".")[-1] not in targets:
                continue
            w = getattr(module, "weight", None)
            if not isinstance(w, torch.nn.Parameter) or w.ndim != 2:
                continue
            out_f, in_f = w.shape
            layer = _Layer(name, module, int(out_f), int(in_f), segs.get(id(w), [int(out_f)]), self.stages)
            module.quant_method = _Method(module.quant_method, self, layer)
            self.layers.append(layer)
        if not self.layers:
            raise RuntimeError("no dense linear layers found")

    # routing (same contract as FuraNative)
    def set_token_slots(self, slots: Optional[torch.Tensor]) -> None:
        self._slots = slots

    def token_slots(self, tokens: int) -> Optional[torch.Tensor]:
        s = self._slots
        if s is None or not self.enabled:
            return None
        if s.shape[0] == tokens:
            return s
        return F.pad(s, (0, tokens - s.shape[0])) if s.shape[0] < tokens else s[:tokens]

    def sides(self) -> List[str]:
        return {"both": ["in", "out"], "in": ["in"], "out": ["out"], "add": ["add"]}[self.mode]

    def member_noise(self, L: _Layer, side: str, stage: int, seed: int, sigma: float) -> torch.Tensor:
        """Angle noise of one member (or the add-mode vector for side == "add"), already scaled by sigma."""
        dev = L.module.weight.device
        if side == "add":
            g = torch.Generator(device=dev).manual_seed((int(seed) ^ L.tid) & _MASK64)
            return float(sigma) * torch.randn(L.out_f, generator=g, device=dev)
        P = (L.pin if side == "in" else L.pout)[0][0].shape[0]
        mix = (int(seed) ^ L.tid ^ (0x9E3779B97F4A7C15 * (stage + 1 + 7 * (side == "out")))) & _MASK64
        g = torch.Generator(device=dev).manual_seed(mix)
        n_rot = len([x for x in self.sides() if x != "add"])
        return float(sigma) / math.sqrt(self.stages * n_rot) * torch.randn(P, generator=g, device=dev)

    def zero_center(self) -> List[dict]:
        """Per layer {side: tensor (stages, P)} (or {"add": (out,)}) of zeros, fp32: the trainable centre."""
        out = []
        for L in self.layers:
            dev = L.module.weight.device
            c = {}
            for side in self.sides():
                if side == "add":
                    c[side] = torch.zeros(L.out_f, device=dev)
                else:
                    P = (L.pin if side == "in" else L.pout)[0][0].shape[0]
                    c[side] = torch.zeros(self.stages, P, device=dev)
            out.append(c)
        return out

    def layout(self) -> List[dict]:
        """What an HF-side implementation needs to reproduce every layer exactly (indices on CPU)."""
        return [{"name": L.name, "tid": L.tid, "out_f": L.out_f, "in_f": L.in_f,
                 "pin": [(a.cpu(), b.cpu()) for a, b in L.pin], "pout": [(a.cpu(), b.cpu()) for a, b in L.pout]}
                for L in self.layers]

    @torch.no_grad()
    def set_slots(self, entries: Sequence[tuple], center: Optional[List[dict]] = None) -> None:
        """Slot i + 1 = member ``(uid_ignored, seed, sigma)`` around ``center``; slot 0 = the centre (default:
        the base model, all angles zero).

        With a centre, slot 0 is not the base model, so ``route_all`` tells the engine to route batches whose
        requests are all slot 0 as well (an all-zero slot vector instead of None)."""
        self.route_all = center is not None
        if center is not None:
            for li, L in enumerate(self.layers):
                L.tab = {}
                for side in self.sides():
                    if side == "add":
                        rows = [center[li]["add"]] + [center[li]["add"] + self.member_noise(L, "add", 0, sd, sg)
                                                      for _, sd, sg in entries]
                        L.tab["add"] = torch.stack(rows)
                        continue
                    per_stage = []
                    for st in range(self.stages):
                        c0 = center[li][side][st]
                        th = torch.stack([c0] + [c0 + self.member_noise(L, side, st, sd, sg) for _, sd, sg in entries])
                        per_stage.append(th.float().contiguous())
                    L.tab[side] = per_stage
            return
        sides = {"both": ["in", "out"], "in": ["in"], "out": ["out"], "add": ["add"]}[self.mode]
        rot_sides = [s for s in sides if s != "add"]
        for L in self.layers:
            dev = L.module.weight.device
            L.tab = {}
            for side in sides:
                if side == "add":
                    rows = [torch.zeros(L.out_f, device=dev)]
                    for _, seed, sigma in entries:
                        g = torch.Generator(device=dev).manual_seed((int(seed) ^ L.tid) & _MASK64)
                        rows.append(float(sigma) * torch.randn(L.out_f, generator=g, device=dev))
                    L.tab["add"] = torch.stack(rows)
                    continue
                P = (L.pin if side == "in" else L.pout)[0][0].shape[0]
                per_stage = []
                for st in range(self.stages):
                    th = [torch.zeros(P, device=dev)]
                    for _, seed, sigma in entries:
                        mix = (int(seed) ^ L.tid ^ (0x9E3779B97F4A7C15 * (st + 1 + 7 * (side == "out")))) & _MASK64
                        g = torch.Generator(device=dev).manual_seed(mix)
                        scale = float(sigma) / math.sqrt(self.stages * len(rot_sides))
                        th.append(scale * torch.randn(P, generator=g, device=dev))
                    th = torch.stack(th)
                    per_stage.append(th.float().contiguous())
                L.tab[side] = per_stage

    def clear_slots(self) -> None:
        self.route_all = False
        for L in self.layers:
            L.tab = {}

    def info(self):
        return {"n_layers": len(self.layers), "mode": self.mode, "stages": self.stages}
