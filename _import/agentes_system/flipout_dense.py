"""Full-parameter ES on a dense vLLM model with one perturbation per sequence (Flipout).

Flipout (Wen et al., ICLR 2018, arXiv:1803.04386): all members of a generation share one
base noise matrix ``E`` per weight, and member ``n`` flips its signs with two random sign
vectors, ``dW_n = sigma * E o (r_n s_n^T)``. The layer then needs no per-member weight:

    y = W x + sigma * r_n o (E (s_n o x))

so every token carries a slot id, slot ``k`` selects row ``k`` of the sign tables, and one
engine batch mixes the whole population. Slot 0 has all-zero rows: the unperturbed centre.
The antithetic partner of a member uses the same ``s`` and ``-r``.

The ES step stays exact and cheap because the direction is a masked low-rank matrix:

    sum_n w_n eps_n = E o (R^T diag(w) S),   R (N, out), S (N, in) the sign rows.

Graph-safe by construction: ``E``, the sign tables and the per-token slot vector are
module buffers that are only written in place, and the extra GEMM always runs, so the
component can be installed before vLLM captures CUDA graphs.

Memory: ``E`` is as large as the perturbed weights (``noise_dtype``), the tables are
``(n_slots + 1) * (in + out)`` per layer. ``master='cpu'`` keeps an fp32 copy of the
centre in host memory so updates accumulate exactly (the live weight is its bf16
rounding); ``master='sr'`` keeps no copy and writes updates with seeded stochastic rounding.

``variant='core'`` (needs ``rank``) is a cheaper, different perturbation family: ``dW_n = sigma * A diag(c_n) B`` with
one sign vector ``c_n`` of length k per member, so the per-token work is a k-wide gather instead of two sign rows of
the layer's input and output width. All members of a generation then share the column space of A and the row space
of B (fresh every generation).

Full-parameter coverage (``extras``): ``embed`` and ``head`` are Flipout layers like the others (the embedding
is a linear layer on a one-hot input, so its term is a second row lookup; the head term is applied to the
sampled rows outside the captured graph). ``norm`` gains and ``bias`` vectors are small, so every member gets
its own Gaussian direction in a per-slot table: ``out + out * D[slot]`` with ``D = sigma * e / gain``, and
``y + sigma * e[slot]``. Their fp32 masters stay on the GPU.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

SKIP_KEYWORDS = ("embed_tokens", "lm_head", "visual")
MLP_SUFFIXES = ("gate_up_proj", "down_proj")
ATTN_SUFFIXES = ("qkv_proj", "o_proj")
EXTRAS = ("embed", "head", "norm", "bias")
_MASK63 = 0x7FFFFFFFFFFFFFFF


def stable_tensor_id(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little")


def stochastic_round(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """fp32 -> bf16 with E[result] = x (bf16 is the top 16 bits of fp32)."""
    bits = x.contiguous().view(torch.int32)
    low = (bits & 0xFFFF).float()
    draw = torch.rand(x.shape, generator=generator, dtype=torch.float32, device=x.device) * 65536.0
    up = (draw < low).to(torch.int32) << 16
    return ((bits & ~0xFFFF) + up).view(torch.float32).to(torch.bfloat16)


class _Layer:
    def __init__(self, name: str, module, weight: torch.nn.Parameter):
        self.name, self.module, self.weight = name, module, weight
        self.tensor_id = stable_tensor_id(name)
        self.out_f, self.in_f = (int(d) for d in weight.shape)
        self.device = weight.device
        self.master: Optional[torch.Tensor] = None  # fp32 centre on CPU (master='cpu')

    def gen(self, seed: int, salt: int = 0) -> torch.Generator:
        g = torch.Generator(device=self.device)
        g.manual_seed((int(seed) * 1000003 + salt ^ self.tensor_id) & _MASK63)
        return g

    def signs(self, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """(s (in,), r (out,)) of +-1, fp32, from one stream per (member seed, tensor)."""
        g = self.gen(seed, 1)
        bits = torch.randint(0, 2, (self.in_f + self.out_f,), generator=g, device=self.device)
        v = bits.float().mul_(2.0).sub_(1.0)
        return v[: self.in_f], v[self.in_f:]

    def core(self, seed: int, k: int) -> torch.Tensor:
        """c (k,) of +-1 for the ``core`` variant."""
        return torch.randint(0, 2, (k,), generator=self.gen(seed, 4), device=self.device).float().mul_(2.0).sub_(1.0)


class _Vec:
    """A vector parameter (norm gain or bias) with one Gaussian direction per member."""

    def __init__(self, name: str, module, param: torch.Tensor, kind: str):
        self.name, self.module, self.param, self.kind = name, module, param, kind
        self.tensor_id = stable_tensor_id(f"{name}#{kind}")
        self.device = param.device
        self.master = param.data.float().clone()

    def noise(self, seed: int) -> torch.Tensor:
        g = torch.Generator(device=self.device)
        g.manual_seed((int(seed) * 1000003 + 3 ^ self.tensor_id) & _MASK63)
        return torch.randn(self.param.shape, generator=g, dtype=torch.float32, device=self.device)

    @property
    def table(self) -> torch.Tensor:
        return self.module.flip_D if self.kind == "norm" else self.module.flip_b

    def fill(self, k: int, e: torch.Tensor, scale: float) -> None:
        if self.kind == "norm":  # relative perturbation of the gain; a zero gain stays zero
            self.table[k].copy_(torch.where(self.master.abs() > 1e-8, e * scale / self.master, torch.zeros_like(e)))
        else:
            self.table[k].copy_(e * scale)


def _norm_forward(module, original):
    def forward(x, residual=None):
        out = original(x, residual)
        h = out if residual is None else out[0]
        d = module.flip_D.index_select(0, module.flip_slots[: h.reshape(-1, h.shape[-1]).shape[0]]).reshape(h.shape)
        h = torch.addcmul(h, h, d)
        return h if residual is None else (h, out[1])

    return forward


class _FlipoutVocab:
    """``quant_method`` of the embedding and / or the LM head (one module when the weights are tied).

    Weight (vocab, hidden), perturbation ``sigma * E o (r_n s_n^T)`` with r over the vocabulary and s over hidden:
    lookup   out[t] += sigma * r_n[token] * (E[token] o s_n)                     (inside the captured graph)
    head     logits += sigma * r_n o (E (s_n o h))    on the sampled rows, after the graph; slots from the owner
    """

    def __init__(self, original, owner: "DenseFlipout", embed: bool, head: bool):
        self.original, self.owner, self.embed, self.head = original, owner, embed, head

    def embedding(self, module, ids: torch.Tensor) -> torch.Tensor:
        y = self.original.embedding(module, ids)
        if not self.embed:
            return y
        flat = ids.reshape(-1)
        slots = module.flip_slots[: flat.shape[0]]
        if self.owner.is_core:
            return y + ((F.embedding(flat, module.flip_A) * module.flip_C.index_select(0, slots)) @ module.flip_B).reshape(y.shape)
        e = F.embedding(flat, module.flip_A) @ module.flip_B if self.owner.rank else F.embedding(flat, module.flip_E)
        d = e * module.flip_S.index_select(0, slots) * module.flip_R[slots, flat].unsqueeze(1)
        return y + d.reshape(y.shape)

    def apply(self, module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.original.apply(module, x, bias)
        slots = self.owner.logit_slots(x.shape[0]) if self.head else None
        if slots is None:
            return y
        if self.owner.is_core:
            return y.add_(F.linear(F.linear(x, module.flip_B) * module.flip_C.index_select(0, slots), module.flip_A).to(y.dtype))
        h = x * module.flip_S.index_select(0, slots)
        d = F.linear(F.linear(h, module.flip_B), module.flip_A) if self.owner.rank else F.linear(h, module.flip_E)
        return y.addcmul_(d.to(y.dtype), module.flip_R.index_select(0, slots).to(y.dtype))  # y is a fresh tensor, outside the graph

    def __getattr__(self, name: str):
        return getattr(self.original, name)


class _FlipoutMethod:
    """Stand-in for a vLLM linear ``quant_method``: base kernel plus the Flipout term."""

    def __init__(self, original, owner: "DenseFlipout", slot_bias: bool = False):
        self.original, self.owner, self.slot_bias = original, owner, slot_bias

    def apply(self, module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.original.apply(module, x, bias)
        owner = self.owner
        if not owner.active and not owner.graph_safe:
            return y
        out = self._term(module, x, y)
        if self.slot_bias:  # per-member bias direction (extras: bias)
            out = out + module.flip_b.index_select(0, module.flip_slots[: x.reshape(-1, x.shape[-1]).shape[0]]).reshape(out.shape)
        return out

    def _term(self, module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        owner = self.owner
        v = owner.variant  # fixed at construction, so the branch is static under CUDA graphs
        if v == "none":  # diagnostic: only the extras are perturbed
            return y
        x2 = x.reshape(-1, x.shape[-1])
        slots = module.flip_slots[: x2.shape[0]]
        if v == "core":  # dW = sigma * A diag(c) B: the member's signs act inside the rank-k bottleneck
            return y + F.linear(F.linear(x2, module.flip_B) * module.flip_C.index_select(0, slots), module.flip_A).reshape(y.shape)
        if v == "core2":  # same maths; the second thin GEMM accumulates into the base output (no separate add pass)
            z = F.linear(x2, module.flip_B) * module.flip_C.index_select(0, slots)
            return torch.addmm(y.reshape(-1, y.shape[-1]), z, module.flip_A.t()).reshape(y.shape)
        if v == "lowrank":  # dW = sigma * diag(r) A B diag(s): two thin GEMMs
            d = F.linear(F.linear(x2 * module.flip_S.index_select(0, slots), module.flip_B), module.flip_A)
            return torch.addcmul(y, d.reshape(y.shape), module.flip_R.index_select(0, slots).reshape(y.shape))
        if v == "exact":  # reference implementation
            s = module.flip_S.index_select(0, slots)
            r = module.flip_R.index_select(0, slots)
            d = F.linear(x2 * s, module.flip_E) * r
            return y + d.reshape(y.shape)
        if v == "fused":  # same maths; one fused multiply-add on the output side
            d = F.linear(x2 * module.flip_S.index_select(0, slots), module.flip_E)
            return torch.addcmul(y, d.reshape(y.shape), module.flip_R.index_select(0, slots).reshape(y.shape))
        if v == "out_only":  # dW = sigma * diag(r) E: output signs only (different perturbation family)
            d = F.linear(x2, module.flip_E)
            return torch.addcmul(y, d.reshape(y.shape), module.flip_R.index_select(0, slots).reshape(y.shape))
        if v == "gemm_only":  # diagnostic: two GEMMs and no sign work (not a valid population)
            return y + F.linear(x2, module.flip_E).reshape(y.shape)
        raise ValueError(v)

    def __getattr__(self, name: str):
        return getattr(self.original, name)


class DenseFlipout:
    route_all = False

    def __init__(self, model, *, n_slots: int, max_tokens: int, targets: Sequence[str] = MLP_SUFFIXES + ATTN_SUFFIXES,
                 master: str = "cpu", graph_safe: bool = True, variant: str = "exact", rank: int = 0, noise_dtype: torch.dtype = torch.bfloat16,
                 extras: Sequence[str] = ()):
        if master not in ("cpu", "sr"):
            raise ValueError(master)
        self.n_slots, self.master_mode, self.graph_safe = int(n_slots), master, bool(graph_safe)
        self.active = False
        if variant in ("core", "core2") and not rank:
            raise ValueError("variant 'core' needs rank > 0")
        self.variant = (variant if variant in ("core", "core2") else "lowrank") if rank else variant
        self.is_core = self.variant in ("core", "core2")
        self.rank = int(rank)
        self.gen_seed: Optional[int] = None
        self.sigma = 0.0
        self.entries: List[Tuple[int, int]] = []
        self.layers: List[_Layer] = []
        self.vectors: List[_Vec] = []
        self.timing: Dict[str, float] = {}
        self.extras = tuple(extras)
        if set(self.extras) - set(EXTRAS):
            raise ValueError(extras)
        self.runner = None  # set by the engine: the head needs the batch's request -> row map
        self.last_slots: Optional[torch.Tensor] = None
        self.row_slots: Optional[torch.Tensor] = None
        self.head_on = "head" in self.extras
        self.logit_override: Optional[torch.Tensor] = None
        self._pre = True
        dev = next(model.parameters()).device
        tied = getattr(model, "lm_head", None) is getattr(getattr(model, "model", None), "embed_tokens", object())
        self.slot_buf = torch.zeros(int(max_tokens), dtype=torch.int64, device=dev)
        with torch.no_grad():
            for name, module in model.named_modules():
                leaf = name.split(".")[-1]
                method = getattr(module, "quant_method", None)
                kind = None
                if method is not None and leaf in targets and not any(k in name for k in SKIP_KEYWORDS) and type(method).__name__ == "UnquantizedLinearMethod":
                    kind = "linear"
                elif method is not None and leaf == "embed_tokens" and ("embed" in self.extras or (tied and "head" in self.extras)):
                    kind = "tied" if tied else "embed"  # tied weights: one matrix, so the lookup and the head are perturbed together
                elif method is not None and leaf == "lm_head" and "head" in self.extras:
                    kind = "head"
                elif type(module).__name__ == "RMSNorm" and "norm" in self.extras and isinstance(getattr(module, "weight", None), torch.nn.Parameter):
                    module.register_buffer("flip_D", torch.zeros(self.n_slots + 1, module.weight.shape[0], dtype=module.weight.dtype, device=dev), persistent=False)
                    module.register_buffer("flip_slots", self.slot_buf, persistent=False)
                    module.forward = _norm_forward(module, module.forward)
                    self.vectors.append(_Vec(name, module, module.weight, "norm"))
                if kind is None:
                    continue
                w = module.weight
                if w.ndim != 2 or w.dtype != torch.bfloat16:
                    continue
                layer = _Layer(name, module, w)
                layer.kind = kind
                if self.rank:  # E = A B with A (out, k) ~ N(0, 1/k), B (k, in) ~ N(0, 1): identity covariance, rank k
                    module.register_buffer("flip_A", torch.zeros(layer.out_f, self.rank, dtype=noise_dtype, device=w.device), persistent=False)
                    module.register_buffer("flip_B", torch.zeros(self.rank, layer.in_f, dtype=noise_dtype, device=w.device), persistent=False)
                else:
                    module.register_buffer("flip_E", torch.zeros(w.shape, dtype=noise_dtype, device=w.device), persistent=False)
                if self.is_core:
                    module.register_buffer("flip_C", torch.zeros(self.n_slots + 1, self.rank, dtype=w.dtype, device=w.device), persistent=False)
                else:
                    module.register_buffer("flip_S", torch.zeros(self.n_slots + 1, layer.in_f, dtype=w.dtype, device=w.device), persistent=False)
                    module.register_buffer("flip_R", torch.zeros(self.n_slots + 1, layer.out_f, dtype=w.dtype, device=w.device), persistent=False)
                module.register_buffer("flip_slots", self.slot_buf, persistent=False)
                if master == "cpu":
                    layer.master = w.data.float().cpu()
                bias = getattr(module, "bias", None) if kind == "linear" and "bias" in self.extras else None
                if bias is not None:
                    module.register_buffer("flip_b", torch.zeros(self.n_slots + 1, layer.out_f, dtype=w.dtype, device=w.device), persistent=False)
                    self.vectors.append(_Vec(name, module, bias, "bias"))
                module.quant_method = _FlipoutMethod(method, self, slot_bias=bias is not None) if kind == "linear" else \
                    _FlipoutVocab(method, self, embed=kind != "head", head=kind != "embed")
                self.layers.append(layer)
        if not self.layers and not self.vectors:
            raise RuntimeError("nothing to perturb")

    # ------------------------------------------------------------------ routing (SlotEngine protocol)
    def set_token_slots(self, slots: Optional[torch.Tensor]) -> None:
        """Called by the engine right before a forward pass (with the batch's slots) and right after it (with None)."""
        self.slot_buf.zero_()
        use = slots is not None and self.active
        if self._pre:
            self.last_slots = slots.clone() if use else None
            self.row_slots = None
            if use and self.runner is not None and self.head_on:
                # Rows the LM head will see: the last scheduled token of every request. Built here, while the GPU is
                # idle; a host-to-device copy inside the head would wait for the whole forward pass and cost the
                # engine its CPU/GPU overlap (measured: 6 ms per engine step).
                n = len(self.runner.input_batch.req_ids)
                idx = torch.as_tensor(self.runner.query_start_loc.np[1: n + 1] - 1, device=slots.device, dtype=torch.int64)
                self.row_slots = slots[idx]
        self._pre = not self._pre
        if use:
            self.slot_buf[: slots.shape[0]].copy_(slots)

    def logit_slots(self, rows: int) -> Optional[torch.Tensor]:
        """Slots of the rows handed to the LM head: one row per request of the last batch (its last scheduled token)."""
        if self.logit_override is not None:
            return self.logit_override
        if not self.active or self.row_slots is None:
            return None
        if rows != self.row_slots.shape[0]:
            raise RuntimeError(f"LM head got {rows} rows for {self.row_slots.shape[0]} requests; pass the rows' slots through logit_override")
        return self.row_slots

    # ------------------------------------------------------------------ population
    @torch.no_grad()
    def set_noise(self, gen_seed: int) -> None:
        """Base noise of a generation: E = randn(generator(gen_seed, tensor)), stored in the noise dtype."""
        t0 = time.perf_counter()
        for layer in self.layers:
            g = layer.gen(gen_seed, 0)
            m = layer.module
            if self.rank:
                m.flip_A.copy_(torch.randn(m.flip_A.shape, generator=g, dtype=torch.float32, device=layer.device) / self.rank ** 0.5)
                m.flip_B.copy_(torch.randn(m.flip_B.shape, generator=g, dtype=torch.float32, device=layer.device))
                continue
            e = m.flip_E
            rows = max(1, (1 << 26) // layer.in_f)
            for lo in range(0, layer.out_f, rows):
                hi = min(layer.out_f, lo + rows)
                e[lo:hi].copy_(torch.randn((hi - lo, layer.in_f), generator=g, dtype=torch.float32, device=layer.device))
        self.gen_seed = int(gen_seed)
        torch.cuda.synchronize()
        self.timing["set_noise_s"] = time.perf_counter() - t0

    @torch.no_grad()
    def set_population(self, gen_seed: int, entries: Sequence[Tuple[int, int]], sigma: float) -> None:
        """Slot ``k + 1`` = member ``entries[k] = (member_seed, sign)``, sign +1 / -1 (antithetic) / 0 (centre)."""
        if len(entries) > self.n_slots:
            raise ValueError(f"{len(entries)} members > {self.n_slots} slots")
        if self.gen_seed != int(gen_seed):
            self.set_noise(gen_seed)
        t0 = time.perf_counter()
        for layer in self.layers:
            if self.is_core:
                C = layer.module.flip_C
                C.zero_()
                cc: Dict[int, torch.Tensor] = {}
                for k, (seed, sign) in enumerate(entries, 1):
                    if sign:
                        if seed not in cc:
                            cc[seed] = layer.core(seed, self.rank)
                        C[k].copy_(cc[seed] * (float(sign) * float(sigma)))
                continue
            S, R = layer.module.flip_S, layer.module.flip_R
            S.zero_(); R.zero_()
            cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
            for k, (seed, sign) in enumerate(entries, 1):
                if not sign:
                    continue
                if seed not in cache:
                    cache[seed] = layer.signs(seed)
                s, r = cache[seed]
                S[k].copy_(s)
                R[k].copy_(r * (float(sign) * float(sigma)))
        for vec in self.vectors:
            vec.table.zero_()
            cache = {}
            for k, (seed, sign) in enumerate(entries, 1):
                if sign:
                    if seed not in cache:
                        cache[seed] = vec.noise(seed)
                    vec.fill(k, cache[seed], float(sign) * float(sigma))
        self.entries, self.sigma, self.active = [(int(a), int(b)) for a, b in entries], float(sigma), True
        torch.cuda.synchronize()
        self.timing["set_tables_s"] = time.perf_counter() - t0

    @torch.no_grad()
    def clear_population(self) -> None:
        for layer in self.layers:
            if self.is_core:
                layer.module.flip_C.zero_()
            else:
                layer.module.flip_S.zero_(); layer.module.flip_R.zero_()
        for vec in self.vectors:
            vec.table.zero_()
        self.entries, self.active, self.last_slots, self.row_slots = [], False, None, None
        self.slot_buf.zero_()

    def base_noise(self, layer: _Layer) -> torch.Tensor:
        """The fp32 (out, in) base noise the layer applies (as stored, i.e. after rounding to the noise dtype)."""
        m = layer.module
        return m.flip_A.float() @ m.flip_B.float() if self.rank else m.flip_E.float()

    # ------------------------------------------------------------------ update
    def base_rows(self, layer: _Layer, lo: int, hi: int) -> torch.Tensor:
        m = layer.module
        return m.flip_A[lo:hi].float() @ m.flip_B.float() if self.rank else m.flip_E[lo:hi].float()

    @torch.no_grad()
    def direction(self, layer: _Layer, seeds: Sequence[int], weights: Sequence[float], lo: int = 0, hi: Optional[int] = None,
                  _cache: Optional[dict] = None) -> torch.Tensor:
        """Rows ``lo:hi`` of  sum_n w_n eps_n = E o (R^T diag(w) S)  for the current base noise (fp32, on device)."""
        hi = layer.out_f if hi is None else hi
        key = id(layer)
        if _cache is None or _cache.get("key") != key:
            acc: Dict[int, float] = {}
            for s, w in zip(seeds, weights):
                acc[int(s)] = acc.get(int(s), 0.0) + float(w)
            acc = {s: w for s, w in acc.items() if w}
            if self.is_core:  # sum_n w_n A diag(c_n) B = A diag(v) B
                v = torch.zeros(self.rank, device=layer.device)
                for seed, w in acc.items():
                    v.add_(layer.core(seed, self.rank), alpha=w)
                if _cache is not None:
                    _cache.update(key=key, S=v, R=None)
                m = layer.module
                return (m.flip_A[lo:hi].float() * v) @ m.flip_B.float()
            S = torch.empty(len(acc), layer.in_f, device=layer.device)
            R = torch.empty(len(acc), layer.out_f, device=layer.device)
            for i, (seed, w) in enumerate(acc.items()):
                s, r = layer.signs(seed)
                S[i], R[i] = s, r * w
            if _cache is not None:
                _cache.update(key=key, S=S, R=R)
        else:
            S, R = _cache["S"], _cache["R"]
        if self.is_core:
            m = layer.module
            return (m.flip_A[lo:hi].float() * S) @ m.flip_B.float()
        if S.shape[0] == 0:
            return torch.zeros((hi - lo, layer.in_f), device=layer.device)
        return (R[:, lo:hi].t() @ S).mul_(self.base_rows(layer, lo, hi))

    @torch.no_grad()
    def update(self, seeds: Sequence[int], weights: Sequence[float], step: float) -> Dict[str, float]:
        """centre <- centre + step * sum_n w_n eps_n (``step`` = alpha / N; base noise of the last set_noise).

        Row chunks of at most 2^26 elements keep the fp32 temporaries near 1.5 GB whatever the layer size."""
        if self.gen_seed is None:
            raise RuntimeError("no base noise set")
        t0 = time.perf_counter()
        sq = 0.0
        for layer in self.layers:
            rows = max(1, (1 << 26) // layer.in_f)
            cache: dict = {}
            rg = layer.gen(self.gen_seed, 2)
            for lo in range(0, layer.out_f, rows):
                hi = min(layer.out_f, lo + rows)
                d = self.direction(layer, seeds, weights, lo, hi, cache).mul_(float(step))
                sq += float(d.pow(2).sum())
                if layer.master is not None:
                    d.add_(layer.master[lo:hi].to(layer.device, non_blocking=True))
                    layer.master[lo:hi].copy_(d)
                    layer.weight.data[lo:hi].copy_(d)
                else:
                    d.add_(layer.weight.data[lo:hi].float())
                    layer.weight.data[lo:hi].copy_(stochastic_round(d, rg))
            del cache
        acc: Dict[int, float] = {}
        for s, w in zip(seeds, weights):
            acc[int(s)] = acc.get(int(s), 0.0) + float(w)
        for vec in self.vectors:
            d = torch.zeros_like(vec.master)
            for seed, w in acc.items():
                if w:
                    d.add_(vec.noise(seed), alpha=w * float(step))
            sq += float(d.pow(2).sum())
            vec.master.add_(d)
            vec.param.data.copy_(vec.master)
        torch.cuda.synchronize()
        self.timing["update_s"] = time.perf_counter() - t0
        return {"update_norm": sq ** 0.5, "update_s": self.timing["update_s"]}

    # ------------------------------------------------------------------ state / diagnostics
    def state(self) -> List[torch.Tensor]:
        return [(l.master.clone() if l.master is not None else l.weight.data.cpu().clone()) for l in self.layers] + [v.master.cpu().clone() for v in self.vectors]

    @torch.no_grad()
    def load_state(self, state: Sequence[torch.Tensor]) -> None:
        for l, t in zip(self.layers, state):
            if l.master is not None:
                l.master.copy_(t.float())
            l.weight.data.copy_(t.to(l.device))
        for v, t in zip(self.vectors, state[len(self.layers):]):
            v.master.copy_(t.to(v.device))
            v.param.data.copy_(v.master)

    @torch.no_grad()
    def materialize(self, slot: int) -> None:
        """Test reference: write member ``slot`` into the model's own parameters (undo with ``restore``)."""
        self._backup = [l.weight.data.clone() for l in self.layers] + [v.param.data.clone() for v in self.vectors]
        for li, l in enumerate(self.layers):
            l.weight.data.copy_(self.member_weight(li, slot))
        for v in self.vectors:
            t = v.table[slot].float()
            v.param.data.copy_(v.master * (1 + t) if v.kind == "norm" else v.master + t)

    @torch.no_grad()
    def restore(self) -> None:
        for p, b in zip([l.weight for l in self.layers] + [v.param for v in self.vectors], self._backup):
            p.data.copy_(b)

    @torch.no_grad()
    def member_weight(self, li: int, slot: int) -> torch.Tensor:
        """Reference: the fp32 dense weight slot ``slot`` applies, W + E o (R_slot S_slot^T)."""
        layer = self.layers[li]
        m = layer.module
        if self.is_core:
            return layer.weight.data.float() + (m.flip_A.float() * m.flip_C[slot].float()) @ m.flip_B.float()
        return layer.weight.data.float() + self.base_noise(layer) * torch.outer(m.flip_R[slot].float(), m.flip_S[slot].float())

    @property
    def n_params(self) -> int:
        return int(sum(l.out_f * l.in_f for l in self.layers) + sum(v.param.numel() for v in self.vectors))

    def info(self) -> Dict[str, Any]:
        e = sum((l.module.flip_A.numel() + l.module.flip_B.numel()) * 2 if self.rank else l.module.flip_E.numel() * l.module.flip_E.element_size() for l in self.layers)
        t = sum((self.rank if self.is_core else l.in_f + l.out_f) * 2 * (self.n_slots + 1) for l in self.layers)
        return {"n_layers": len(self.layers), "n_params": self.n_params, "noise_gb": e / 1e9, "tables_gb": t / 1e9,
                "master": self.master_mode, "master_cpu_gb": sum(l.out_f * l.in_f for l in self.layers) * 4 / 1e9 if self.master_mode == "cpu" else 0.0,
                "n_slots": self.n_slots, "variant": self.variant, "rank": self.rank, "extras": list(self.extras),
                "n_vectors": len(self.vectors), "n_vector_params": int(sum(v.param.numel() for v in self.vectors)), **self.timing}
