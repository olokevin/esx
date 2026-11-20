"""Inference-native FuRA-S for vLLM: the quantized base GEMM is never touched.

``fura_s`` factors every input block of a linear weight as
``W[:, blk_j] = U_j diag(S0_j) Vh_j`` and trains offsets ``c_j`` of the singular
values (``structured_es.py``). With the frames frozen,

    W'_j = W0_j (I + Vh_j^T diag(c_j / S0_j) Vh_j),

so the layer keeps applying its original (here INT4 Marlin) kernel to a
pre-transformed activation ``x + dx`` with ``dx_j = Vh_j^T diag(r_j) Vh_j x_j`` and
``r = c / S0`` (``fura_s_pretransform.py``). This module installs that
pre-transform into a vLLM model and adds *slots*: every token carries a slot id
and slot ``s`` applies its own ratio vector ``r_s``. A slot is an adapter state
plus an optional ES perturbation (``c_user + sign * sigma * eps(seed)``), so one
forward pass can mix every population member of every user (the activation-space
trick of Flipout). Slot 0 is the unmodified base model: its ratio is exactly zero,
so its tokens see bit-identical base activations.

Parameterization options (RQ4 ablations):
* ``targets``     which module suffixes get a state (``all`` / ``mlp`` / ``attn`` or a list);
* ``rank``        train only the top-``rank`` singular values of every block (0 = all);
* ``subspace``    ``svd`` (the FuRA spectral frame) or ``random`` (a seeded random
                  orthogonal frame, ratio scale ``1 / rms(S0_j)``).

The noise of a coefficient tensor is ``randn((n_blk, b), generator(seed ^ tensor_id))``,
the same draw ``StructuredES`` / ``FuraSPretransform`` use for scope ``fura_s``.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from algorithms.es.structured_es import closest_factor_pair

SKIP_KEYWORDS = ("embed_tokens", "lm_head", "visual", "mtp", "in_proj_ba", "in_proj_a", "in_proj_b")
MLP_SUFFIXES = ("gate_up_proj", "down_proj")
ATTN_SUFFIXES = ("qkv_proj", "o_proj", "in_proj_qkvz", "in_proj_qkv", "in_proj_z", "out_proj")
_MASK64 = 0xFFFFFFFFFFFFFFFF


def stable_tensor_id(name: str) -> int:
    digest = hashlib.blake2b(str(name).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _suffixes(targets) -> Tuple[str, ...]:
    if targets in (None, "all"):
        return MLP_SUFFIXES + ATTN_SUFFIXES
    if targets == "mlp":
        return MLP_SUFFIXES
    if targets == "attn":
        return ATTN_SUFFIXES
    return tuple(targets)


@torch.no_grad()
def probe_weight(module, chunk: int = 2048) -> torch.Tensor:
    """Effective (out, in) fp32 weight a vLLM linear applies, read through its own kernel."""
    method = module.quant_method
    in_f = int(getattr(module, "input_size_per_partition", None) or module.input_size)
    param = next(p for p in module.parameters())
    device = param.device
    dtype = param.dtype if param.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64) else torch.bfloat16
    cols = []
    for lo in range(0, in_f, chunk):
        hi = min(in_f, lo + chunk)
        eye = torch.zeros(hi - lo, in_f, dtype=dtype, device=device)
        eye[:, lo:hi] = torch.eye(hi - lo, dtype=dtype, device=device)
        cols.append(method.apply(module, eye, None).float())  # (hi - lo, out) = W[:, lo:hi]^T
    return torch.cat(cols, dim=0).t().contiguous()


class _Layer:
    def __init__(self, name: str, module, weight: torch.Tensor, *, rank: int, subspace: str,
                 core_dtype: torch.dtype, cached: Optional[Dict[str, torch.Tensor]] = None):
        self.name = name
        self.tensor_id = stable_tensor_id(name)
        self.module = module
        out_f, in_f = weight.shape
        self.out_f, self.in_f = int(out_f), int(in_f)
        self.n_blk, self.b = closest_factor_pair(self.in_f)
        self.device = weight.device
        if cached is not None:
            s, vh = cached["S0"].to(self.device), cached["Vh"].to(self.device)
        else:
            w = weight.float().reshape(out_f, self.n_blk, self.b).permute(1, 0, 2)
            _, s, vh = torch.linalg.svd(w, full_matrices=False)
            del w
        self.svd = {"S0": s.cpu(), "Vh": vh.cpu()}
        if subspace == "random":
            gen = torch.Generator(device="cpu").manual_seed(self.tensor_id & 0x7FFFFFFFFFFFFFFF)
            q, _ = torch.linalg.qr(torch.randn(self.n_blk, self.b, self.b, generator=gen))
            vh = q.transpose(1, 2).to(self.device)
            rms = s.pow(2).mean(dim=-1, keepdim=True).sqrt()
            inv_s = (1.0 / rms).expand_as(s).contiguous()
        elif subspace == "svd":
            floor = s.amax(dim=-1, keepdim=True) * 1e-3
            inv_s = 1.0 / torch.maximum(s, floor)
        else:
            raise ValueError(subspace)
        self.vh = vh.to(core_dtype).contiguous()  # (n_blk, b, b): rows are the right singular vectors
        self.vh_t = self.vh.transpose(1, 2).contiguous()
        self.inv_s = inv_s.float().contiguous()  # (n_blk, b)
        self.mask = None
        if rank and rank < self.b:
            self.mask = torch.zeros(self.n_blk, self.b, device=self.device)
            self.mask[:, :rank] = 1.0  # singular values are sorted, so the top-r are the first r
        self.table: Optional[torch.Tensor] = None  # (n_slots, n_blk, b) ratio, activation dtype

    @property
    def n_params(self) -> int:
        return int(self.n_blk * self.b if self.mask is None else self.mask.sum().item())

    def noise(self, seed: int) -> torch.Tensor:
        gen = torch.Generator(device=self.device)
        gen.manual_seed((int(seed) ^ self.tensor_id) & _MASK64)
        eps = torch.randn((self.n_blk, self.b), generator=gen, dtype=torch.float32, device=self.device)
        return eps if self.mask is None else eps * self.mask

    def noise_block(self, seed: int, n: int) -> torch.Tensor:
        """``n`` directions ``(n, n_blk, b)`` from one stream ``generator(seed ^ tensor_id)``;
        direction ``k`` of block ``seed`` is row ``k`` (used for per-sequence perturbations)."""
        gen = torch.Generator(device=self.device)
        gen.manual_seed((int(seed) ^ self.tensor_id) & _MASK64)
        eps = torch.randn((int(n), self.n_blk, self.b), generator=gen, dtype=torch.float32, device=self.device)
        return eps if self.mask is None else eps * self.mask

    def delta(self, x: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        tokens = x.shape[0]
        xb = x.view(tokens, self.n_blk, self.b).transpose(0, 1)  # (n_blk, T, b)
        z = torch.bmm(xb, self.vh_t.to(x.dtype))  # (n_blk, T, b) = Vh x
        ratio = self.table.index_select(0, slots).transpose(0, 1)  # (n_blk, T, b)
        dx = torch.bmm(z * ratio, self.vh.to(x.dtype))  # Vh^T diag(r) Vh x
        return dx.transpose(0, 1).reshape(tokens, self.in_f)


class _PreTransformMethod:
    """Stand-in for a vLLM ``quant_method``: base kernel on ``x + dx``."""

    def __init__(self, original, owner: "FuraNative", layer: _Layer):
        self.original = original
        self.owner = owner
        self.layer = layer

    def apply(self, module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.owner.graph_safe:
            # Static module buffers (graph inputs, never constants) and no data-dependent
            # branch: capturable by torch.compile and CUDA graphs.
            lead = x.shape[:-1]
            x2 = x.reshape(-1, x.shape[-1])
            layer = self.layer
            tokens = x2.shape[0]
            slots = module.fura_slots[:tokens]
            xb = x2.view(tokens, layer.n_blk, layer.b).transpose(0, 1)
            z = torch.bmm(xb, module.fura_vh_t)
            ratio = module.fura_table.index_select(0, slots).transpose(0, 1)
            dx = torch.bmm(z * ratio, module.fura_vh).transpose(0, 1).reshape(tokens, layer.in_f)
            x = (x2 + dx).reshape(*lead, x.shape[-1])
            return self.original.apply(module, x, bias)
        slots = self.owner.token_slots(x.shape[0])
        if slots is not None and self.layer.table is not None:
            lead = x.shape[:-1]
            x2 = x.reshape(-1, x.shape[-1])
            x = (x2 + self.layer.delta(x2, slots)).reshape(*lead, x.shape[-1])
        return self.original.apply(module, x, bias)

    def __getattr__(self, name: str):
        return getattr(self.original, name)


class FuraNative:
    """FuRA-S states for many adapters (users) and slot-routed perturbations."""

    def __init__(self, model, *, targets="all", rank: int = 0, subspace: str = "svd",
                 core_dtype: torch.dtype = torch.bfloat16, svd_cache: Optional[str] = None,
                 graph_slots: int = 0, graph_tokens: int = 0):
        """``graph_slots > 0``: graph-safe mode for CUDA graphs / torch.compile. Ratio tables
        (``graph_slots + 1`` rows) and the per-token slot buffer (``graph_tokens``) are
        allocated once and only written in place; the pre-transform always runs."""
        self.graph_safe = graph_slots > 0
        self.graph_slots = int(graph_slots)
        self.targets = targets
        self.rank = int(rank)
        self.subspace = subspace
        self.layers: List[_Layer] = []
        self.adapters: Dict[str, List[torch.Tensor]] = {}  # uid -> per-layer fp32 coef
        self.slot_keys: List[Any] = []
        self._slots: Optional[torch.Tensor] = None
        self.enabled = True
        suffixes = _suffixes(targets)
        cache = torch.load(svd_cache, map_location="cpu") if svd_cache and os.path.exists(svd_cache) else None
        new_cache = {}
        with torch.no_grad():
            for name, module in model.named_modules():
                if not hasattr(module, "quant_method") or module.quant_method is None:
                    continue
                if not hasattr(module.quant_method, "apply") or any(k in name for k in SKIP_KEYWORDS):
                    continue
                if not (hasattr(module, "input_size") and hasattr(module, "output_size")):
                    continue
                if name.split(".")[-1] not in suffixes:
                    continue
                cached = None if cache is None else cache.get(name)
                weight = None
                if cached is None:
                    weight = probe_weight(module)
                    shape = weight.shape
                else:
                    shape = (int(cached["out"]), int(cached["in"]))
                n_blk, b = closest_factor_pair(int(shape[1]))
                if shape[0] < b:
                    continue
                if weight is None:
                    weight = torch.empty(shape, device=next(module.parameters()).device)
                layer = _Layer(name, module, weight, rank=self.rank, subspace=subspace,
                               core_dtype=core_dtype, cached=cached)
                new_cache[name] = {**layer.svd, "out": shape[0], "in": shape[1]}
                layer.svd = None
                del weight
                module.quant_method = _PreTransformMethod(module.quant_method, self, layer)
                self.layers.append(layer)
        if not self.layers:
            raise RuntimeError("No linear layers usable for FuRA-S.")
        if self.graph_safe:
            device = self.layers[0].device
            self.slot_buf = torch.zeros(int(graph_tokens), dtype=torch.int64, device=device)
            for layer in self.layers:
                layer.table = torch.zeros(self.graph_slots + 1, layer.n_blk, layer.b, dtype=torch.bfloat16,
                                          device=device)
                # Buffers of the linear itself, so the compiler lifts them as graph inputs.
                layer.module.register_buffer("fura_table", layer.table, persistent=False)
                layer.module.register_buffer("fura_vh", layer.vh.to(torch.bfloat16), persistent=False)
                layer.module.register_buffer("fura_vh_t", layer.vh_t.to(torch.bfloat16), persistent=False)
                layer.module.register_buffer("fura_slots", self.slot_buf, persistent=False)
        if svd_cache and cache is None:
            os.makedirs(os.path.dirname(os.path.abspath(svd_cache)), exist_ok=True)
            torch.save(new_cache, svd_cache)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ routing
    def set_token_slots(self, slots: Optional[torch.Tensor]) -> None:
        if self.graph_safe:
            # Padded / unrouted rows read slot 0, whose ratio row is all zeros.
            self.slot_buf.zero_()
            if slots is not None and self.enabled:
                self.slot_buf[: slots.shape[0]].copy_(slots)
            return
        self._slots = slots

    def token_slots(self, tokens: int) -> Optional[torch.Tensor]:
        slots = self._slots
        if slots is None or not self.enabled:
            return None
        if slots.shape[0] == tokens:
            return slots
        if slots.shape[0] < tokens:
            return F.pad(slots, (0, tokens - slots.shape[0]))
        return slots[:tokens]

    # ------------------------------------------------------------------ adapters
    def new_adapter(self, uid: str) -> None:
        self.adapters[uid] = [torch.zeros(l.n_blk, l.b, device=l.device) for l in self.layers]

    def drop_adapter(self, uid: str) -> None:
        self.adapters.pop(uid, None)

    def adapter_state(self, uid: str) -> List[torch.Tensor]:
        return [c.detach().cpu().clone() for c in self.adapters[uid]]

    def load_adapter_state(self, uid: str, state: Sequence[torch.Tensor]) -> None:
        self.adapters[uid] = [c.to(l.device).float().clone() for c, l in zip(state, self.layers)]

    @torch.no_grad()
    def set_slots(self, entries: Sequence[Tuple[Optional[str], int, float]]) -> None:
        """Slot ``i + 1`` = adapter ``uid`` (None = base) plus ``sigma * eps(seed)``.

        ``entries`` items: ``(uid, seed, signed_sigma)``; ``signed_sigma = 0`` means
        the adapter itself (evaluation)."""
        self.slot_keys = [None] + list(entries)
        for li, layer in enumerate(self.layers):
            rows = [torch.zeros(layer.n_blk, layer.b, device=layer.device)]
            noise_cache: Dict[int, torch.Tensor] = {}
            for uid, seed, sigma in entries:
                coef = self.adapters[uid][li] if uid is not None else torch.zeros_like(rows[0])
                if sigma:
                    if seed not in noise_cache:
                        noise_cache[seed] = layer.noise(seed)
                    coef = coef + float(sigma) * noise_cache[seed]
                rows.append(coef)
            self._write_table(layer, (torch.stack(rows) * layer.inv_s).to(torch.bfloat16))

    @torch.no_grad()
    def set_slot_groups(self, groups: Sequence[Tuple[str, int, int, float]]) -> List[int]:
        """Antithetic per-direction slots in bulk (one direction per sequence or small group).

        ``groups`` items: ``(uid, block_seed, n_dirs, sigma)``. A group adds ``2 * n_dirs``
        slots ``c_uid + sigma * eps_k``, ``c_uid - sigma * eps_k`` for ``k = 0 .. n_dirs - 1``
        (in that order), ``eps = noise_block(block_seed, n_dirs)``. Returns the first slot id
        of every group."""
        firsts, nxt = [], 1
        for _, _, n, _ in groups:
            firsts.append(nxt)
            nxt += 2 * int(n)
        self.slot_keys = [None] + [(uid, seed, k, sign * sigma) for uid, seed, n, sigma in groups
                                   for k in range(int(n)) for sign in (1.0, -1.0)]
        for li, layer in enumerate(self.layers):
            rows = [torch.zeros(1, layer.n_blk, layer.b, device=layer.device)]
            for uid, seed, n, sigma in groups:
                eps = layer.noise_block(seed, n).mul_(float(sigma))
                coef = self.adapters[uid][li]
                rows.append(torch.stack((coef + eps, coef - eps), 1).reshape(2 * int(n), layer.n_blk, layer.b))
                del eps
            self._write_table(layer, (torch.cat(rows) * layer.inv_s).to(torch.bfloat16))
        return firsts

    def _write_table(self, layer: _Layer, table: torch.Tensor) -> None:
        if self.graph_safe:
            if table.shape[0] > layer.table.shape[0]:
                raise ValueError(f"{table.shape[0] - 1} slots > graph_slots={self.graph_slots}")
            layer.table[: table.shape[0]].copy_(table)
            layer.table[table.shape[0]:].zero_()
        else:
            layer.table = table

    def clear_slots(self) -> None:
        for layer in self.layers:
            if self.graph_safe:
                layer.table.zero_()
            else:
                layer.table = None
        self.slot_keys = []

    @torch.no_grad()
    def update(self, uid: str, seeds: Sequence[int], weights: Sequence[float], eta: float) -> None:
        """``c <- c + eta / N * sum_i w_i eps(seed_i)`` (``w`` already includes any 1/sigma)."""
        scale = float(eta) / float(len(seeds))
        for li, layer in enumerate(self.layers):
            total = torch.zeros(layer.n_blk, layer.b, device=layer.device)
            for seed, w in zip(seeds, weights):
                total.add_(layer.noise(seed), alpha=scale * float(w))
            self.adapters[uid][li].add_(total)

    @torch.no_grad()
    def update_block(self, uid: str, block_seed: int, weights: Sequence[float], eta: float) -> None:
        """``c <- c + eta / n * sum_k w_k eps_k`` with ``eps = noise_block(block_seed, n)``."""
        w = torch.as_tensor(list(weights), dtype=torch.float32)
        n = w.numel()
        for li, layer in enumerate(self.layers):
            eps = layer.noise_block(block_seed, n)
            self.adapters[uid][li].add_(torch.tensordot(w.to(layer.device), eps, dims=1), alpha=float(eta) / n)

    @torch.no_grad()
    def set_adapter_grad_step(self, uid: str, grads: Sequence[torch.Tensor], lr: float) -> None:
        for c, g in zip(self.adapters[uid], grads):
            c.add_(g.to(c.device), alpha=-float(lr))

    # ------------------------------------------------------------------ info
    def info(self) -> Dict[str, Any]:
        core = sum(l.vh.numel() * l.vh.element_size() for l in self.layers)
        return {
            "n_layers": len(self.layers),
            "n_params": int(sum(l.n_params for l in self.layers)),
            "core_bytes": int(core),
            "inv_s_bytes": int(sum(l.inv_s.numel() * 4 for l in self.layers)),
            "table_bytes_per_slot": int(sum(l.in_f * 2 for l in self.layers)),
            "targets": self.targets if isinstance(self.targets, str) else list(self.targets),
            "rank": self.rank,
            "subspace": self.subspace,
            "layer_names": [l.name for l in self.layers],
        }

    @torch.no_grad()
    def materialize(self, coef: Sequence[torch.Tensor], li: int, weight: torch.Tensor) -> torch.Tensor:
        """Reference: the dense weight ``W0 (I + Vh^T diag(c / S0) Vh)`` blockwise (fp32)."""
        layer = self.layers[li]
        vh = layer.vh.float()
        ratio = coef[li].float() * layer.inv_s
        t = torch.bmm(vh.transpose(1, 2) * ratio.unsqueeze(1), vh)  # (n_blk, b, b)
        w = weight.float().reshape(layer.out_f, layer.n_blk, layer.b).permute(1, 0, 2)
        return (w + torch.bmm(w, t)).permute(1, 0, 2).reshape(layer.out_f, layer.in_f)
