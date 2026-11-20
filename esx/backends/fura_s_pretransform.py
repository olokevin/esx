"""``fura_s`` as an input pre-transform on the frozen base weight.

``fura_s`` (``structured_es.py``) factors every input block of a weight as
``W[:, blk_j] = U_j diag(S0_j) Vh_j`` and trains offsets ``c_j`` on the singular
values. Because the frames are frozen and ``Vh_j`` is a square orthogonal
``b x b`` core (``out >= b`` for every LLM projection),

    W'_j = U_j diag(S0_j + c_j) Vh_j = W0_j (I + Vh_j^T diag(c_j / S0_j) Vh_j),

so the layer can keep its original weight ``W0`` untouched -- dense, or packed
INT4 for a GPTQ checkpoint -- and only the small core, ``S0`` and ``c`` are
stored. The extra work is a per-block ``b x b`` transform of the activation:

    y = W0 (x + dx),   dx_j = Vh_j^T diag(c_j / S0_j) Vh_j x_j.

It is the residual form: at ``c = 0`` the layer is bit-identical to the base
model whatever precision the core is stored in. The ES noise, its seeding and
the coefficient shape are those of ``StructuredES`` with scope ``fura_s``, so a
history replays to the same function on either implementation (up to the
rounding of the dense render, which this form avoids).

``mode="exact"`` evaluates ``W0 x + W0 dx`` with two GEMMs instead of rounding
``x + dx`` to the activation dtype; ``trainable=True`` exposes ``c`` as fp32
``nn.Parameter``s for backpropagation (FuRA-S BP).
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch

from esx.backends.structured_es import closest_factor_pair

SKIP_KEYWORDS = ("embed_tokens", "lm_head", "visual", "vision", "mtp")


def stable_tensor_id(name: str) -> int:
    """Same per-tensor id as ``SeedReplayModelES._stable_tensor_id``."""
    digest = hashlib.blake2b(str(name).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _dense_weight(module) -> Optional[torch.Tensor]:
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2 and torch.is_floating_point(weight):
        return weight.data
    if hasattr(module, "qweight"):
        return _probe_weight(module)
    return None


@torch.no_grad()
def _probe_weight(module, chunk: int = 2048) -> torch.Tensor:
    """Effective (out, in) weight of a quantized linear, read through its own kernel.

    Packing layouts differ between GPTQ backends (Marlin repacks), so the weight is
    recovered as ``module(I) - bias``: exactly what the layer applies to an input."""
    in_f = int(module.in_features)
    device = module.qweight.device
    dtype = getattr(module.scales, "dtype", torch.float16)
    bias = getattr(module, "bias", None)
    cols = []
    for lo in range(0, in_f, chunk):
        hi = min(in_f, lo + chunk)
        eye = torch.zeros(hi - lo, in_f, dtype=dtype, device=device)
        eye[:, lo:hi] = torch.eye(hi - lo, dtype=dtype, device=device)
        y = module(eye).float()  # (hi - lo, out) = W[:, lo:hi]^T
        if bias is not None:
            y = y - bias.float()
        cols.append(y)
    return torch.cat(cols, dim=0).t().contiguous()


def default_filter(name: str, module) -> bool:
    if any(key in name for key in SKIP_KEYWORDS):
        return False
    return isinstance(module, torch.nn.Linear) or hasattr(module, "qweight")


class FuraSPretransform:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        module_filter: Callable[[str, torch.nn.Module], bool] = default_filter,
        core_dtype: torch.dtype = torch.float32,
        core_quantizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        s_floor: float = 1e-3,
        mode: str = "fused",
        trainable: bool = False,
        svd_cache: Optional[str] = None,
    ):
        if mode not in ("fused", "exact"):
            raise ValueError("mode must be 'fused' or 'exact'.")
        self.mode = mode
        self.trainable = trainable
        self.states: List[Dict[str, Any]] = []
        self.skipped: List[str] = []
        self._handles = []
        self._perturbed = False
        # The block SVDs depend only on the base weights; reuse them across runs.
        cache = torch.load(svd_cache, map_location="cpu") if svd_cache and os.path.exists(svd_cache) else None
        self._svd: Dict[str, Any] = {}
        with torch.no_grad():
            for name, module in model.named_modules():
                if not default_filter(name, module) or not module_filter(name, module):
                    continue
                state = self._init_state(f"{name}.weight", module, core_dtype, core_quantizer, s_floor,
                                         None if cache is None else cache.get(f"{name}.weight"))
                if state is None:
                    self.skipped.append(name)
                    continue
                self.states.append(state)
                self._install(module, state)
        if not self.states:
            raise RuntimeError("No linear layers usable for fura_s.")
        if svd_cache and cache is None:
            os.makedirs(os.path.dirname(os.path.abspath(svd_cache)), exist_ok=True)
            torch.save(self._svd, svd_cache)
        self._svd = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ setup
    def _init_state(self, name, module, core_dtype, core_quantizer, s_floor, cached=None):
        weight = _dense_weight(module)
        if weight is None:
            return None
        out_f, in_f = weight.shape
        n_blk, b = closest_factor_pair(int(in_f))
        if out_f < b:
            return None
        if cached is not None:
            s, vh = cached["S0"].to(weight.device), cached["Vh"].to(weight.device)
        else:
            w = weight.float().reshape(out_f, n_blk, b).permute(1, 0, 2)
            _, s, vh = torch.linalg.svd(w, full_matrices=False)  # s (n_blk, b), vh (n_blk, b, b)
            del w
            self._svd[name] = {"S0": s.cpu(), "Vh": vh.cpu()}
        if core_quantizer is not None:
            vh = core_quantizer(vh)
        floor = s.amax(dim=-1, keepdim=True) * float(s_floor)
        coef = torch.zeros_like(s)
        if self.trainable:
            coef = torch.nn.Parameter(coef)
        return {
            "name": name,
            "tensor_id": stable_tensor_id(name),
            "module": module,
            "Vh": vh.to(core_dtype).contiguous(),
            "S0": s,
            "inv_s": 1.0 / torch.maximum(s, floor),
            "coef": coef,
            "offset": None,
            "n_blk": n_blk,
            "b": b,
        }

    def _install(self, module, state) -> None:
        def delta(x: torch.Tensor) -> Optional[torch.Tensor]:
            c = state["coef"] if state["offset"] is None else state["offset"]
            if not self.trainable and state["offset"] is None and not state.get("nonzero", False):
                return None
            # The correction is small next to x, so it is computed in the activation dtype.
            vh = state["Vh"].to(x.dtype)
            ratio = (c * state["inv_s"]).to(x.dtype)
            n_blk, b = state["n_blk"], state["b"]
            if ratio.dim() == 3:
                # One population member per group of rows (see perturb_members).
                xb = x.reshape(ratio.shape[0], -1, n_blk, b)
                z = torch.einsum("nkb,mtnb->mtnk", vh, xb) * ratio[:, None]
                return torch.einsum("nkb,mtnk->mtnb", vh, z).reshape(x.shape)
            xb = x.reshape(-1, n_blk, b)
            z = torch.einsum("nkb,tnb->tnk", vh, xb) * ratio
            return torch.einsum("nkb,tnk->tnb", vh, z).reshape(x.shape)

        if self.mode == "fused":
            def pre_hook(mod, args):
                dx = delta(args[0])
                if dx is None:
                    return None
                return (args[0] + dx.to(args[0].dtype),) + tuple(args[1:])

            self._handles.append(module.register_forward_pre_hook(pre_hook))
        else:
            original = module.forward

            def forward(x, *args, **kwargs):
                y = original(x, *args, **kwargs)
                dx = delta(x)
                if dx is None:
                    return y
                bias = getattr(mod_ref, "bias", None)
                extra = original(dx.to(x.dtype), *args, **kwargs)
                if bias is not None:
                    extra = extra - bias
                return y + extra

            mod_ref = module
            module.forward = forward
            state["original_forward"] = original

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        for state in self.states:
            if "original_forward" in state:
                state["module"].forward = state.pop("original_forward")
        self._handles = []

    # ------------------------------------------------------------------ noise
    @staticmethod
    def _noise(state: Dict[str, Any], seed: int) -> torch.Tensor:
        # Identical to StructuredES._generator / _coef_noise for scope fura_s.
        coef = state["coef"]
        generator = torch.Generator(device=coef.device)
        generator.manual_seed((int(seed) ^ state["tensor_id"]) & 0xFFFFFFFFFFFFFFFF)
        return torch.randn(coef.shape, generator=generator, dtype=torch.float32, device=coef.device)

    # ------------------------------------------------------------------ ES API
    @torch.no_grad()
    def perturb(self, *, seed: int, sigma: float) -> None:
        for state in self.states:
            state["offset"] = state["coef"].detach() + float(sigma) * self._noise(state, seed)
        self._perturbed = True

    @torch.no_grad()
    def perturb_members(self, *, seeds: Sequence[int], sigmas: Sequence[float]) -> None:
        """Perturb a whole population at once: the batch must hold ``len(seeds)``
        equal groups of rows, group ``m`` evaluated with ``coef + sigmas[m] * eps(seeds[m])``."""
        for state in self.states:
            noise = torch.stack([self._noise(state, seed) for seed in seeds])
            scale = torch.tensor([float(s) for s in sigmas], device=noise.device).view(-1, 1, 1)
            state["offset"] = state["coef"].detach().unsqueeze(0) + scale * noise
        self._perturbed = True

    @torch.no_grad()
    def restore(self) -> None:
        for state in self.states:
            state["offset"] = None
        self._perturbed = False

    @torch.no_grad()
    def update(self, *, seeds: Sequence[int], weights: Sequence[float], eta: float) -> Dict[str, float]:
        if self._perturbed:
            raise RuntimeError("Restore the perturbation before updating.")
        scale = float(eta) / float(len(seeds))
        for state in self.states:
            total = torch.zeros_like(state["coef"])
            for seed, weight in zip(seeds, weights):
                total.add_(self._noise(state, seed), alpha=scale * float(weight))
            state["coef"].data.add_(total)
            state["nonzero"] = True
        return self.metrics()

    @torch.no_grad()
    def reset(self) -> None:
        for state in self.states:
            state["coef"].data.zero_()
            state["offset"] = None
            state["nonzero"] = False
        self._perturbed = False

    def mark_trained(self) -> None:
        """After an optimizer step on ``coef`` (BP), make the hooks apply it at inference."""
        for state in self.states:
            state["nonzero"] = True

    def parameters(self) -> List[torch.Tensor]:
        return [state["coef"] for state in self.states]

    @torch.no_grad()
    def metrics(self) -> Dict[str, float]:
        rel = [float((state["coef"] * state["inv_s"]).abs().max()) for state in self.states]
        return {"max_rel_s_change": max(rel) if rel else 0.0}

    def info(self) -> Dict[str, Any]:
        core = sum(state["Vh"].numel() * state["Vh"].element_size() for state in self.states)
        return {
            "n_tensors": len(self.states),
            "total_params": int(sum(state["coef"].numel() for state in self.states)),
            "core_bytes": int(core),
            "skipped_tensors": len(self.skipped),
            "mode": self.mode,
        }
