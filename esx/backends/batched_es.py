"""Batched ``isobtt`` / ``fura_s``: one forward pass, a different perturbation per sequence.

``structured_es.py`` renders a perturbed dense weight, so an engine can only
evaluate one population member at a time and must keep the frozen SVD frame
*next to* the dense weight. This module removes the dense weight instead. Each
2-D linear weight is replaced by its block-SVD factors and the layer computes

    y = frame @ perturb(core, x)

directly from them, in the model dtype. The perturbation is applied to the
small intermediate activation, per token, from the population member that owns
the token's sequence -- the activation-space trick of Flipout (Wen et al.,
"Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches",
ICLR 2018), which is what lets one batch mix all members.

``isobtt``  ``W[:, blk_j] = A_j R_j``, ``A_j = U_j diag(S_j)`` frozen, the
            orthogonal core ``R_j`` trained. A generation shares one skew
            generator ``Omega_j`` and member ``n`` flips its sign pattern,
            ``Omega_{n,j} = D_n Omega_j D_n`` with ``D_n = diag(+-1)``. Because
            ``Cay(D Omega D) = D Cay(Omega) D`` the member's rotation is exactly
            orthogonal and costs one shared ``k x k`` product per block:
            ``z <- z + D_n (Cay(sigma Omega) - I) D_n z`` with ``z = R x``.
            The update is ``R <- Cay(alpha * Omega o M) R`` with
            ``M = (1/N) sum_n w_n d_n d_n^T``.
``fura_s``  ``W[:, blk_j] = U_j diag(s_j) Vh_j`` with both frames frozen and the
            spectrum ``s_j`` trained. The core is a vector, so every member gets
            its own Gaussian draw (no sign sharing is needed):
            ``z <- (s + sigma * eps_n) o z`` with ``z = Vh x``. The draws are the
            ones the unbatched ``fura_s`` scope uses for the same seeds.

Cores are stored and updated in the model dtype. One ES step is smaller than a
bf16 ulp for many entries, so the update is written back with seeded stochastic
rounding: unbiased, and reproducible when a history is replayed.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from esx.backends.structured_es import _cayley, _skew, closest_factor_pair, normalize_scope_options


ParameterInfo = Tuple[str, int, torch.nn.Parameter]

BATCHED_SCOPES = ("fura_s", "isobtt")

_ROUNDING_SALT = 0x5DEECE66D
_MASK64 = 0xFFFFFFFFFFFFFFFF


def group_seed(seeds: Sequence[int]) -> int:
    """Seed of the generator a generation's members share; a function of all its seeds."""
    payload = struct.pack(f"<{len(seeds)}q", *[int(seed) for seed in seeds])
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") >> 1


def stochastic_round(x: torch.Tensor, dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    """Round fp32 ``x`` to ``dtype`` so that the expectation equals ``x``.

    bf16 is the top 16 bits of fp32, so the two neighbours of ``x`` are the
    truncated bit pattern and that pattern plus one bf16 ulp; the dropped low
    bits are the probability of the upper one.
    """
    if dtype != torch.bfloat16:
        return x.to(dtype)
    bits = x.contiguous().view(torch.int32)
    low = (bits & 0xFFFF).float()
    draw = torch.rand(x.shape, generator=generator, dtype=torch.float32, device=x.device) * 65536.0
    up = (draw < low).to(torch.int32) << 16
    return ((bits & ~0xFFFF) + up).view(torch.float32).to(dtype)


class _FactoredLinear:
    """Block-SVD factors of one linear weight and the forward pass through them."""

    def __init__(self, owner: "BatchedStructuredES", name: str, tensor_id: int, weight: torch.Tensor):
        self.owner = owner
        self.name = name
        self.tensor_id = int(tensor_id)
        self.out_f, self.in_f = (int(dim) for dim in weight.shape)
        self.dtype = weight.dtype
        self.device = weight.device
        n_blk, b = closest_factor_pair(self.in_f)
        if owner.options["swap_blocks"]:
            n_blk, b = b, n_blk
        self.n_blk, self.b, self.k = n_blk, b, min(self.out_f, b)

        w = weight.reshape(self.out_f, n_blk, b).permute(1, 0, 2).float()
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        if owner.scope == "isobtt":
            frame, core = u * s.unsqueeze(1), vh  # core (n_blk, k, b)
            self.right: Optional[torch.Tensor] = None
            self.manifold_dim = n_blk * self.k * (self.k - 1) // 2
        else:
            frame, core = u, s  # core (n_blk, k)
            self.right = vh.to(self.dtype).contiguous()
            self.manifold_dim = int(s.numel())
        # (out, n_blk * k): the last product is one GEMM the size of the dense weight.
        self.frame = frame.permute(1, 0, 2).reshape(self.out_f, n_blk * self.k).to(self.dtype).contiguous()
        self.core = core.to(self.dtype).contiguous()
        self.core0 = self.core.clone()
        # Per-generation population state; slot 0 is "no perturbation".
        self.rot_delta_t: Optional[torch.Tensor] = None  # isobtt (n_blk, k, k) = (Cay - I)^T
        self.slot_table: Optional[torch.Tensor] = None  # (n_blk, N + 1, k): signs or sigma * eps

    def generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        # Same per-tensor mixing as every other scope.
        generator.manual_seed((int(seed) ^ self.tensor_id) & _MASK64)
        return generator

    def signs(self, seed: int) -> torch.Tensor:
        bits = torch.randint(0, 2, (self.n_blk, self.k), generator=self.generator(seed), device=self.device)
        return bits.float().mul_(2.0).sub_(1.0)

    def noise(self, seed: int) -> torch.Tensor:
        return torch.randn(
            (self.n_blk, self.k), generator=self.generator(seed), dtype=torch.float32, device=self.device
        )

    def omega(self, seeds: Sequence[int]) -> torch.Tensor:
        return _skew(self.n_blk, self.k, self.generator(group_seed(seeds)), self.device)

    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        lead = x.shape[:-1]
        x = x.reshape(-1, self.in_f)
        tokens = x.shape[0]
        blocks = x.view(tokens, self.n_blk, self.b).transpose(0, 1)  # (n_blk, T, b)
        right = self.core if self.right is None else self.right
        z = torch.bmm(blocks, right.transpose(1, 2))  # (n_blk, T, k)
        slots = self.owner.token_slots(tokens) if self.slot_table is not None else None
        if self.right is None:
            if slots is not None:
                sign = self.slot_table.index_select(1, slots)
                z = z + torch.bmm(z * sign, self.rot_delta_t) * sign
        else:
            scaled = z * self.core.unsqueeze(1)
            if slots is not None:
                scaled = scaled + z * self.slot_table.index_select(1, slots)
            z = scaled
        y = F.linear(z.transpose(0, 1).reshape(tokens, self.n_blk * self.k), self.frame, bias)
        return y.reshape(*lead, self.out_f)

    def dense(self, slot: int = 0) -> torch.Tensor:
        """The fp32 weight this layer applies for ``slot``; for tests and diagnostics."""
        core = self.core.float()
        if self.right is None:
            if slot and self.slot_table is not None:
                sign = self.slot_table[:, slot].float()
                rotation = self.rot_delta_t.float().transpose(1, 2) * sign.unsqueeze(1) * sign.unsqueeze(2)
                core = core + torch.bmm(rotation, core)
            mid = core  # (n_blk, k, b)
        else:
            if slot and self.slot_table is not None:
                core = core + self.slot_table[:, slot].float()
            mid = core.unsqueeze(2) * self.right.float()
        frame = self.frame.float().view(self.out_f, self.n_blk, self.k).permute(1, 0, 2)
        return torch.bmm(frame, mid).permute(1, 0, 2).reshape(self.out_f, self.in_f)


class _FactoredMethod:
    """Stand-in for a vLLM linear ``quant_method`` that routes through the factors."""

    def __init__(self, original, layer: _FactoredLinear):
        self.original = original
        self.layer = layer

    def apply(self, module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.layer.forward(x, bias)

    def __getattr__(self, name: str):
        return getattr(self.original, name)


class BatchedStructuredES:
    """Factored linears plus per-sequence population state for ``isobtt`` / ``fura_s``."""

    def __init__(
        self,
        model,
        parameter_infos: Sequence[ParameterInfo],
        *,
        scope: str,
        options: Optional[Mapping[str, Any]] = None,
    ):
        if scope not in BATCHED_SCOPES:
            raise ValueError(f"Batched perturbation supports {BATCHED_SCOPES}, not {scope!r}.")
        self.scope = scope
        self.options = normalize_scope_options(options)
        self.layers: List[_FactoredLinear] = []
        self.skipped: List[str] = []
        self.population: List[int] = []
        self.sigma = 0.0
        self._slots: Optional[torch.Tensor] = None
        self._build(model, parameter_infos)

    # ------------------------------------------------------------------ setup
    @torch.no_grad()
    def _build(self, model, parameter_infos: Sequence[ParameterInfo]) -> None:
        owners = {}
        for module in model.modules():
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.nn.Parameter):
                owners.setdefault(id(weight), module)
        for name, tensor_id, parameter in parameter_infos:
            module = owners.get(id(parameter))
            if parameter.ndim != 2 or module is None or not self._replaceable(module):
                self.skipped.append(name)
                continue
            n_blk, b = closest_factor_pair(int(parameter.shape[1]))
            if self.scope == "isobtt" and min(int(parameter.shape[0]), n_blk if self.options["swap_blocks"] else b) < 2:
                self.skipped.append(name)
                continue
            layer = _FactoredLinear(self, name, tensor_id, parameter.data)
            self._install(module, layer)
            # The dense weight is not kept: the factors replace it.
            parameter.data = torch.empty((0, layer.in_f), dtype=layer.dtype, device=layer.device)
            self.layers.append(layer)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not self.layers:
            raise RuntimeError(f"No linear weights are usable for batched scope {self.scope!r}.")

    @staticmethod
    def _replaceable(module) -> bool:
        if isinstance(module, torch.nn.Linear):
            return True
        method = getattr(module, "quant_method", None)
        return method is not None and type(method).__name__ == "UnquantizedLinearMethod"

    @staticmethod
    def _install(module, layer: _FactoredLinear) -> None:
        if isinstance(module, torch.nn.Linear):
            module.forward = lambda x, _layer=layer, _module=module: _layer.forward(x, _module.bias)
        else:
            module.quant_method = _FactoredMethod(module.quant_method, layer)

    # ------------------------------------------------------------- slot routing
    def set_token_slots(self, slots: Optional[torch.Tensor]) -> None:
        """Per-token population slot for the next forward pass: 0 = unperturbed, n = member n-1."""
        self._slots = slots

    def token_slots(self, tokens: int) -> Optional[torch.Tensor]:
        slots = self._slots
        if slots is None:
            return None
        if slots.shape[0] == tokens:
            return slots
        if slots.shape[0] < tokens:
            # Padded batches: the padding rows are unperturbed.
            return F.pad(slots, (0, tokens - slots.shape[0]))
        raise RuntimeError(f"A linear layer saw {tokens} tokens but {slots.shape[0]} population slots were set.")

    # ------------------------------------------------------------------ ES API
    @torch.no_grad()
    def set_population(self, *, seeds: Sequence[int], sigma: float) -> None:
        seeds = [int(seed) for seed in seeds]
        if not seeds:
            raise ValueError("A population needs at least one seed.")
        for layer in self.layers:
            table = torch.zeros((layer.n_blk, len(seeds) + 1, layer.k), dtype=torch.float32, device=layer.device)
            if self.scope == "isobtt":
                rotation = _cayley(layer.omega(seeds), float(sigma))
                rotation.diagonal(dim1=1, dim2=2).sub_(1.0)
                layer.rot_delta_t = rotation.transpose(1, 2).to(layer.dtype).contiguous()
                for index, seed in enumerate(seeds, 1):
                    table[:, index] = layer.signs(seed)
            else:
                for index, seed in enumerate(seeds, 1):
                    table[:, index] = layer.noise(seed).mul_(float(sigma))
            layer.slot_table = table.to(layer.dtype)
        self.population = seeds
        self.sigma = float(sigma)

    @torch.no_grad()
    def clear_population(self) -> None:
        for layer in self.layers:
            layer.rot_delta_t = None
            layer.slot_table = None
        self.population = []
        self.sigma = 0.0
        self._slots = None

    @torch.no_grad()
    def update(self, *, seeds: Sequence[int], weights: Sequence[float], eta: float) -> Dict[str, float]:
        if not seeds:
            raise ValueError("ES update requires at least one seed.")
        if len(seeds) != len(weights):
            raise ValueError("ES update requires seeds and weights to have the same length.")
        if self.population:
            raise RuntimeError("Clear the population before applying an ES update.")
        seeds = [int(seed) for seed in seeds]
        coeffs = [float(eta) / float(len(seeds)) * float(weight) for weight in weights]
        rounding_seed = group_seed(seeds) ^ _ROUNDING_SALT
        for layer in self.layers:
            if self.scope == "isobtt":
                signs = torch.stack([layer.signs(seed) for seed in seeds])  # (N, n_blk, k)
                scale = torch.tensor(coeffs, dtype=torch.float32, device=layer.device)
                mix = torch.einsum("n,nbi,nbj->bij", scale, signs, signs)
                core = self._orthonormalize(layer.core.float())
                value = torch.bmm(_cayley(layer.omega(seeds) * mix, 1.0), core)
            else:
                value = layer.core.float()
                for seed, coeff in zip(seeds, coeffs):
                    value.add_(layer.noise(seed), alpha=coeff)
            layer.core.copy_(stochastic_round(value, layer.dtype, layer.generator(rounding_seed)))
        return self.metrics()

    @staticmethod
    def _orthonormalize(core: torch.Tensor) -> torch.Tensor:
        """One Newton-Schulz step: removes the rounding left by the low-precision store."""
        gram = torch.bmm(core, core.transpose(1, 2))
        return torch.baddbmm(core, gram, core, beta=1.5, alpha=-0.5)

    @torch.no_grad()
    def reset(self) -> None:
        self.clear_population()
        for layer in self.layers:
            layer.core.copy_(layer.core0)

    @torch.no_grad()
    def metrics(self) -> Dict[str, float]:
        """``orth_err``: worst ``|R R^T - I|`` entry of the stored (model-dtype) isobtt cores."""
        if self.scope != "isobtt":
            return {}
        worst = 0.0
        for layer in self.layers:
            core = layer.core.float()
            gram = torch.bmm(core, core.transpose(1, 2))
            gram.diagonal(dim1=1, dim2=2).sub_(1.0)
            worst = max(worst, float(gram.abs().max()))
        return {"orth_err": worst}

    def parameter_infos(self) -> List[Tuple[str, int, torch.Tensor]]:
        return [(layer.name, layer.tensor_id, layer.core) for layer in self.layers]

    def info(self) -> Dict[str, Any]:
        return {
            "n_tensors": len(self.layers),
            "total_params": sum(int(layer.core.numel()) for layer in self.layers),
            "manifold_dim": sum(layer.manifold_dim for layer in self.layers),
            "skipped_tensors": len(self.skipped),
            "scope_options": dict(self.options),
        }
