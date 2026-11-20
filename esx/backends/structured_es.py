"""Structured ES scopes: ``fura``, ``fura_s``, ``iso``, and ``isobtt``.

The dense scopes in ``seeded_model_es.py`` add seeded Gaussian noise to the live
weights. The scopes here instead keep an fp32 *state* per linear weight and
render the weight from it, so a perturbation is ``W = render(perturb(state))``
and a revert is ``W = render(state)`` -- exact, with no add/subtract round-off.
Noise is still never stored: every direction is regenerated from
``(seed, stable per-tensor id)``, so a history replays to the same weights.

``fura``    Full-rank block-wise SVD of each weight. Per input block ``j``,
            ``W[:, blk_j] = A_j R_j`` with ``A_j = U_j diag(S_j)`` frozen and the
            small core ``R_j = Vh_j`` trained additively:
            ``W[:, blk_j] = A_j (R0_j + C_j)``, Gaussian noise on ``C``.
            ``sigma`` is an absolute noise std on the core.
``fura_s``  The same factorisation with both frames frozen and only the
            singular values trained: ``W[:, blk_j] = U_j diag(S0_j + c_j) Vh_j``,
            Gaussian noise on ``c``. ``sigma`` is an absolute noise std on the
            singular values. This is the complement of ``isobtt``, which moves
            the frames and freezes the spectrum.
``isobtt``  The same factorisation with the core kept on the orthogonal group:
            ``R_j <- Cay(sigma * Omega_j) R_j`` for a seeded skew ``Omega_j``.
            Each block's singular values are preserved exactly.
``iso``     Fixed-spectrum ES on the whole matrix (ISO, arXiv:2607.19331):
            ``W <- C_L W C_R^T`` with ``C_L``, ``C_R`` exactly orthogonal Cayley
            transforms of block-diagonal skew generators in a seed-dependent
            permuted basis. The singular values of ``W`` are preserved exactly.

For ``iso`` and ``isobtt`` the skew entries are ``N(0, 1/b)``, which makes
``sigma`` the *relative* weight-space displacement ``||dW||_F / ||W||_F`` rather
than a noise std; the two sides of ``iso`` share ``sigma / sqrt(2)``. The update
lives in the Lie algebra and is realised as the ordered product of the per-seed
Cayley factors at scale ``(alpha / N) * weight_i``, which is exactly orthogonal.

This module is shared by ``SeedReplayModelES`` and the in-process vLLM worker
(``vllm_math_es_worker.py``) so the two serving paths cannot drift apart.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


ParameterInfo = Tuple[str, int, torch.nn.Parameter]

STRUCTURED_SCOPES = ("fura", "fura_s", "iso", "isobtt")

# Scopes whose trained state is an additive coefficient tensor with Gaussian noise.
_ADDITIVE_SCOPES = ("fura", "fura_s")

DEFAULT_SCOPE_OPTIONS = {
    # Block size of the iso skew generator; one perturbation costs O(b * |W|).
    "iso_block_size": 128,
    # Re-draw the iso block basis for every seed, so the directions explored
    # across generations are not confined to one block-diagonal subgroup.
    "iso_perm": True,
    # Swap (n_blocks, block_size) of the fura / fura_s / isobtt input-block factoring.
    "swap_blocks": False,
    # fura_s / isobtt only: drop the dense weight, run the layer from its factors,
    # and perturb each sequence of a batch separately (see batched_es.py). In-process vLLM only.
    "batched": False,
}


def normalize_scope_options(options: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    normalized = dict(DEFAULT_SCOPE_OPTIONS)
    for key, value in dict(options or {}).items():
        if key not in DEFAULT_SCOPE_OPTIONS:
            supported = ", ".join(sorted(DEFAULT_SCOPE_OPTIONS))
            raise ValueError(f"Unknown scope option {key!r}. Supported options: {supported}")
        normalized[key] = value
    normalized["iso_block_size"] = int(normalized["iso_block_size"])
    if normalized["iso_block_size"] < 2:
        raise ValueError("iso_block_size must be at least 2.")
    normalized["iso_perm"] = bool(normalized["iso_perm"])
    normalized["swap_blocks"] = bool(normalized["swap_blocks"])
    normalized["batched"] = bool(normalized["batched"])
    return normalized


def closest_factor_pair(n: int) -> Tuple[int, int]:
    """``(n_blocks, block_size)`` with ``n_blocks * block_size == n``, both near ``sqrt(n)``."""
    for p in range(math.isqrt(int(n)), 0, -1):
        if n % p == 0:
            return p, n // p
    return 1, int(n)


def _block_size(dims: Sequence[int], requested: int) -> int:
    """Largest ``b <= requested`` dividing every dim, so no block straddles a segment."""
    common = 0
    for dim in dims:
        common = math.gcd(common, int(dim))
    b = min(int(requested), common)
    while b > 1 and common % b:
        b -= 1
    return b


def _skew(n_blk: int, b: int, generator: torch.Generator, device) -> torch.Tensor:
    """Block-diagonal skew generator with entries ~ N(0, 1/b), so ``||Omega F|| ~ ||F||``."""
    e = torch.randn(n_blk, b, b, dtype=torch.float32, device=device, generator=generator)
    return (e - e.transpose(1, 2)).mul_((2.0 * b) ** -0.5)


def _cayley(omega: torch.Tensor, scale: float) -> torch.Tensor:
    """``Cay(scale * Omega) = (I - X/2)^-1 (I + X/2)``, exactly orthogonal for skew ``Omega``."""
    n_blk, b, _ = omega.shape
    a = omega * (0.5 * float(scale))
    eye = torch.eye(b, dtype=omega.dtype, device=omega.device).expand(n_blk, b, b)
    return torch.linalg.solve(eye - a, eye + a)


def _bmm_exact(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # The TF32 mantissa is coarser than one ES step, so it must stay off here.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return torch.bmm(a, b)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _rotate_rows(x: torch.Tensor, perm: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``Cay(Omega_L) @ X`` with the blocks laid over the permuted rows."""
    m, n = x.shape
    n_blk, b, _ = c.shape
    y = torch.bmm(c, x.index_select(0, perm).view(n_blk, b, n)).view(m, n)
    out = torch.empty_like(x)
    out.index_copy_(0, perm, y)
    return out


def _rotate_columns(x: torch.Tensor, perm: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``X @ Cay(Omega_R)^T`` with the blocks laid over the permuted columns."""
    m, n = x.shape
    n_blk, b, _ = c.shape
    xp = x.index_select(1, perm).view(m, n_blk, b).transpose(0, 1).contiguous()
    y = torch.bmm(xp, c.transpose(1, 2)).transpose(0, 1).reshape(m, n)
    out = torch.empty_like(x)
    out.index_copy_(1, perm, y)
    return out


def _output_segments(model) -> Dict[int, List[int]]:
    """Output-row segments of fused weights, keyed by ``id(weight)``.

    vLLM fuses q/k/v and gate/up into one matrix and records the split in
    ``output_sizes``. An iso row rotation must not mix rows across that split,
    or it would preserve the spectrum of the fused matrix instead of the
    spectra of the matrices the checkpoint actually defines.
    """
    segments: Dict[int, List[int]] = {}
    for module in model.modules():
        weight = getattr(module, "weight", None)
        sizes = getattr(module, "output_sizes", None)
        if weight is None or not sizes or getattr(weight, "ndim", 0) != 2:
            continue
        sizes = [int(size) for size in sizes]
        if len(sizes) > 1 and sum(sizes) == int(weight.shape[0]):
            segments[id(weight)] = sizes
    return segments


class StructuredES:
    """State and update math for one structured scope over a set of 2-D weights."""

    def __init__(
        self,
        model,
        parameter_infos: Sequence[ParameterInfo],
        *,
        scope: str,
        options: Optional[Mapping[str, Any]] = None,
    ):
        if scope not in STRUCTURED_SCOPES:
            raise ValueError(f"Unsupported structured scope: {scope!r}")
        self.scope = scope
        self.options = normalize_scope_options(options)
        if self.options["batched"]:
            raise ValueError(
                "The 'batched' scope option is served by BatchedStructuredES in the in-process "
                "vLLM worker; this serving path renders dense weights and cannot use it."
            )
        self.states: List[Dict[str, Any]] = []
        self.skipped: List[str] = []
        self.last_drift = 0.0
        # iso keeps no copy of the base weights, so reset replays this backwards.
        self._rotation_log: List[Tuple[List[int], List[float]]] = []
        self._build(model, parameter_infos)

    # ------------------------------------------------------------------ setup
    @torch.no_grad()
    def _build(self, model, parameter_infos: Sequence[ParameterInfo]) -> None:
        segments = _output_segments(model) if self.scope == "iso" else {}
        for name, tensor_id, parameter in parameter_infos:
            if parameter.ndim != 2:
                continue
            if self.scope == "iso":
                state = self._init_iso(parameter, segments.get(id(parameter)))
            else:
                state = self._init_block_svd(parameter)
            if state is None:
                self.skipped.append(name)
                continue
            state.update({"name": name, "tensor_id": int(tensor_id), "parameter": parameter})
            self.states.append(state)
            if self.scope != "iso":
                # The block-SVD reconstruction differs from the loaded weight by
                # rounding in the model dtype; write it so step 0 is consistent.
                self._render(state, self._current(state))
        if not self.states:
            raise RuntimeError(f"No 2-D weights are usable for scope {self.scope!r}.")
        self._release_cached_memory()

    def _init_iso(self, parameter, segments: Optional[List[int]]) -> Optional[Dict[str, Any]]:
        m, n = parameter.shape
        segments = segments or [int(m)]
        bl = _block_size(segments, self.options["iso_block_size"])
        br = _block_size([n], self.options["iso_block_size"])
        if bl < 2 or br < 2:
            return None
        master = parameter.data.detach().clone().float()
        return {
            "state": master,
            "segments": segments,
            "bl": bl,
            "br": br,
            "fro0": float(master.norm()),
            "coef_params": int(master.numel()),
            "manifold_dim": (int(m) * (bl - 1) + int(n) * (br - 1)) // 2,
        }

    def _init_block_svd(self, parameter) -> Optional[Dict[str, Any]]:
        out_f, in_f = parameter.shape
        n_blk, b = closest_factor_pair(in_f)
        if self.options["swap_blocks"]:
            n_blk, b = b, n_blk
        k = min(int(out_f), b)
        if self.scope == "isobtt" and k < 2:
            return None
        w = parameter.data.reshape(out_f, n_blk, b).permute(1, 0, 2).float()
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        if self.scope == "fura_s":
            # Frames frozen in the model dtype; only the fp32 spectrum is trained.
            spectrum = s.float().contiguous()  # (n_blk, k)
            return {
                "U": u.to(parameter.dtype).contiguous(),
                "Vh": vh.to(parameter.dtype).contiguous(),
                "S0": spectrum,
                "coef": torch.zeros_like(spectrum),
                "n_blk": n_blk,
                "b": b,
                "k": k,
                "coef_params": int(spectrum.numel()),
                "manifold_dim": int(spectrum.numel()),
            }
        frame = (u * s.unsqueeze(1)).to(parameter.dtype).contiguous()  # (n_blk, out, k), frozen
        core = vh.float().contiguous()  # (n_blk, k, b), trained
        state: Dict[str, Any] = {"A": frame, "R0": core, "n_blk": n_blk, "b": b, "k": k}
        if self.scope == "fura":
            state["coef"] = torch.zeros_like(core)
            state["coef_params"] = int(core.numel())
            state["manifold_dim"] = int(core.numel())
        else:
            state["state"] = core.clone()
            state["coef_params"] = int(core.numel())
            state["manifold_dim"] = n_blk * k * (k - 1) // 2
        return state

    # ------------------------------------------------------------------ noise
    @staticmethod
    def _generator(state: Dict[str, Any], seed: int) -> torch.Generator:
        generator = torch.Generator(device=state["parameter"].device)
        # Same per-tensor mixing as the dense scopes.
        generator.manual_seed((int(seed) ^ state["tensor_id"]) & 0xFFFFFFFFFFFFFFFF)
        return generator

    def _rotate(self, state: Dict[str, Any], x: torch.Tensor, seed: int, scale: float) -> torch.Tensor:
        """Apply one seed's exactly orthogonal rotation to ``x``."""
        generator = self._generator(state, seed)
        device = x.device
        if self.scope == "isobtt":
            omega = _skew(state["n_blk"], state["k"], generator, device)
            return torch.bmm(_cayley(omega, scale), x)
        m, n = x.shape
        if self.options["iso_perm"]:
            parts, offset = [], 0
            for segment in state["segments"]:
                parts.append(torch.randperm(segment, generator=generator, device=device) + offset)
                offset += segment
            perm_l = parts[0] if len(parts) == 1 else torch.cat(parts)
            perm_r = torch.randperm(n, generator=generator, device=device)
        else:
            perm_l = torch.arange(m, device=device)
            perm_r = torch.arange(n, device=device)
        side_scale = float(scale) * (0.5 ** 0.5)
        c_l = _cayley(_skew(m // state["bl"], state["bl"], generator, device), side_scale)
        c_r = _cayley(_skew(n // state["br"], state["br"], generator, device), side_scale)
        return _rotate_columns(_rotate_rows(x, perm_l, c_l), perm_r, c_r)

    def _coef_noise(self, state: Dict[str, Any], seed: int) -> torch.Tensor:
        coef = state["coef"]
        return torch.randn(
            coef.shape,
            generator=self._generator(state, seed),
            dtype=torch.float32,
            device=coef.device,
        )

    # ------------------------------------------------------------------ render
    def _current(self, state: Dict[str, Any]) -> torch.Tensor:
        if self.scope == "fura":
            return state["R0"] + state["coef"]
        if self.scope == "fura_s":
            return state["S0"] + state["coef"]
        return state["state"]

    def _render(self, state: Dict[str, Any], x: torch.Tensor) -> None:
        """Write the weight for core / master ``x`` into the live parameter."""
        parameter = state["parameter"]
        if self.scope == "iso":
            parameter.data.copy_(x)
            return
        if self.scope == "fura_s":
            w = _bmm_exact(state["U"].float() * x.unsqueeze(1), state["Vh"].float())
        else:
            w = _bmm_exact(state["A"].float(), x)  # (n_blk, out, b)
        parameter.data.copy_(w.permute(1, 0, 2).reshape(parameter.shape))

    def _recondition(self, state: Dict[str, Any]) -> float:
        """Remove fp32 accumulation drift; return the violation that was there."""
        x = state["state"]
        if self.scope == "iso":
            # Repeated fp32 Cayley products leave an isotropic gain on the
            # singular values. ||W||_F is an exact invariant of the orbit, so
            # one scalar per matrix removes it.
            fro = float(x.norm())
            if fro <= 0.0 or state["fro0"] <= 0.0:
                return 0.0
            x.mul_(state["fro0"] / fro)
            return abs(fro / state["fro0"] - 1.0)
        # isobtt: one Newton-Schulz step pulls the core rows back to orthonormal.
        gram = torch.bmm(x, x.transpose(1, 2))
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        error = float((gram - eye).abs().max())
        x.copy_(torch.baddbmm(x, gram, x, beta=1.5, alpha=-0.5))
        return error

    @staticmethod
    def _release_cached_memory() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ ES API
    @torch.no_grad()
    def perturb(self, *, seed: int, sigma: float) -> None:
        for state in self.states:
            if self.scope in _ADDITIVE_SCOPES:
                x = self._current(state).add_(self._coef_noise(state, seed), alpha=float(sigma))
            else:
                x = self._rotate(state, state["state"], seed, float(sigma))
            self._render(state, x)
            del x
        self._release_cached_memory()

    @torch.no_grad()
    def restore(self) -> None:
        for state in self.states:
            self._render(state, self._current(state))
        self._release_cached_memory()

    @torch.no_grad()
    def update(self, *, seeds: Sequence[int], weights: Sequence[float], eta: float) -> Dict[str, float]:
        if not seeds:
            raise ValueError("ES update requires at least one seed.")
        if len(seeds) != len(weights):
            raise ValueError("ES update requires seeds and weights to have the same length.")
        seeds = [int(seed) for seed in seeds]
        scale = float(eta) / float(len(seeds))
        coeffs = [scale * float(weight) for weight in weights]
        drift = 0.0
        for state in self.states:
            if self.scope in _ADDITIVE_SCOPES:
                total = torch.zeros_like(state["coef"])
                for seed, coeff in zip(seeds, coeffs):
                    total.add_(self._coef_noise(state, seed), alpha=coeff)
                state["coef"].add_(total)
                del total
            else:
                drift = max(drift, self._apply_rotations(state, seeds, coeffs))
            self._render(state, self._current(state))
        if self.scope == "iso":
            self._rotation_log.append((seeds, coeffs))
        self.last_drift = drift
        self._release_cached_memory()
        return self.metrics()

    def _apply_rotations(self, state: Dict[str, Any], seeds: Sequence[int], coeffs: Sequence[float]) -> float:
        """``state <- (prod_i Cay(coeff_i * Omega_i)) state``, then recondition."""
        x = state["state"]
        for seed, coeff in zip(seeds, coeffs):
            if coeff == 0.0:
                continue
            x = self._rotate(state, x, seed, coeff)
        if x is not state["state"]:
            state["state"].copy_(x)
        return self._recondition(state)

    @torch.no_grad()
    def reset(self) -> None:
        """Return to the state right after construction and rewrite the weights."""
        for state in self.states:
            if self.scope in _ADDITIVE_SCOPES:
                state["coef"].zero_()
            elif self.scope == "isobtt":
                state["state"].copy_(state["R0"])
            else:
                # Cay(-s Omega) is the exact inverse of Cay(s Omega).
                for seeds, coeffs in reversed(self._rotation_log):
                    self._apply_rotations(
                        state,
                        list(reversed(seeds)),
                        [-coeff for coeff in reversed(coeffs)],
                    )
            self._render(state, self._current(state))
        self._rotation_log = []
        self.last_drift = 0.0
        self._release_cached_memory()

    @torch.no_grad()
    def metrics(self) -> Dict[str, float]:
        """Constraint health of the fixed-spectrum scopes.

        ``frob_drift`` (iso) is the worst relative change of ``||W||_F`` seen at
        the last update, before it was corrected. ``orth_err`` (isobtt) is the
        worst ``|R R^T - I|`` entry of the current cores.
        """
        if self.scope == "iso":
            return {"frob_drift": float(self.last_drift)}
        if self.scope == "isobtt":
            worst = 0.0
            for state in self.states:
                x = state["state"]
                gram = torch.bmm(x, x.transpose(1, 2))
                eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
                worst = max(worst, float((gram - eye).abs().max()))
            return {"orth_err": worst}
        return {}

    def info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "n_tensors": len(self.states),
            "total_params": sum(state["coef_params"] for state in self.states),
            "manifold_dim": sum(state["manifold_dim"] for state in self.states),
            "skipped_tensors": len(self.skipped),
            "scope_options": dict(self.options),
        }
        if self.scope == "iso":
            info["fused_tensors"] = sum(len(state["segments"]) > 1 for state in self.states)
        return info
