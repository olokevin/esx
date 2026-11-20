"""vLLM worker extension: ES over one of several weight parameterizations.

Every linear weight is rendered as ``W = W0 + Phi(theta)`` from an fp32 state ``theta``
(W0 is a frozen bf16 copy of the base weight), so perturb / restore are exact:

  full         Phi = Delta                                    theta: (out, in)
  lora         Phi = B A,  B0 = 0, A0 ~ Kaiming-uniform        theta: B (out, r), A (r, in)
  lora_fa      Phi = B A0, A frozen at its init               theta: B
  fura_s       Phi_j = U_j diag(c_j) Vh_j  (input-block SVD of W0, frames frozen), isotropic noise on c
               -- equal to the repo's fura_s scope U diag(S0 + c) Vh, written as a residual
  fura_s_rel   fura_s with the paper's noise scale: eps_c = |S0| z (D = diag(|theta0|) of FuRA-S Eq. 3), so
               dW_j = W_j Vh_j^T diag(sigma z_j) Vh_j: the spectrally preconditioned search distribution
  fura_s_rand  same block structure as fura_s, Haar-random frames (basis ablation)
  unilora      Uni-LoRA (Li et al., 2025; PEFT / official construction): LoRA rank r on every weight, every A / B
               entry = (theta * nrm)[idx], ONE vector theta (length --uni-d) shared by the whole model,
               idx a balanced random partition, nrm = 1 / sqrt(count), theta0 ~ U(+-0.02); Phi = B A
  steer        output-side adapter: y = W0 x + b, b (out,) per linear (a steering vector after every weight);
               applied with forward hooks, so the engine must run eager (es_rl.py sets enforce_eager)

Noise for a seed is regenerated from ``(seed ^ stable tensor id)`` as in algorithms/es.
Loaded with ``worker_extension_cls="esx.backends.mwt.param_es_worker.WorkerExtension"``.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from esx.backends.structured_es import closest_factor_pair

SKIP = ("embed_tokens", "lm_head", "visual", "mtp", "conv1d")
SCOPES = ("full", "lora", "lora_fa", "fura_s", "fura_s_rel", "fura_s_rand", "unilora", "steer")
UNI_TID = 0x5EED_0F_0E_C7  # tensor id of the shared Uni-LoRA vector


def _tid(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little")


def _find_model(worker):
    if hasattr(worker, "model_runner") and hasattr(worker.model_runner, "model"):
        return worker.model_runner.model
    return worker.gpu_model_runner.model


def _gen(device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)


def _haar(n_blk: int, n: int, k: int, g: torch.Generator, device) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(n_blk, n, k, generator=g, device=device))
    return q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)


class _UniBank:
    """The single Uni-LoRA vector and its fixed index assignment over all LoRA entries of the model."""

    def __init__(self, d: int, total: int, init_seed: int, device, init: float = 0.02):
        rng = np.random.default_rng(init_seed)
        base, rem = divmod(total, d)
        data = np.repeat(np.arange(d), base)
        if rem:
            data = np.concatenate([data, rng.choice(d, rem, replace=False)])
        rng.shuffle(data)
        self.idx = torch.from_numpy(data).to(device)
        self.nrm = torch.tensor(1.0 / np.sqrt(np.bincount(data, minlength=d)), dtype=torch.float32, device=device)
        g = torch.Generator().manual_seed(init_seed)
        self.theta = {"theta": torch.empty(d).uniform_(-init, init, generator=g).to(device)}
        self.theta0 = {k: v.clone() for k, v in self.theta.items()}
        self.ptr = 0

    def take(self, n: int) -> torch.Tensor:
        out = self.idx[self.ptr:self.ptr + n]
        self.ptr += n
        return out

    def noise(self, seed: int) -> Dict[str, torch.Tensor]:
        v = self.theta["theta"]
        return {"theta": torch.randn(v.shape, generator=_gen(v.device, seed ^ UNI_TID), device=v.device)}


class _Weight:
    def __init__(self, name: str, p: torch.nn.Parameter, scope: str, rank: int, init_seed: int,
                 bank: Optional[_UniBank] = None, module: Optional[torch.nn.Module] = None):
        self.name, self.p, self.scope, self.tid, self.bank = name, p, scope, _tid(name), bank
        dev = p.device
        out_f, in_f = p.shape
        self.w0 = p.data.clone()
        self.theta: Dict[str, torch.Tensor] = {}
        self.D: Dict[str, torch.Tensor] = {}  # per-coordinate noise scale (FuRA-S's D); absent = 1
        g = _gen(dev, init_seed ^ self.tid)
        if scope == "full":
            self.theta["D"] = torch.zeros(out_f, in_f, device=dev)
        elif scope in ("lora", "lora_fa"):
            bound = 1.0 / math.sqrt(in_f)
            a0 = (torch.rand(rank, in_f, generator=g, device=dev) * 2 - 1) * bound
            self.theta["B"] = torch.zeros(out_f, rank, device=dev)
            if scope == "lora":
                self.theta["A"] = a0
            else:
                self.a0 = a0
        elif scope == "unilora":
            self.r = rank
            self.ia = bank.take(rank * in_f).view(rank, in_f)
            self.ib = bank.take(out_f * rank).view(out_f, rank)
        elif scope == "steer":
            self.theta["b"] = torch.zeros(out_f, device=dev)
            self.bias = torch.zeros(out_f, device=dev, dtype=p.dtype)
            self.hook = module.register_forward_hook(self._hook)
        else:
            n_blk, b = closest_factor_pair(in_f)
            self.n_blk, self.b = n_blk, b
            if scope in ("fura_s", "fura_s_rel"):
                wb = p.data.float().reshape(out_f, n_blk, b).permute(1, 0, 2)
                u, s, vh = torch.linalg.svd(wb, full_matrices=False)
                if scope == "fura_s_rel":
                    self.D["c"] = s.abs()
            else:
                k = min(out_f, b)
                u = _haar(n_blk, out_f, k, g, dev)
                vh = _haar(n_blk, b, k, g, dev).transpose(1, 2)
            self.U = u.to(p.dtype).contiguous()
            self.Vh = vh.float().contiguous()
            self.theta["c"] = torch.zeros(n_blk, u.shape[-1], device=dev)
        self.theta0 = {k: v.clone() for k, v in self.theta.items()}

    def _hook(self, mod, inp, out):
        if isinstance(out, tuple):
            return (out[0] + self.bias,) + tuple(out[1:])
        return out + self.bias

    def dim(self) -> int:
        return sum(v.numel() for v in self.theta.values())

    def noise(self, seed: int) -> Dict[str, torch.Tensor]:
        g = _gen(self.p.device, seed ^ self.tid)
        eps = {k: torch.randn(v.shape, generator=g, device=v.device) for k, v in self.theta.items()}
        return {k: e * self.D[k] if k in self.D else e for k, e in eps.items()}

    def phi(self, th, glob=None) -> Optional[torch.Tensor]:
        if self.scope == "full":
            return th["D"]
        if self.scope == "lora":
            return th["B"] @ th["A"]
        if self.scope == "lora_fa":
            return th["B"] @ self.a0
        if self.scope == "unilora":
            tn = glob["theta"] * self.bank.nrm
            return tn[self.ib] @ tn[self.ia]
        if self.scope == "steer":
            return None
        w = torch.bmm(self.U.float() * th["c"].unsqueeze(1), self.Vh)  # (n_blk, out, b)
        return w.permute(1, 0, 2).reshape(self.p.shape)

    def render(self, th, glob=None) -> None:
        if self.scope == "steer":
            self.bias.copy_(th["b"].to(self.bias.dtype))
            return
        self.p.data.copy_((self.w0.float() + self.phi(th, glob)).to(self.p.dtype))

    def stats(self, glob=None) -> Dict[str, float]:
        out = {"w0_fro2": float(self.w0.float().pow(2).sum())}
        ph = self.phi(self.theta, glob)
        if ph is not None:
            out["phi_fro2"] = float(ph.pow(2).sum())
        for k, v in self.theta.items():
            out[f"{k}_drift2"] = float((v - self.theta0[k]).pow(2).sum())
            out[f"{k}0_fro2"] = float(self.theta0[k].pow(2).sum())
        if self.scope == "lora":
            # Delta W = B A0 + B (A - A0): the second term comes from A's (signal-free at B=0) walk.
            B, A, A0 = self.theta["B"], self.theta["A"], self.theta0["A"]
            out["BA0_fro2"] = float((B @ A0).pow(2).sum())
            out["BdA_fro2"] = float((B @ (A - A0)).pow(2).sum())
        return out


def _angles(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Principal angles (degrees) between the column spans of orthonormal a and b."""
    c = torch.linalg.svdvals(a.T @ b).clamp(max=1.0)
    return torch.rad2deg(torch.arccos(c))


class WorkerExtension:
    @torch.no_grad()
    def pes_init(self, scope: str, rank: int = 16, init_seed: int = 0, uni_d: int = 313344) -> dict:
        assert scope in SCOPES, scope
        model = _find_model(self)
        params = [(n, p) for n, p in model.named_parameters()
                  if p.ndim == 2 and ".layers." in n and not any(s in n for s in SKIP)]
        self._scope, self._bank = scope, None
        if scope == "unilora":
            total = sum(rank * (p.shape[0] + p.shape[1]) for _, p in params)
            self._bank = _UniBank(uni_d, total, init_seed, params[0][1].device)
        mods = dict(model.named_modules())
        self._pes: List[_Weight] = []
        for name, p in params:
            self._pes.append(_Weight(name, p, scope, rank, init_seed, self._bank, mods.get(name.rsplit(".", 1)[0])))
        if self._bank is not None:
            assert self._bank.ptr == self._bank.idx.numel()
            self._render(self._bank.theta)  # Uni-LoRA starts at B A != 0 (theta0 != 0), as the official code
        torch.cuda.empty_cache()
        dim = sum(w.dim() for w in self._pes) + (self._bank.theta["theta"].numel() if self._bank else 0)
        return {"weights": len(self._pes), "dim": dim, "names": [w.name for w in self._pes][:12]}

    # ---- state helpers: per-weight theta, plus the shared Uni-LoRA vector
    def _render(self, glob=None, per=None) -> None:
        for i, w in enumerate(self._pes):
            w.render(per[i] if per is not None else w.theta, glob)

    def _glob_noise(self, seed):
        return self._bank.noise(seed) if self._bank else None

    @torch.no_grad()
    def pes_perturb(self, seed: int, sigma: float) -> None:
        glob = None
        if self._bank:
            e = self._bank.noise(seed)
            glob = {"theta": self._bank.theta["theta"] + sigma * e["theta"]}
        for w in self._pes:
            if w.theta:
                eps = w.noise(seed)
                w.render({k: v + sigma * eps[k] for k, v in w.theta.items()}, glob)
            else:
                w.render(w.theta, glob)

    @torch.no_grad()
    def pes_restore(self) -> None:
        self._render(self._bank.theta if self._bank else None)

    @torch.no_grad()
    def pes_update(self, seeds: Sequence[int], coeffs: Sequence[float]) -> None:
        for s, c in zip(seeds, coeffs):
            if self._bank:
                self._bank.theta["theta"].add_(self._bank.noise(s)["theta"], alpha=float(c))
            for w in self._pes:
                if w.theta:
                    for k, e in w.noise(s).items():
                        w.theta[k].add_(e, alpha=float(c))
        self.pes_restore()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def pes_stats(self) -> dict:
        glob = self._bank.theta if self._bank else None
        tot: Dict[str, float] = {}
        for w in self._pes:
            for k, v in w.stats(glob).items():
                tot[k] = tot.get(k, 0.0) + v
        if self._bank:
            t, t0 = self._bank.theta["theta"], self._bank.theta0["theta"]
            tot["theta_drift2"], tot["theta0_fro2"] = float((t - t0).pow(2).sum()), float(t0.pow(2).sum())
        return tot

    @torch.no_grad()
    def pes_lora_decomp(self, seed: int, sigma: float) -> dict:
        """Bilinear perturbation dW = sigma (eB A + B eA) + sigma^2 eB eA: Frobenius energy of each part."""
        first = second = 0.0
        ge = self._glob_noise(seed)
        for w in self._pes:
            if w.scope == "unilora":
                tn, en = self._bank.theta["theta"] * self._bank.nrm, ge["theta"] * self._bank.nrm
                A, B, eA, eB = tn[w.ia], tn[w.ib], en[w.ia], en[w.ib]
                first += float((sigma * (eB @ A + B @ eA)).pow(2).sum())
                second += float((sigma * sigma * eB @ eA).pow(2).sum())
                continue
            eps = w.noise(seed)
            A = w.theta.get("A", getattr(w, "a0", None))
            lin = sigma * eps["B"] @ A
            if "A" in eps:
                lin += sigma * w.theta["B"] @ eps["A"]
                second += float((sigma * sigma * eps["B"] @ eps["A"]).pow(2).sum())
            first += float(lin.pow(2).sum())
        return {"first_order_fro2": first, "second_order_fro2": second}

    # ---- RLVR-dynamics diagnostics of the current (trained) weights against W0 (ISO / FuRA-S Sec. 4 metrics)
    @torch.no_grad()
    def pes_dynamics(self, top: int = 16) -> dict:
        rows = []
        if self._scope == "steer":
            for w in self._pes:
                b = w.theta["b"]
                rows.append({"name": w.name, "b_norm": float(b.norm()),
                             "out_norm_ratio": float(b.norm() / w.w0.float().norm())})
            return {"rows": rows}
        for w in self._pes:
            glob = self._bank.theta if self._bank else None
            W0 = w.w0.float()
            dW = w.phi(w.theta, glob)  # fp32, free of the bf16 rounding of the rendered weight
            W1 = W0 + dW
            dn2 = float(dW.pow(2).sum())
            if dn2 == 0:
                continue
            U0, S0, V0h = torch.linalg.svd(W0, full_matrices=False)
            U1, S1, V1h = torch.linalg.svd(W1, full_matrices=False)
            m, n = W0.shape
            q = min(m, n)
            diag = ((U0.T @ dW) * V0h).sum(1)  # diag(U0^T dW V0)
            sv = torch.linalg.svdvals(dW)
            p_ = sv / sv.sum()
            k = min(top, q)
            rows.append({
                "name": w.name, "dW_rel": math.sqrt(dn2) / float(W0.norm()),
                "rho_sigma": float((S1 - S0).norm()) / math.sqrt(dn2),
                "kappa": m * n / q * float(diag.pow(2).sum()) / dn2,
                "rotU_top": float(_angles(U0[:, :k], U1[:, :k]).mean()),
                "rotV_top": float(_angles(V0h[:k].T, V1h[:k].T).mean()),
                "erank": float(torch.exp(-(p_ * torch.log(p_.clamp_min(1e-30))).sum())),
                "colspace": float((U0.T @ dW).pow(2).sum()) / dn2,  # = 1 when m <= n (U0 spans R^m)
                "rowspace": float((dW @ V0h.T).pow(2).sum()) / dn2,
            })
        return {"rows": rows}

    @torch.no_grad()
    def pes_spectral_swap(self, mode: str) -> None:
        """ISO's restoration test. base_spectrum: U1 diag(S0) V1^T (learned frames, base spectrum);
        base_frames: U0 diag(S1) V0^T (base frames, learned spectrum). pes_restore() undoes it."""
        for w in self._pes:
            if w.scope == "steer":
                continue
            W0 = w.w0.float()
            W1 = W0 + w.phi(w.theta, self._bank.theta if self._bank else None)
            U0, S0, V0h = torch.linalg.svd(W0, full_matrices=False)
            U1, S1, V1h = torch.linalg.svd(W1, full_matrices=False)
            W = (U1 * S0) @ V1h if mode == "base_spectrum" else (U0 * S1) @ V0h
            w.p.data.copy_(W.to(w.p.dtype))

    @torch.no_grad()
    def pes_perturb_energy(self, seed: int, sigma: float) -> dict:
        """Weight-space size of one perturbation: sum ||Phi(theta + sigma eps) - Phi(theta)||_F^2."""
        glob = glob_p = None
        if self._bank:
            glob = self._bank.theta
            glob_p = {"theta": glob["theta"] + sigma * self._bank.noise(seed)["theta"]}
        tot = w0 = 0.0
        for w in self._pes:
            if w.scope == "steer":
                continue
            eps = w.noise(seed) if w.theta else {}
            dW = w.phi({k: v + sigma * eps[k] for k, v in w.theta.items()}, glob_p) - w.phi(w.theta, glob)
            tot += float(dW.pow(2).sum())
            w0 += float(w.w0.float().pow(2).sum())
        return {"dW_fro2": tot, "w0_fro2": w0}
