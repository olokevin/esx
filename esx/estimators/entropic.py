"""Entropic weights of TTT-Discover (Yuksekgonul et al. 2026, arXiv 2601.16175 v2, App. A.1) for a batch of rewards.

q_beta(n) = exp(beta r_n) / sum_m exp(beta r_m), with beta >= 0 chosen by bisection so that KL(q_beta || uniform) =
gamma (ln 2 in the paper). If even beta -> inf (q uniform over the maximisers) stays below gamma, q is uniform over the
maximisers. The soft value V = (1/beta) log mean exp(beta r) lies between mean (beta = 0) and max (beta -> inf).
Both are invariant to r -> a r + b (a > 0) up to the same affine map of V.
"""
from __future__ import annotations

import math

import numpy as np

LN2 = math.log(2.0)


def _q(r: np.ndarray, beta: float) -> np.ndarray:
    z = beta * (r - r.max())
    e = np.exp(z)
    return e / e.sum()


def _kl_uniform(q: np.ndarray) -> float:
    n = len(q)
    q = q[q > 0]
    return float(np.sum(q * np.log(q * n)))


def entropic(r, gamma: float = LN2, iters: int = 60):
    """Returns (q, beta, soft value). beta = inf when q is uniform over the maximisers."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n == 0:
        raise ValueError("empty reward batch")
    span = r.max() - r.min()
    if span == 0:
        return np.full(n, 1.0 / n), 0.0, float(r.mean())
    top = r == r.max()
    kl_max = math.log(n / top.sum())
    if kl_max <= gamma:
        q = top / top.sum()
        return q.astype(float), math.inf, float(r.max())
    lo, hi = 0.0, 1.0 / span
    while _kl_uniform(_q(r, hi)) < gamma:
        hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _kl_uniform(_q(r, mid)) < gamma:
            lo = mid
        else:
            hi = mid
    beta = 0.5 * (lo + hi)
    q = _q(r, beta)
    v = float(r.max() + math.log(np.mean(np.exp(beta * (r - r.max())))) / beta)
    return q, beta, v
