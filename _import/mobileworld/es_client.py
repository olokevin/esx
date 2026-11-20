"""HTTP client for a ``vllm serve`` instance that loads an ES worker extension (workers/fura_worker.py).

The server must run with ``VLLM_SERVER_DEV_MODE=1`` so that ``POST /collective_rpc`` is exposed (launch/start_es_server.sh).
Every worker method is reachable as ``EsServer(url).call("<method>", *args)``.

ES runs log one "gen" record per generation with the update's (seeds, coeffs); updates are linear in the trained
coordinates, so ``replay`` rebuilds a center exactly and ``unreplay`` restores the base weights.
"""
from __future__ import annotations

import json
from typing import Iterable, List, Optional

import requests

# pes_init arguments used by every MobileWorld run (rank and uni_d only matter for the low-rank / uni scopes)
SCOPE, RANK, INIT_SEED, UNI_D = "fura_s_paper", 4, 0, 344064


class EsServer:
    def __init__(self, url: str, timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def call(self, method: str, *args):
        r = requests.post(f"{self.url}/collective_rpc", json={"method": method, "args": list(args)},
                          timeout=self.timeout)
        r.raise_for_status()
        if not r.content:
            return None
        return r.json()["results"][0]

    def init(self, scope: str = SCOPE, rank: int = RANK, seed: int = INIT_SEED, uni_d: int = UNI_D) -> dict:
        return self.call("pes_init", scope, rank, seed, uni_d)

    def perturb(self, seed: int, sigma: float) -> None:
        self.call("pes_perturb", seed, sigma)

    def restore(self) -> None:
        self.call("pes_restore")

    def update(self, seeds, coeffs) -> None:
        self.call("pes_update", list(seeds), list(coeffs))

    def stats(self) -> dict:
        return self.call("pes_stats")

    def healthy(self) -> bool:
        try:
            return requests.get(f"{self.url}/health", timeout=5).ok
        except requests.RequestException:
            return False


def load_gens(history: str, upto: Optional[int] = None) -> List[dict]:
    """The "gen" records of an ES history.jsonl (optionally only generations <= upto)."""
    with open(history) as f:
        gens = [r for r in map(json.loads, f) if r.get("kind") == "gen"]
    return [r for r in gens if upto is None or r["gen"] <= upto]


def replay(servers: Iterable[EsServer], gens: List[dict]) -> None:
    for s in servers:
        for r in gens:
            s.update(r["seeds"], r["coeffs"])


def unreplay(servers: Iterable[EsServer], gens: List[dict]) -> None:
    """Undo ``replay``: negated coeffs in reverse order (exact, the updates are linear)."""
    for s in servers:
        for r in reversed(gens):
            s.update(r["seeds"], [-c for c in r["coeffs"]])


def flat_update(gens: List[dict]) -> tuple:
    """All generations of a run as one (seeds, coeffs) list, in order."""
    return [s for r in gens for s in r["seeds"]], [c for r in gens for c in r["coeffs"]]
