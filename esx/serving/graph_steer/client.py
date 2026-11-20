"""Client side of graph-safe steering ES (esx.serving.graph_steer): the head theta_L and the member noise live here.
The server only holds tables: row 0 is the head, and row k is a member theta + sign * sigma * eps(seed). These are
written as an npz on the server host's filesystem and loaded on every rank.

**Noise.**
- eps_L(seed) = scale_L * z_L(seed), with z_L(seed) = numpy default_rng([seed, L]).standard_normal(d) in float32.
- scale_L is the layer's residual RMS, so sigma is relative to the residual stream, as in the E4 spec steer.
- Update: theta_L += sum_j c_j * eps_L(seed_j). This is the same algebra as `pes_update`, so the ES drivers are
  unchanged.

After any table change, callers must reset the prefix cache and use `cache_salt` = (version, slot); `version` counts
the table writes. HTTP only: run the driver on the server host, since the npz path must be visible to the workers.
"""
from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import requests


class GraphSteer:
    def __init__(self, url: str, layers, d: int, scale: dict, tmpdir: str = None, slots: int = 65):
        self.url = url.rstrip("/")
        self.layers = list(layers)
        self.d = d
        self.scale = {int(k): float(v) for k, v in scale.items()}
        self.theta = {L: np.zeros(d, np.float32) for L in self.layers}
        self.slots = slots
        self.version = 0
        self.tmp = tmpdir or tempfile.mkdtemp(prefix="esx_steer_")
        self._z: dict = {}

    def call(self, method: str, *args):
        r = requests.post(f"{self.url}/collective_rpc", json={"method": method, "args": list(args)}, timeout=600)
        r.raise_for_status()
        return r.json()

    def z(self, seed: int, L: int) -> np.ndarray:
        k = (int(seed), L)
        if k not in self._z:
            if len(self._z) > 4096:
                self._z.clear()
            self._z[k] = np.random.default_rng([int(seed), L]).standard_normal(self.d).astype(np.float32)
        return self._z[k]

    def eps(self, seed: int, L: int) -> np.ndarray:
        return self.scale.get(L, 1.0) * self.z(seed, L)

    def write(self, members=()) -> None:
        """Load the tables: row 0 = head, row k = head + sign * sigma * eps(seed) for members [(k, seed, sign, sigma)]."""
        rows = 1 + max([k for k, *_ in members], default=0)
        assert rows <= self.slots, f"{rows} rows > {self.slots} slots"
        tab = {}
        for L in self.layers:
            t = np.zeros((rows, self.d), np.float32)
            t[:] = self.theta[L]
            for k, seed, sign, sigma in members:
                t[k] = self.theta[L] + sign * sigma * self.eps(seed, L)
            tab[str(L)] = t
        path = os.path.join(self.tmp, f"t{self.version + 1}.npz")
        np.savez(path, **tab)
        self.call("esx_steer_load", path)
        self.version += 1
        try:
            requests.post(f"{self.url}/reset_prefix_cache", timeout=60)
        except Exception:  # noqa: BLE001
            pass
        try:
            os.remove(path)
        except OSError:
            pass

    def update(self, seeds, coeffs) -> None:
        for L in self.layers:
            for s, c in zip(seeds, coeffs):
                self.theta[L] += np.float32(c) * self.eps(s, L)
        self.write()

    def salt(self, slot: int = 0) -> str:
        return f"g{self.version}-s{int(slot)}"

    def state(self) -> dict:
        return {"theta": {str(L): v.tolist() for L, v in self.theta.items()}, "version": self.version,
                "time": time.time()}
