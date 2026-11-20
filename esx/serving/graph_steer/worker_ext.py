"""Worker extension of the graph-safe steering server (esx.serving.graph_steer.patch). Importing this module applies
the patch, and vLLM imports the extension class in every worker before building the model. So launching with

    ESX_STEER_LAYERS=8,16,24 vllm serve <model> --worker-extension-cls esx.serving.graph_steer.worker_ext.SteerWorkerExtension

gives steering under torch.compile and CUDA graphs, with no plugin install. Calls through POST /collective_rpc:

- **esx_steer_info():** steered layers, rows, d, dtype, the first row norms, and the patched classes.
- **esx_steer_load(path):** an npz with one array (rows, d) per layer (key str(L)). Rows beyond the file's, and layers
  it omits, are zeroed. Every rank loads the same file, so tensor-parallel ranks stay identical.
- **esx_steer_zero():** zeroes every table.
- **esx_steer_norms(reset):** mean output norm per steered layer. Measure mode only (ESX_STEER_MEASURE=1, eager).
- **esx_steer_debug():** where the slot buffer lives and the last step's slots.

ES with this backend keeps the head and the noise on the client (esx.serving.graph_steer.client): row 0 is the head,
row k is a member.
"""
from __future__ import annotations

import torch

from esx.serving.graph_steer import patch as _patch

_patch.apply()


class SteerWorkerExtension:
    def _esx_layers(self):
        return {m.layer_idx: m for m in self.model_runner.model.modules() if getattr(m, "esx_steered", False)}

    def esx_steer_info(self):
        layers = self._esx_layers()
        return {"classes": _patch.STATE.get("classes", []), "slots": _patch.SLOTS,
                "layers": {L: {"rows": int(m.esx_steer_table.shape[0]), "d": int(m.esx_steer_table.shape[1]),
                               "dtype": str(m.esx_steer_table.dtype), "device": str(m.esx_steer_table.device),
                               "norms": m.esx_steer_table.float().norm(dim=1)[:4].tolist()}
                           for L, m in sorted(layers.items())}}

    def esx_steer_load(self, path: str):
        import numpy as np
        with np.load(path) as z:
            tables = {int(k): torch.from_numpy(z[k]) for k in z.files}
        layers = self._esx_layers()
        unknown = set(tables) - set(layers)
        assert not unknown, f"layers {sorted(unknown)} are not steered (ESX_STEER_LAYERS)"
        for L, m in layers.items():
            m.esx_steer_table.zero_()
            t = tables.get(L)
            if t is not None:
                assert t.shape[0] <= m.esx_steer_table.shape[0] and t.shape[1] == m.esx_steer_table.shape[1], t.shape
                m.esx_steer_table[: t.shape[0]].copy_(t.to(m.esx_steer_table))
        torch.cuda.synchronize()
        return {"ok": True, "layers": sorted(layers)}

    def esx_steer_zero(self):
        for m in self._esx_layers().values():
            m.esx_steer_table.zero_()
        torch.cuda.synchronize()
        return {"ok": True}

    def esx_steer_norms(self, reset: bool = False):
        acc = _patch.STATE.setdefault("norms", {})
        out = {L: {"mean_norm": s / max(n, 1), "tokens": n} for L, (s, n) in sorted(acc.items())}
        if reset:
            acc.clear()
        return out

    def esx_steer_debug(self):
        idx = _patch.STATE.get("idx")
        return {"idx": None if idx is None else {"device": str(idx.device), "head": idx[:12].tolist()},
                "last_slots": _patch.STATE.get("last_slots"), "calls": _patch.STATE.get("calls", 0)}
