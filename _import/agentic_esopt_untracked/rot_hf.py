"""HF-side, differentiable twin of ``rot_slots.RotSlots`` (mode in / out / both), for backprop on the angles.

vLLM merges HF linears (q/k/v -> qkv_proj, gate/up -> gate_up_proj, in_proj_qkv/in_proj_z -> in_proj_qkvz). The
RotSlots layout is defined on the merged layer; every HF member applies the shared input rotation and the part of the
output rotation that falls on its rows (pairs never cross output segments, which coincide with member boundaries;
this is asserted). Angles = trainable centre + optional fixed member noise (for behaviour log-probs).
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional

import torch

GROUPS = {  # HF suffix -> vLLM merged suffix, in vLLM's row-stacking order
    "mlp.gate_proj": "mlp.gate_up_proj", "mlp.up_proj": "mlp.gate_up_proj", "mlp.down_proj": "mlp.down_proj",
    "self_attn.q_proj": "self_attn.qkv_proj", "self_attn.k_proj": "self_attn.qkv_proj",
    "self_attn.v_proj": "self_attn.qkv_proj", "self_attn.o_proj": "self_attn.o_proj",
    "linear_attn.in_proj_qkv": "linear_attn.in_proj_qkvz", "linear_attn.in_proj_z": "linear_attn.in_proj_qkvz",
    "linear_attn.out_proj": "linear_attn.out_proj",
}
ORDER = list(GROUPS)


def givens(x: torch.Tensor, i0: torch.Tensor, i1: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Differentiable stage: pairs (i0, i1) of the last dim rotated by theta (fp32 math, like the kernel)."""
    c, s = torch.cos(theta), torch.sin(theta)
    a, b = x[..., i0].float(), x[..., i1].float()
    y = x.clone()
    y[..., i0] = (c * a - s * b).to(x.dtype)
    y[..., i1] = (s * a + c * b).to(x.dtype)
    return y


class HFRot:
    def __init__(self, model, layout: List[dict], mode: str, stages: int, device="cuda"):
        assert mode in ("in", "out", "both")
        self.mode, self.stages = mode, stages
        self.sides = {"both": ["in", "out"], "in": ["in"], "out": ["out"]}[mode]
        pat = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.(.+)$")
        members: Dict[str, list] = defaultdict(list)
        for name, mod in model.named_modules():
            m = pat.match(name)
            if m and m.group(2) in GROUPS and isinstance(mod, torch.nn.Linear):
                members[f"language_model.model.layers.{m.group(1)}.{GROUPS[m.group(2)]}"].append((ORDER.index(m.group(2)), mod))
        self.center: List[Dict[str, torch.nn.Parameter]] = []
        self.noise: List[Optional[Dict[str, torch.Tensor]]] = []
        self.hooks = []
        for li, L in enumerate(layout):
            mods = [mod for _, mod in sorted(members[L["name"]], key=lambda t: t[0])]
            assert mods, L["name"]
            assert sum(m.weight.shape[0] for m in mods) == L["out_f"] and mods[0].weight.shape[1] == L["in_f"], L["name"]
            c = {}
            for side in self.sides:
                P = L["pin" if side == "in" else "pout"][0][0].shape[0]
                c[side] = torch.nn.Parameter(torch.zeros(stages, P, device=device))
            self.center.append(c)
            self.noise.append(None)
            pin = [(a.long().to(device), b.long().to(device)) for a, b in L["pin"]]
            off = 0
            for mod in mods:
                rows = mod.weight.shape[0]
                local = []
                for a, b in [(a.long(), b.long()) for a, b in L["pout"]]:
                    ma = (a >= off) & (a < off + rows)
                    mb = (b >= off) & (b < off + rows)
                    assert torch.equal(ma, mb), f"output pair crosses HF members in {L['name']}"
                    idx = ma.nonzero().squeeze(1)
                    local.append(((a[ma] - off).to(device), (b[mb] - off).to(device), idx.to(device)))
                off += rows
                if "in" in self.sides:
                    self.hooks.append(mod.register_forward_pre_hook(self._pre(li, pin)))
                if "out" in self.sides:
                    self.hooks.append(mod.register_forward_hook(self._post(li, local)))

    def _theta(self, li, side, st, idx=None):
        th = self.center[li][side][st]
        nz = self.noise[li]
        if nz is not None:
            th = th + nz[side][st]
        return th if idx is None else th[idx]

    def _pre(self, li, pin):
        def hook(mod, inputs):
            x = inputs[0]
            for st, (a, b) in enumerate(pin):
                x = givens(x, a, b, self._theta(li, "in", st))
            return (x,) + tuple(inputs[1:])
        return hook

    def _post(self, li, local):
        def hook(mod, inputs, y):
            for st, (a, b, idx) in enumerate(local):
                y = givens(y, a, b, self._theta(li, "out", st, idx))
            return y
        return hook

    def params(self) -> List[torch.nn.Parameter]:
        return [p for c in self.center for p in c.values()]

    def set_noise(self, noise: Optional[List[Dict[str, torch.Tensor]]]) -> None:
        """noise[li][side] = (stages, P) angle offsets (no grad), or None for the centre."""
        self.noise = [None] * len(self.center) if noise is None else noise

    def center_tensors(self) -> List[dict]:
        return [{k: v.detach().clone() for k, v in c.items()} for c in self.center]
