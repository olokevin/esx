"""LoRA-ES on vLLM's multi-LoRA (punica) path: one resident adapter slot per population member.

vLLM keeps ``max_loras`` adapters in stacked GPU buffers and routes every token
to its request's slot. Here ``n_slots`` dummy adapters (zero weights, same target
modules) are registered once and kept resident; their buffers are then written
directly: slot ``s`` holds ``state(uid) + sigma * eps(seed)`` in the model dtype.
The fp32 master state of every user lives here, never in vLLM. ``scaling``
(``alpha / r``) is folded into B exactly as vLLM does when it loads an adapter.

Noise per tensor: ``randn(shape, generator(seed ^ stable_id(f"{module}.{slice}.{A|B}")))``.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from algorithms.es.fura_native import stable_tensor_id

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                "gate_proj", "up_proj", "down_proj"]
_MASK64 = 0xFFFFFFFFFFFFFFFF


def write_zero_adapter(base_model: str, out_dir: str, rank: int, alpha: float,
                       targets: Sequence[str] = LORA_TARGETS) -> str:
    """A PEFT adapter with all-zero weights for ``targets`` of the language model."""
    if os.path.exists(os.path.join(out_dir, "adapter_config.json")):
        return out_dir
    import re

    from safetensors import safe_open
    from safetensors.torch import save_file

    # Module names and shapes straight from a dense checkpoint of the same architecture.
    dense = base_model if not os.path.exists(base_model + "-dequant") else base_model + "-dequant"
    index = os.path.join(dense, "model.safetensors.index.json")
    files = sorted(set(json.load(open(index))["weight_map"].values())) if os.path.exists(index) else ["model.safetensors"]
    pat = re.compile(r"^(model\.language_model\.layers\.\d+\..*\.(" + "|".join(targets) + r"))\.weight$")
    state = {}
    for fname in files:
        with safe_open(os.path.join(dense, fname), "pt") as f:
            for key in f.keys():
                m = pat.match(key)
                if not m:
                    continue
                out_f, in_f = f.get_slice(key).get_shape()
                state[f"base_model.model.{m.group(1)}.lora_A.weight"] = torch.zeros(rank, in_f, dtype=torch.bfloat16)
                state[f"base_model.model.{m.group(1)}.lora_B.weight"] = torch.zeros(out_f, rank, dtype=torch.bfloat16)
    if not state:
        raise RuntimeError("no LoRA target modules found")
    os.makedirs(out_dir, exist_ok=True)
    save_file(state, os.path.join(out_dir, "adapter_model.safetensors"))
    conf = {"peft_type": "LORA", "r": rank, "lora_alpha": alpha, "lora_dropout": 0.0, "bias": "none",
            "target_modules": list(targets), "base_model_name_or_path": base_model, "task_type": "CAUSAL_LM",
            "fan_in_fan_out": False, "use_rslora": False, "modules_to_save": None}
    json.dump(conf, open(os.path.join(out_dir, "adapter_config.json"), "w"), indent=2)
    return out_dir


class LoraSlots:
    def __init__(self, engine, adapter_dir: str, n_slots: int, rank: int, alpha: float):
        from vllm.lora.request import LoRARequest

        self.engine = engine
        self.rank = rank
        self.scaling = float(alpha) / float(rank)
        self.requests = [LoRARequest(f"slot{i}", i + 1, adapter_dir) for i in range(n_slots)]
        engine.lora_requests = self.requests
        # Make every slot resident: one tiny request per adapter in a single batch.
        engine.generate([(i + 1, "hi") for i in range(n_slots)], max_tokens=1)
        manager = engine.runner.lora_manager._adapter_manager
        self.index_of = {req.lora_int_id: manager.lora_index_to_id.index(req.lora_int_id) for req in self.requests}
        self.modules = []  # (name, module)
        for name, module in manager.modules.items():
            if hasattr(module, "lora_a_stacked") and "language_model" in name and "visual" not in name:
                self.modules.append((name, module))
        if not self.modules:
            raise RuntimeError("No LoRA modules found.")
        self.tensors: List[Tuple[str, int, Tuple[int, ...], str, int, int]] = []
        # (key, tensor_id, shape, kind, module_idx, slice_idx)
        for mi, (name, module) in enumerate(self.modules):
            for si, a in enumerate(module.lora_a_stacked):
                r, din = a.shape[2], a.shape[3]
                key = f"{name}.{si}.A"
                self.tensors.append((key, stable_tensor_id(key), (r, din), "A", mi, si))
            for si, b in enumerate(module.lora_b_stacked):
                dout, r = b.shape[2], b.shape[3]
                key = f"{name}.{si}.B"
                self.tensors.append((key, stable_tensor_id(key), (dout, r), "B", mi, si))
        self.device = self.modules[0][1].lora_a_stacked[0].device
        self.dtype = self.modules[0][1].lora_a_stacked[0].dtype
        self.adapters: Dict[str, List[torch.Tensor]] = {}
        self.enabled_slots = 0

    @property
    def n_params(self) -> int:
        return int(sum(math.prod(t[2]) for t in self.tensors))

    def noise(self, ti: int, seed: int) -> torch.Tensor:
        _, tid, shape, _, _, _ = self.tensors[ti]
        gen = torch.Generator(device=self.device)
        gen.manual_seed((int(seed) ^ tid) & _MASK64)
        return torch.randn(shape, generator=gen, dtype=torch.float32, device=self.device)

    def new_adapter(self, uid: str, init_seed: int = 0) -> None:
        """PEFT-style init: A ~ U(-1/sqrt(in), 1/sqrt(in)) (kaiming_uniform a=sqrt(5)), B = 0."""
        state = []
        for ti, (key, tid, shape, kind, _, _) in enumerate(self.tensors):
            if kind == "A":
                gen = torch.Generator(device=self.device)
                gen.manual_seed((int(init_seed) * 1000003 ^ tid) & _MASK64)
                bound = 1.0 / math.sqrt(shape[1])
                state.append((torch.rand(shape, generator=gen, device=self.device) * 2 - 1) * bound)
            else:
                state.append(torch.zeros(shape, device=self.device))
        self.adapters[uid] = state

    def drop_adapter(self, uid: str) -> None:
        self.adapters.pop(uid, None)

    def adapter_state(self, uid: str) -> List[torch.Tensor]:
        return [t.detach().cpu().clone() for t in self.adapters[uid]]

    def load_adapter_state(self, uid: str, state: Sequence[torch.Tensor]) -> None:
        self.adapters[uid] = [t.to(self.device).float().clone() for t in state]

    @torch.no_grad()
    def set_slots(self, entries: Sequence[Tuple[Optional[str], int, float]]) -> None:
        if len(entries) > len(self.requests):
            raise ValueError(f"{len(entries)} entries > {len(self.requests)} slots")
        for ti, (key, tid, shape, kind, mi, si) in enumerate(self.tensors):
            module = self.modules[mi][1]
            buf = (module.lora_a_stacked if kind == "A" else module.lora_b_stacked)[si]
            cache: Dict[int, torch.Tensor] = {}
            for k, (uid, seed, sigma) in enumerate(entries):
                value = self.adapters[uid][ti] if uid is not None else torch.zeros(shape, device=self.device)
                if sigma:
                    if seed not in cache:
                        cache[seed] = self.noise(ti, seed)
                    value = value + float(sigma) * cache[seed]
                if kind == "B":
                    value = value * self.scaling
                idx = self.index_of[self.requests[k].lora_int_id]
                buf[idx, 0, : shape[0], : shape[1]].copy_(value.to(self.dtype))
        self.enabled_slots = len(entries)

    def clear_slots(self) -> None:
        self.enabled_slots = 0

    @torch.no_grad()
    def update(self, uid: str, seeds: Sequence[int], weights: Sequence[float], eta: float) -> None:
        scale = float(eta) / float(len(seeds))
        for ti in range(len(self.tensors)):
            total = torch.zeros_like(self.adapters[uid][ti])
            for seed, w in zip(seeds, weights):
                total.add_(self.noise(ti, seed), alpha=scale * float(w))
            self.adapters[uid][ti].add_(total)

    def info(self) -> Dict[str, object]:
        return {"n_modules": len(self.modules), "n_tensors": len(self.tensors), "n_params": self.n_params,
                "rank": self.rank, "scaling": self.scaling, "n_slots": len(self.requests)}
