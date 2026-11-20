"""Tensor-parallel version of the RotSlots path: vLLM with ``tensor_parallel_size > 1``, ``RotSlots`` installed in
every worker (``worker_extension_cls``), slots routed per token inside each worker.

Under TP each worker holds a shard of every linear. ``RotSlots`` is built on the local shard shapes, so
* column-parallel layers (qkv, gate_up): the input rotation acts on the replicated input; the output rotation
  pairs local rows only (fused q/k/v and gate/up segments are the global ``output_sizes`` divided by tp);
* row-parallel layers (o_proj, down_proj): the input rotation pairs local input columns only; the output rotation
  acts on the partial sums before the all-reduce, which is exact because the rotation is linear and identical on
  every rank (same layer name -> same pairing and same seeded angles on every rank).
A member is therefore still a block-Givens rotation of every weight (pairs restricted to shards), but not the same
member as with tp = 1.

  eng = TPSlotEngine(model, tp=4, mode="both", stages=2)
  eng.rot.set_slots([(None, seed, sigma), ...]);  agent.rollout(eng, ...)   # eng.llm / eng.tokenizer as SlotEngine
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Sequence

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
_REPO = str(Path(__file__).resolve().parents[2])


class RotWorkerExt:
    """Mixed into vLLM's GPU worker; methods are called through ``LLM.collective_rpc``."""

    def rot_install(self, mode: str, stages: int):
        if _REPO not in sys.path:
            sys.path.insert(0, _REPO)
        from algorithms.es.rot_slots import RotSlots

        runner = self.model_runner
        self._rot = RotSlots(runner.model, mode=mode, stages=stages)
        forward = runner._model_forward
        ext = self

        def model_forward(*args, **kwargs):
            inputs = kwargs.get("input_ids")
            if inputs is None:
                inputs = kwargs.get("inputs_embeds")
            slots = ext._slots_for_batch(inputs.shape[0], inputs.device)
            if slots is None and getattr(ext._rot, "route_all", False):  # slot 0 is a trained centre, not the base
                import torch
                slots = torch.zeros(inputs.shape[0], dtype=torch.int64, device=inputs.device)
            ext._rot.set_token_slots(slots)
            try:
                return forward(*args, **kwargs)
            finally:
                ext._rot.set_token_slots(None)

        runner._model_forward = model_forward
        return {**self._rot.info(), "rank": int(getattr(self, "rank", -1)),
                "angles": int(sum(L.pin[0][0].numel() + L.pout[0][0].numel() for L in self._rot.layers) * stages)}

    def _slots_for_batch(self, num_tokens: int, device):
        import torch

        runner = self.model_runner
        table, any_slot = [], False
        for req_id in runner.input_batch.req_ids:
            request = runner.requests.get(req_id) if req_id is not None else None
            extra = getattr(getattr(request, "sampling_params", None), "extra_args", None) or {}
            slot = int(extra.get("slot", 0))
            any_slot = any_slot or slot != 0
            table.append(slot)
        if not any_slot:
            return None
        table = torch.tensor(table, dtype=torch.int64, device=device)
        n_real = min(num_tokens, int(runner.query_start_loc.np[len(runner.input_batch.req_ids)]))
        slots = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        slots[:n_real] = table[runner.req_indices.gpu[:n_real]]
        return slots

    def rot_set_slots(self, entries, center_path: Optional[str] = None, use_center: bool = False):
        """Slots 1.. = members around the centre: the worker's ES centre (use_center), a saved centre
        (center_path, one file per rank), or the base model."""
        import torch

        center = None
        if use_center:
            center = [{k: v.detach() for k, v in c.items()} for c in self._center]
        elif center_path:
            center = torch.load(f"{center_path}.rank{self.rank}", map_location=self.model_runner.device)
        self._rot.set_slots([tuple(e) for e in entries], center=center)
        return True

    # ---- ES on the centre angles, each rank on its own shard (same math as rl_train.py --arm es)
    def rot_es_init(self, lr: float):
        import torch

        self._center = [{k: v.requires_grad_(False) for k, v in c.items()} for c in self._rot.zero_center()]
        self._opt = torch.optim.Adam([t for c in self._center for t in c.values()], lr=lr)
        return True

    def rot_es_update(self, members, adv, sig_s: float):
        """members[i] = (seed, signed sigma) of slot i + 1, adv[i] its advantage; ascent on E[R]."""
        import torch

        rs, N = self._rot, len(members)
        self._opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for li, L in enumerate(rs.layers):
                for side in rs.sides():
                    stg = [0] if side == "add" else range(rs.stages)
                    acc = None
                    for (sd, sg), a in zip(members, adv):
                        if a == 0:
                            continue
                        nz = torch.stack([rs.member_noise(L, side, st, sd, sg) for st in stg])
                        nz = nz[0] if side == "add" else nz
                        acc = nz * a if acc is None else acc + nz * a
                    if acc is not None:
                        self._center[li][side].grad = -acc / (N * sig_s ** 2)
        self._opt.step()
        rms = torch.cat([t.flatten() for c in self._center for t in c.values()]).pow(2).mean().sqrt()
        return float(rms)

    def rot_save_center(self, path: str):
        import torch

        torch.save([{k: v.detach().cpu() for k, v in c.items()} for c in self._center], f"{path}.rank{self.rank}")
        return True

    def rot_clear(self):
        self._rot.clear_slots()
        return True


class _RotProxy:
    def __init__(self, llm):
        self.llm = llm

    def set_slots(self, entries: Sequence[tuple], center_path: Optional[str] = None, use_center: bool = False):
        self.llm.collective_rpc("rot_set_slots", args=([list(e) for e in entries], center_path, use_center))

    def es_init(self, lr: float):
        self.llm.collective_rpc("rot_es_init", args=(lr,))

    def es_update(self, members, adv, sig_s: float) -> float:
        return self.llm.collective_rpc("rot_es_update", args=([list(m) for m in members], [float(a) for a in adv],
                                                              float(sig_s)))[0]

    def save_center(self, path: str):
        self.llm.collective_rpc("rot_save_center", args=(path,))

    def clear_slots(self):
        self.llm.collective_rpc("rot_clear")


class TPSlotEngine:
    def __init__(self, model_path: str, *, tp: int, mode: str = "both", stages: int = 2, max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.85, max_num_batched_tokens: int = 32768, max_num_seqs: int = 256,
                 seed: int = 0, **kw):
        os.environ["PYTHONPATH"] = _REPO + os.pathsep + os.environ.get("PYTHONPATH", "")
        from vllm import LLM

        self.model_path = model_path
        self.llm = LLM(model_path, tensor_parallel_size=tp, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, enforce_eager=True,
                       enable_prefix_caching=False, max_num_batched_tokens=max_num_batched_tokens,
                       max_num_seqs=max_num_seqs, seed=seed,
                       worker_extension_cls="algorithms.es.rot_tp.RotWorkerExt", **kw)
        self.tokenizer = self.llm.get_tokenizer()
        self.lora_requests = None
        self.info = self.llm.collective_rpc("rot_install", args=(mode, stages))
        self.rot = _RotProxy(self.llm)

    def _lora_for(self, slots):
        return None
