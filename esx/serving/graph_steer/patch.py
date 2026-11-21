"""Per-request residual-stream steering for vLLM that stays inside torch.compile and CUDA graphs.

Ported from HarnessES `methods/steer_es/hes_steer/patch.py` (Gemma 4, vLLM 0.30). The patched decoder class is now
configurable, and the default covers Qwen3.5 / Qwen3.6 / Qwen3.8 (`Qwen3_5DecoderLayer`), Qwen3-Next and Gemma 4.

**Mechanism.**
- Each steered decoder layer L (env ESX_STEER_LAYERS, e.g. "8,16,24") owns a table T_L of ESX_STEER_SLOTS rows
  (d = hidden size).
- The layer adds T_L[slot(token)] to its output hidden_states, i.e. to the residual stream after layer L.
- A request picks its row with `vllm_xargs {"steer_slot": k}` (SamplingParams.extra_args). Requests without it use
  row 0, which by convention holds the ES head.
- Rows are written through the worker extension (esx.serving.graph_steer.worker_ext) via POST /collective_rpc.

**Graph-safe.**
- The add is an opaque custom op (esx::steer_add) that reads the tables and the token -> slot buffer from STATE, so
  torch.compile cannot fold them as constants.
- CUDA graphs capture the op's kernels on these fixed buffers. The buffers are filled in place: the slot buffer from
  the host side of every step, in the runner's prepare_inputs.
- Padded graph tokens read stale slots; their outputs are discarded.
- With every row 0 the output is bitwise the unpatched model's (x + 0 = x).

**Where the patch is applied.** The patch runs at import of this module. The worker extension imports it, and vLLM
imports the extension class in every worker process before the model is constructed, so the decoder classes and the
runner are patched before load_model, profiling and graph capture. No vLLM plugin registration or install is needed.

**Prefix cache.** Requests of different slots must not share prefix-cache blocks. Callers set
`cache_salt` = (table version, slot) and reset the prefix cache after every table change.

**Env.**
- ESX_STEER_LAYERS (empty = patch disabled).
- ESX_STEER_SLOTS (65).
- ESX_STEER_MAX_TOKENS (65536; >= max_num_batched_tokens).
- ESX_STEER_CLASSES (comma list of "module:Class"; default below).
- ESX_STEER_MEASURE=1 (eager only): accumulate the output norm of every steered layer.
- ESX_STEER_CHECK=1 (tests; syncs every step): audit the token -> slot buffer of every step. (a) Per request, the slot
  read through the runner's req_index indirection equals the slot of that request's own SamplingParams; (b) the
  token layout assumed by the patch (batch order, num_scheduled_tokens per request) gives the runner's own positions
  (as HarnessES's HES_STEER_CHECK). Counts are in esx_steer_debug (check_steps, check_bad_slot, check_bad_pos).
"""
from __future__ import annotations

import importlib
import os

import numpy as np
import torch

LAYERS = sorted({int(x) for x in os.environ.get("ESX_STEER_LAYERS", "").split(",") if x.strip()})
SLOTS = int(os.environ.get("ESX_STEER_SLOTS", "65"))
MAX_TOKENS = int(os.environ.get("ESX_STEER_MAX_TOKENS", "65536"))
MEASURE = os.environ.get("ESX_STEER_MEASURE") == "1"
CHECK = os.environ.get("ESX_STEER_CHECK") == "1"
DEFAULT_CLASSES = ("vllm.model_executor.models.qwen3_5:Qwen3_5DecoderLayer,"
                   "vllm.model_executor.models.qwen3_next:Qwen3NextDecoderLayer,"
                   "vllm.model_executor.models.gemma4:Gemma4DecoderLayer")
CLASSES = [c.strip() for c in os.environ.get("ESX_STEER_CLASSES", DEFAULT_CLASSES).split(",") if c.strip()]
STATE: dict = {}


def _slot_buffer():
    if "idx" not in STATE:
        STATE["idx"] = torch.zeros(MAX_TOKENS, dtype=torch.long)  # default device of the model's init context
    return STATE["idx"]


def _register_op():
    if "op" in STATE:
        return STATE["op"]

    @torch.library.custom_op("esx::steer_add", mutates_args=())
    def steer_add(hidden: torch.Tensor, layer: int) -> torch.Tensor:
        table, idx = STATE["tables"][layer], STATE["idx"]
        return hidden + table[idx[: hidden.shape[0]]]

    @steer_add.register_fake
    def _(hidden, layer):
        return torch.empty_like(hidden)

    STATE["op"] = steer_add
    return steer_add


def _hidden_size(layer, args, kwargs) -> int:
    for k in ("hidden_size",):
        if isinstance(getattr(layer, k, None), int):
            return getattr(layer, k)
    vc = kwargs.get("vllm_config") or next((a for a in args if hasattr(a, "model_config")), None)
    if vc is not None:
        return int(vc.model_config.hf_text_config.hidden_size)
    cfg = kwargs.get("config") or next((a for a in args if hasattr(a, "hidden_size")), None)
    return int(cfg.hidden_size)


def _patch_class(Layer, op) -> None:
    if getattr(Layer, "_esx_steer_patched", False):
        return
    init0, forward0 = Layer.__init__, Layer.forward

    def __init__(self, *args, **kwargs):
        init0(self, *args, **kwargs)
        L = getattr(self, "layer_idx", None)
        if L in LAYERS and L not in STATE.get("tables", {}):
            # plain attributes, not buffers: kept out of the module state the compiler sees
            table = torch.zeros(SLOTS, _hidden_size(self, args, kwargs))
            STATE.setdefault("tables", {})[L] = table
            self.__dict__["esx_steer_table"] = table
            self.__dict__["esx_steer_idx"] = _slot_buffer()
            self.__dict__["esx_steered"] = True

    def forward(self, *args, **kwargs):
        out = forward0(self, *args, **kwargs)
        if not getattr(self, "esx_steered", False):
            return out
        h, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, None)
        if MEASURE:
            acc = STATE.setdefault("norms", {}).setdefault(self.layer_idx, [0.0, 0])
            acc[0] += float(h.float().norm(dim=-1).sum())
            acc[1] += int(h.shape[0])
        h = op(h, self.layer_idx)
        return (h, *rest) if rest is not None else h

    Layer.__init__, Layer.forward, Layer._esx_steer_patched = __init__, forward, True


def apply() -> list:
    """Patch every configured decoder class that imports, and the model runner. Returns the patched classes."""
    if not LAYERS or STATE.get("applied"):
        return STATE.get("classes", [])
    op = _register_op()
    done = []
    for spec in CLASSES:
        mod, cls = spec.split(":")
        try:
            Layer = getattr(importlib.import_module(mod), cls)
        except Exception:  # noqa: BLE001  (model family not in this vLLM)
            continue
        _patch_class(Layer, op)
        done.append(spec)
    from vllm.v1.worker import gpu_model_runner
    try:
        _patch_v2_runner()
    except Exception:  # noqa: BLE001  (older vLLM without the V2 runner)
        pass
    _patch_v1_runner(gpu_model_runner)
    STATE["applied"], STATE["classes"] = True, done
    return done


def _slot_of(sampling_params) -> int:
    extra = sampling_params.extra_args if sampling_params is not None else None
    if extra and "steer_slot" in extra:
        return min(max(int(extra["steer_slot"]), 0), SLOTS - 1)
    return 0


def _write_slots(tok: np.ndarray):
    """Token -> slot for this step, copied the way vLLM copies its own step inputs (pinned staging, non-blocking, on
    the current stream), so it is ordered before this step's forward and after the previous one."""
    idx = STATE.get("idx")
    if idx is None or not len(tok):
        return
    from vllm.utils.torch_utils import async_tensor_h2d
    async_tensor_h2d(tok, out=idx[: len(tok)])
    STATE["calls"] = STATE.get("calls", 0) + 1


def _check(bad_slot: bool, bad_pos: bool) -> None:
    STATE["check_steps"] = STATE.get("check_steps", 0) + 1
    STATE["check_bad_slot"] = STATE.get("check_bad_slot", 0) + int(bad_slot)
    STATE["check_bad_pos"] = STATE.get("check_bad_pos", 0) + int(bad_pos)


def _patch_v2_runner():
    """vLLM's V2 runner (default in 0.30): requests live at a fixed req_index in req_states; a step's batch maps
    batch position -> req_index (idx_mapping_np) and lays its tokens out in batch order."""
    from vllm.v1.worker.gpu import model_runner as mr

    Runner = mr.GPUModelRunner
    if getattr(Runner, "_esx_steer_patched", False):
        return
    add0, prepare0 = Runner.add_requests, Runner.prepare_inputs

    def add_requests(self, scheduler_output):
        add0(self, scheduler_output)
        slots = STATE.setdefault("req_slot", np.zeros(self.max_num_reqs, dtype=np.int64))
        for new in scheduler_output.scheduled_new_reqs:
            slots[self.req_states.req_id_to_index[new.req_id]] = _slot_of(new.sampling_params)
            if CHECK:
                STATE.setdefault("check_req", {})[new.req_id] = _slot_of(new.sampling_params)

    def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
        out = prepare0(self, scheduler_output, batch_req_state, batch_desc)
        slots = STATE.setdefault("req_slot", np.zeros(self.max_num_reqs, dtype=np.int64))
        idx, n = batch_req_state.idx_mapping_np, batch_req_state.num_scheduled_tokens
        per_req = slots[idx]
        STATE["last_slots"] = per_req[:16].tolist()
        tok = np.repeat(per_req, n)
        if CHECK:
            want = np.array([STATE["check_req"][r] for r in batch_req_state.req_ids], dtype=np.int64)
            start = self.req_states.num_computed_tokens_np[idx].astype(np.int64)
            pos = np.repeat(start - (np.cumsum(n) - n), n) + np.arange(len(tok))
            _check(not np.array_equal(want, per_req), not np.array_equal(pos, out.positions[: len(tok)].cpu().numpy()))
        _write_slots(tok)
        return out

    Runner.add_requests, Runner.prepare_inputs, Runner._esx_steer_patched = add_requests, prepare_inputs, True


def _patch_v1_runner(gpu_model_runner):
    """The V1 runner (VLLM_USE_V2_MODEL_RUNNER=0)."""
    Runner = gpu_model_runner.GPUModelRunner
    if getattr(Runner, "_esx_steer_patched", False):
        return
    prepare0 = Runner._prepare_inputs

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        out = prepare0(self, scheduler_output, num_scheduled_tokens)
        n = self.input_batch.num_reqs
        slots = np.array([_slot_of(self.requests[r].sampling_params) for r in self.input_batch.req_ids[:n]],
                         dtype=np.int64)
        STATE["last_slots"] = slots[:16].tolist()
        if CHECK:  # V1: slots come straight from the requests' SamplingParams; audit the token layout only
            req = num_scheduled_tokens[:n].astype(np.int64)
            start = self.input_batch.num_computed_tokens_cpu[:n].astype(np.int64)
            pos = np.repeat(start - (np.cumsum(req) - req), req) + np.arange(int(req.sum()))
            ref = getattr(getattr(self, "positions", None), "np", None)
            if ref is None:
                ref = getattr(self, "positions_np", None)
            _check(False, ref is None or not np.array_equal(pos, np.asarray(ref[: len(pos)])))
        _write_slots(np.repeat(slots, num_scheduled_tokens[:n]))
        return out

    Runner._prepare_inputs, Runner._esx_steer_patched = _prepare_inputs, True
