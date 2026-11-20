"""In-process vLLM engine with the dense Flipout component installed before graph capture."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# vLLM caches compiled graphs under a key that does not see the Flipout patch: a cached plain graph would
# silently drop the perturbation (and a cached Flipout graph would slow plain runs). Never use the cache.
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # esx repo root (was: the Agentic-ESOpt checkout)


def make_engine(model: str, *, n_slots: int = 0, eager: bool = False, master: str = "cpu", variant: str = "exact", rank: int = 0, gpu_mem: float = 0.9,
                max_model_len: int = 4672, max_num_batched_tokens: int = 16384, max_num_seqs: int = 256, prefix_cache: bool = False, extras=(), targets=None, **kw):
    """``n_slots = 0``: plain engine (shared-weight upper bound). Returns (engine, flipout or None)."""
    from esx.serving.slot_engine import SlotEngine
    import torch
    from esx.backends.flipout_dense import DenseFlipout

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    box = {}
    original_load = GPUModelRunner.load_model
    if n_slots:
        def load_model(runner, *args, **kwargs):
            out = original_load(runner, *args, **kwargs)
            before = torch.cuda.memory_allocated()
            box["flip"] = DenseFlipout(runner.get_model(), n_slots=n_slots, master=master, graph_safe=not eager, variant=variant, rank=rank, extras=extras,
                                       max_tokens=max_num_batched_tokens + max_num_seqs, **({} if targets is None else {"targets": targets}))
            # vLLM sizes the KV cache from this number; the base noise and sign tables are resident like weights.
            runner.model_memory_usage += torch.cuda.memory_allocated() - before
            return out

        GPUModelRunner.load_model = load_model
    import vllm
    real_llm = vllm.LLM

    def llm_with_cache(*args, **kwargs):  # SlotEngine hardcodes enable_prefix_caching=False; callers that turn it on must salt per slot
        kwargs["enable_prefix_caching"] = bool(prefix_cache)
        if os.environ.get("VLLM_BATCH_INVARIANT") == "1":  # batch-invariant kernels need an explicit supported backend
            kwargs.setdefault("attention_backend", "FLASH_ATTN")
        return real_llm(*args, **kwargs)

    vllm.LLM = llm_with_cache
    try:
        eng = SlotEngine(model, max_model_len=max_model_len, gpu_memory_utilization=gpu_mem, enforce_eager=eager,
                         max_num_batched_tokens=max_num_batched_tokens, max_num_seqs=max_num_seqs, **kw)
    finally:
        GPUModelRunner.load_model = original_load
        vllm.LLM = real_llm
    flip = box.get("flip")
    eng.slot_component = flip
    if flip is not None:
        flip.runner = eng.runner
        if "head" in flip.extras:
            _install_head_scoring(eng, flip)
    return eng, flip


def _install_head_scoring(eng, flip) -> None:
    """SlotEngine scores prompt tokens by calling the LM head on rows it picks itself; tell the head those rows' slots."""
    import torch

    compute = eng.model.compute_logits

    @torch.no_grad()
    def score_step(hidden_states, num_scheduled_tokens):
        runner = eng.runner
        rows, targets, keys = [], [], []
        for req_id, num_tokens in num_scheduled_tokens.items():
            request = runner.requests.get(req_id)
            extra = getattr(getattr(request, "sampling_params", None), "extra_args", None) or {}
            if "score_from" not in extra:
                continue
            prompt = request.prompt_token_ids
            start = request.num_computed_tokens
            lo = max(start, int(extra["score_from"]) - 1)
            hi = min(start + num_tokens, len(prompt) - 1)
            if hi <= lo:
                continue
            offset = int(runner.query_start_loc.np[runner.input_batch.req_id_to_index[req_id]])
            rows.append(torch.arange(offset + lo - start, offset + hi - start))
            targets.extend(prompt[lo + 1: hi + 1])
            keys.append((extra["key"], hi - lo))
        if not rows:
            return
        device = hidden_states.device
        idx = torch.cat(rows).to(device)
        tgt = torch.tensor(targets, dtype=torch.int64, device=device)
        out = []
        try:
            for lo in range(0, idx.numel(), 4096):
                sel = idx[lo:lo + 4096]
                flip.logit_override = flip.last_slots[sel] if (flip.active and flip.last_slots is not None) else torch.zeros_like(sel)
                logits = compute(hidden_states.index_select(0, sel)).float()
                out.append(logits.gather(1, tgt[lo:lo + 4096, None]).squeeze(1) - torch.logsumexp(logits, dim=-1))
        finally:
            flip.logit_override = None
        lp = torch.cat(out).cpu().tolist()
        pos = 0
        for key, n in keys:
            eng.scores.setdefault(key, []).extend(lp[pos:pos + n])
            pos += n

    eng._score_step = score_step
