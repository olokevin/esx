"""In-process vLLM engine with slot routing and compact teacher-forced scoring.

Run with ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` so the model runner lives in this
process. Two things are patched into the runner (nothing in vLLM itself changes):

* **slot routing** -- a request names its slot in ``SamplingParams.extra_args["slot"]``;
  before every forward pass the per-token slot vector is built from the batch's
  request -> token map and handed to whatever per-slot component is installed
  (``FuraNative``, or the dense-perturbation fallbacks);
* **compact scoring** -- a request with ``extra_args["score_from"] = k`` gets the
  log-probabilities of prompt tokens ``k ..`` computed in the runner (logsumexp and
  gather over the target positions only) and stashed under ``extra_args["key"]``,
  instead of vLLM's full prompt-logprob path, which builds a full-vocabulary
  log-softmax and Python dicts for every prompt token.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch


class SlotEngine:
    def __init__(self, model_path: str, *, max_model_len: int = 4096, gpu_memory_utilization: float = 0.6,
                 enforce_eager: bool = True, max_num_batched_tokens: int = 16384, max_num_seqs: int = 512,
                 seed: int = 0, pre_compile_hook=None, **llm_kwargs):
        """``pre_compile_hook(model)`` runs right after the weights are loaded and before
        vLLM profiles memory, compiles and captures CUDA graphs, so components installed
        there (graph-safe FuRA) are part of the captured graphs; its return value is kept
        as ``self.hook_result``."""
        from vllm import LLM

        self.hook_result = None
        if pre_compile_hook is not None:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner

            original_load = GPUModelRunner.load_model
            engine = self

            def load_model(runner, *args, **kwargs):
                out = original_load(runner, *args, **kwargs)
                engine.hook_result = pre_compile_hook(runner.get_model())
                return out

            GPUModelRunner.load_model = load_model

        self.model_path = model_path
        self.llm = LLM(
            model_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            enable_prefix_caching=False,  # cached KV would leak across slots
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            limit_mm_per_prompt={"image": 0, "video": 0},
            seed=seed,
            **llm_kwargs,
        )
        if pre_compile_hook is not None:
            GPUModelRunner.load_model = original_load
        self.tokenizer = self.llm.get_tokenizer()
        self.runner = self._find_runner()
        self.model = self.runner.model
        self.slot_component = None
        self.lora_requests = None  # slot s > 0 -> lora_requests[s - 1] (LoraSlots / BP adapters)
        self.scores: Dict[Any, List[float]] = {}
        self.stats = {"score_tokens": 0, "score_time": 0.0, "gen_tokens": 0, "gen_time": 0.0,
                      "score_calls": 0, "gen_calls": 0, "score_seqs": 0, "gen_seqs": 0}
        self._install_routing()
        self._install_scoring()

    # ------------------------------------------------------------------ plumbing
    def _find_runner(self):
        executor = self.llm.llm_engine.model_executor
        worker = getattr(executor, "driver_worker", None)
        worker = getattr(worker, "worker", worker)
        runner = getattr(worker, "model_runner", None)
        if runner is None:
            raise RuntimeError("Run with VLLM_ENABLE_V1_MULTIPROCESSING=0 (in-process engine core).")
        return runner

    def _slots_for_batch(self, num_tokens: int, device) -> Optional[torch.Tensor]:
        runner = self.runner
        table = []
        any_slot = False
        for req_id in runner.input_batch.req_ids:
            request = runner.requests.get(req_id) if req_id is not None else None
            extra = getattr(getattr(request, "sampling_params", None), "extra_args", None) or {}
            slot = int(extra.get("slot", 0))
            any_slot = any_slot or slot != 0
            table.append(slot)
        if not any_slot:
            return None
        table = torch.tensor(table, dtype=torch.int64, device=device)
        # Inputs can be padded (CUDA-graph sizes); only the scheduled tokens have a request.
        n_real = min(num_tokens, int(runner.query_start_loc.np[len(runner.input_batch.req_ids)]))
        slots = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        slots[:n_real] = table[runner.req_indices.gpu[:n_real]]
        return slots

    def _install_routing(self) -> None:
        runner = self.runner
        forward = runner._model_forward
        engine = self

        def model_forward(*args, **kwargs):
            comp = engine.slot_component
            if comp is None:
                return forward(*args, **kwargs)
            inputs = kwargs.get("input_ids")
            if inputs is None:
                inputs = kwargs.get("inputs_embeds")
            num_tokens = inputs.shape[0]
            slots = engine._slots_for_batch(num_tokens, inputs.device)
            if slots is None and getattr(comp, "route_all", False):  # slot 0 is a trained centre, not the base
                slots = torch.zeros(num_tokens, dtype=torch.int64, device=inputs.device)
            comp.set_token_slots(slots)
            try:
                return forward(*args, **kwargs)
            finally:
                comp.set_token_slots(None)

        runner._model_forward = model_forward

    def _install_scoring(self) -> None:
        runner = self.runner
        original = runner._get_prompt_logprobs_dict
        engine = self

        def get_prompt_logprobs_dict(hidden_states, num_scheduled_tokens):
            engine._score_step(hidden_states, num_scheduled_tokens)
            return original(hidden_states, num_scheduled_tokens)

        runner._get_prompt_logprobs_dict = get_prompt_logprobs_dict

    @torch.no_grad()
    def _score_step(self, hidden_states: torch.Tensor, num_scheduled_tokens: Dict[str, int]) -> None:
        runner = self.runner
        rows, targets, keys = [], [], []
        for req_id, num_tokens in num_scheduled_tokens.items():
            request = runner.requests.get(req_id)
            extra = getattr(getattr(request, "sampling_params", None), "extra_args", None) or {}
            if "score_from" not in extra:
                continue
            prompt = request.prompt_token_ids
            start = request.num_computed_tokens  # first position of this chunk
            lo = max(start, int(extra["score_from"]) - 1)  # position p predicts token p + 1
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
        for lo in range(0, idx.numel(), 4096):  # bound the (tokens, vocab) logits buffer
            logits = self.model.compute_logits(hidden_states.index_select(0, idx[lo:lo + 4096])).float()
            lp = logits.gather(1, tgt[lo:lo + 4096, None]).squeeze(1) - torch.logsumexp(logits, dim=-1)
            out.append(lp)
        lp = torch.cat(out).cpu().tolist()
        pos = 0
        for key, n in keys:
            self.scores.setdefault(key, []).extend(lp[pos:pos + n])
            pos += n

    # ------------------------------------------------------------------ API
    def score(self, items: Sequence[Tuple[int, List[int], int]]) -> List[List[float]]:
        """items: ``(slot, token_ids, score_from)``; returns per-token log-probs of tokens
        ``score_from ..`` of every sequence."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        self.scores = {}
        prompts, params = [], []
        for i, (slot, ids, start) in enumerate(items):
            prompts.append(TokensPrompt(prompt_token_ids=list(ids)))
            params.append(SamplingParams(max_tokens=1, temperature=0.0,
                                         extra_args={"slot": int(slot), "score_from": int(start), "key": i}))
        t0 = time.perf_counter()
        self.llm.generate(prompts, params, use_tqdm=False, lora_request=self._lora_for([s for s, _, _ in items]))
        torch.cuda.synchronize()
        self.stats["score_time"] += time.perf_counter() - t0
        self.stats["score_calls"] += 1
        self.stats["score_seqs"] += len(items)
        self.stats["score_tokens"] += sum(len(ids) for _, ids, _ in items)
        out = [self.scores.get(i, []) for i in range(len(items))]
        for (slot, ids, start), lp in zip(items, out):
            if len(lp) != len(ids) - start:
                raise RuntimeError(f"scored {len(lp)} tokens, expected {len(ids) - start}")
        self.scores = {}
        return out

    def generate(self, items: Sequence[Tuple[int, str]], *, max_tokens: int = 64, temperature: float = 0.0,
                 top_p: float = 1.0, seed: Optional[int] = None, stop: Optional[List[str]] = None) -> List[str]:
        from vllm import SamplingParams

        prompts, params = [], []
        for i, (slot, prompt) in enumerate(items):
            prompts.append(prompt)
            params.append(SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                                         seed=None if seed is None else seed + i, stop=stop,
                                         extra_args={"slot": int(slot)}))
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, params, use_tqdm=False,
                                 lora_request=self._lora_for([s for s, _ in items]))
        torch.cuda.synchronize()
        self.stats["gen_time"] += time.perf_counter() - t0
        self.stats["gen_calls"] += 1
        self.stats["gen_seqs"] += len(items)
        texts = [o.outputs[0].text for o in outs]
        self.stats["gen_tokens"] += sum(len(o.outputs[0].token_ids) for o in outs)
        return texts

    def _lora_for(self, slots):
        if self.lora_requests is None:
            return None
        return [self.lora_requests[s - 1] if s > 0 else None for s in slots]

    def memory(self) -> Dict[str, float]:
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "device_used_gb": (total - free) / 1e9,
        }
