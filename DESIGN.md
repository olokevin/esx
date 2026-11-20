# esx — one ES library for all projects: design

## Status
**Phase 0 (2026-11-20): consolidation without numeric change.**
- Every ES implementation that the projects import was moved here verbatim, with only its import paths changed:
  - Agentic-ESOpt `algorithms/es`, including the 12 files that were never committed there;
  - the MobileWorld ES path;
  - AgentES-System's Flipout engine;
  - ES-SEAgent's estimators.
- `tests/test_legacy_parity.py` checks each moved file against its source line by line, imports normalised.
- The original unit tests pass here: 56 passed, 1 skipped, which is the Math-worker parity test that needs
  Agentic-ESOpt task code.
- Old histories therefore replay unchanged on esx.
- The phases after this one change numerics, and only behind versioned names (see "Noise specs").

The cross-project audit this design is based on is `docs/AUDIT_2026-11-20.md`. It covers 8 FuRA-S implementations,
5 steering-ES designs and 15 numerics risks.

## Layout
```
esx/
  backends/            how a model is perturbed (one file per family, moved verbatim)
    seeded_model_es.py   SeedReplayModelES: full / all_linear / lora scopes, + structured scopes via structured_es
    structured_es.py     fura, fura_s, iso, isobtt (fp32 state per linear weight)
    batched_es.py        per-token-member (Flipout-style) fura_s / isobtt for in-process vLLM
    flipout_dense.py     AgentES-System DenseFlipout: exact / fused / low-rank / core Flipout + extras (embed, head, norm, bias)
    dense_inplace.py     in-place dense + iso with bf16 backup and stochastic rounding
    fura_native.py, fura_s_pretransform.py, fura_s_quant.py   native-layout / HF-pretransformed / quantised FuRA-S
    lora_slots.py, rot_slots.py, rot_tp.py, rot_hf.py, steer_slots.py   per-request slot backends (LoRA, rotations, steering)
    mwt/                 the MobileWorld path: param_es_worker (scopes full/lora/steer/fura_s/fura_s_rel/...),
                         fura_worker (fura_s_paper), fura_blocks (paper block layout and SVD)
  serving/             how members reach the model
    slot_engine.py       SlotEngine: one in-process vLLM engine, whole population in one batch (extra_args["slot"])
    flipout_engine.py    make_engine: installs DenseFlipout inside load_model (before profiling / CUDA graphs)
    collective_client.py EsServer: client for `vllm serve --worker-extension-cls ...` + /collective_rpc (pes_* calls)
    http_client.py       client for the /es/{init,apply,revert,update,reset,status} HTTP routes
  estimators/          how fitness becomes an update
    entropic.py          entropic tilt q = softmax(beta r), KL(q || uniform) = gamma
    archive.py           PUCT archive of states
  run_state.py         sigma schedules, atomic history.json, seed-stream validation, replay onto fresh endpoints
  registry.py          ESMethodSpec registry
tests/                 moved unit tests + legacy parity
docs/                  audit, the original algorithms/es README, AgentES-System README
```

## Target architecture: the AgentES-System design, adopted

This is the serving design every project moves to. Details are in `docs/AUDIT_2026-11-20.md` §3 and
`~/Project/docs/AgentES-System/design.md`.

1. **One engine, the whole population in one batch.**
   - Each request names its member slot.
   - The backend writes the per-token slot vector into a fixed buffer before every forward, so CUDA graphs stay valid.
   - Member state lives in module buffers written in place.
   - The extra GEMM always runs, so the graph has no data-dependent branch.
2. **Install before profiling.** Perturbation buffers are created inside `GPUModelRunner.load_model` and counted in
   the model's memory usage, so the KV cache is sized correctly.
3. **Seed-only coordination.**
   - The coordinator sends `(gen_seed, [(member_seed, sign)], sigma)` and receives one fitness per member, a few KB
     per generation.
   - Workers on several GPUs stay bit-identical without exchanging weights.
4. **Backends behind the same slot interface:**
   - dense Flipout (exact or fused);
   - low-rank Flipout (rank k; 0.3 GB of noise at 7B instead of 13–15 GB);
   - core;
   - LoRA slots;
   - steering slots;
   - FuRA-S batched.
5. **Measured costs (7B).**
   - Throughput relative to plain shared weights: dense Flipout 2.0–2.1×, rank-64 1.23–1.26×, multi-LoRA 2.9×.
   - Continuous scheduling adds +14–18 %.
6. **The `vllm serve` + `collective_rpc` path stays supported.** This is `serving/collective_client.py` with the
   `backends/mwt` worker. ES-SEAgent and MobileWorld use it, and the E4 spec worker extends it. In this path a member
   is evaluated by moving the centre (`pes_update(±sigma)`) one member at a time, or through ES-SEAgent's per-request
   `es_slot` steering.

## Noise specs (how numerics stay reproducible)
A noise spec fixes everything that turns `(seed, tensor)` into numbers:
- the tensor-id hash (for example blake2b-8 of the name with or without `.weight`, or crc32);
- the seed mix (`seed ^ tid` on 64 bits, or `(seed * 1000003 + salt) ^ tid` on 63 bits);
- the generator device;
- the chunking of the draw (8M-element chunks, one call per tensor, or 2^26-element rows);
- the sigma convention (absolute, relative to |S0|, or relative to residual RMS).

**Phase 0** keeps each backend's own spec. These are the legacy specs. They are listed in the audit §10, where R1–R4
and R6–R7 are the numerics differences.

**Phase 1** names them explicitly. Each history records `noise_spec`, and a backend refuses to replay a history whose
spec it does not implement. Scope names that hide two conventions are split:
- `fura_s` becomes `fura_s_abs` (closest-factor blocks of the fused weight, absolute sigma), `fura_s_rel` (relative
  |S0|) and `fura_s_paper` (per-projection / per-head layout, relative);
- `full` becomes `dense_iid` and `flipout` (sign-flips of one shared E per generation), plus `lowrank`.

## Decisions (defaults used now; to be confirmed by the user)
| # | Question | Default in esx now | Proposed for phase 1 |
|---|---|---|---|
| 1 | Canonical noise spec; must old histories replay bit-exactly? | each backend keeps its legacy spec; yes | new runs: `esx-v1` = blake2b-8 of the full parameter name incl. `.weight`, 63-bit mixF with salts, device generator, 2^26 row chunks; legacy specs kept as named aliases |
| 2 | What N is in alpha/N; where rewards are normalised | unchanged per caller | N = number of members (2 × pairs); normalisation client-side by an `estimators/normalize.py` shared by all |
| 3 | Scope names that hide a sigma convention | unchanged | split as in "Noise specs" |
| 4 | Default perturbation family | per project | low-rank Flipout (rank 64) for full-model ES; steering and LoRA slots for placement ES |
| 5 | Centre precision | per backend | fp32 master (GPU for small states, CPU pinned for 7B+); stochastic rounding only as an option |
| 6 | Serving | both paths exist | in-process SlotEngine for new work; `vllm serve` + `collective_rpc` kept for MobileWorld/OSWorld agents that need an OpenAI server |
| 7 | vLLM pin | SlotEngine patches private runner internals of vLLM 0.19.1; HarnessES needs 0.30 | CI matrix 0.19.1 and 0.30; patches isolated in `serving/patches/` |
| 8 | Canonical steering | ES-SEAgent spec steer (after `down_proj`, after the TP all-reduce, sigma relative to the residual RMS, eager mode) and `steer_slots` (per-token relative) both kept | one `backends/steer.py` with the injection point and the scale as explicit options |
| 9 | Where the library lives | this repo; projects use it as a git submodule | same, plus `pip install -e` in each environment |
| 10 | Runtime monkeypatches that change noise (`--vec-mode scalar`, `--basis`, IsoSeq `_init_iso`) | untouched in their projects | explicit backend options |
| 11 | Prefix-cache salt policy | per project | owned by the engine: salt = (slot, population version) |
| 12 | Resume / rollout-dump contract | run_state.py | one history schema with `noise_spec`, plus one rollout-dump format |
| 13 | ES-SEAgent member evaluation | moving the centre (`pes_update`), or `es_slot` for steering | slots wherever the backend supports them |

## Provenance (phase 0)
| esx file | source |
|---|---|
| backends/{seeded_model_es,structured_es,batched_es}.py, run_state.py, registry.py, serving/http_client.py (model_es_client) | Agentic-ESOpt `algorithms/es` (git history kept: `git subtree split`, 4 commits) |
| backends/{dense_inplace,fura_native,fura_s_pretransform,fura_s_quant,lora_slots,rot_hf,rot_slots,rot_tp,steer_slots}.py, serving/slot_engine.py (fura_vllm_engine), tests/test_fura_{native,s_pretransform}.py | Agentic-ESOpt `algorithms/es` working tree, never committed there (snapshot 2026-11-20) |
| backends/mwt/{param_es_worker,fura_worker,fura_blocks}.py, serving/collective_client.py | Agentic-ESOpt ef7b126 `scripts/motivation/param_es_worker.py`, `mobileworld-train-time/{workers/fura_worker.py, mwt/fura.py, mwt/es_client.py}` |
| backends/flipout_dense.py, serving/flipout_engine.py | AgentES-System `methods/{flipout_dense,engine}.py` (uncommitted project) |
| estimators/{entropic,archive}.py | ES-SEAgent 695fee1 `methods/discovery/` |
