# Migrating a project to esx

esx is used as a git submodule. Migrate one project at a time, and verify every switch numerically before trusting new
results. Only ES-SEAgent has been migrated so far, on 2026-11-20. The other projects below have a plan but are
untouched.

## Steps (any project)
1. **Add the submodule.**
   `git submodule add git@github.com:olokevin/esx.git third_party/esx`, then
   `git submodule update --init`.
2. **Make the package importable** in the environment that runs the ES code. Either:
   - `pip install -e third_party/esx`; or
   - put `third_party/esx` on `sys.path` / `PYTHONPATH`, as ES-SEAgent does.
3. **Replace imports** with this map:

   | Old | New |
   |---|---|
   | `algorithms.es.seeded_model_es` / `structured_es` / `batched_es` / `dense_inplace` / `fura_native` / `fura_s_pretransform` / `fura_s_quant` / `lora_slots` / `rot_*` / `steer_slots` | `esx.backends.<same name>` |
   | `algorithms.es.run_state`, `algorithms.es.registry` | `esx.run_state`, `esx.registry` |
   | `algorithms.es.fura_vllm_engine` | `esx.serving.slot_engine` |
   | `algorithms.es.model_es_client` | `esx.serving.http_client` |
   | `mwt.es_client` | `esx.serving.collective_client` |
   | `param_es_worker`, `fura_worker` (worker extensions) | `esx.backends.mwt.param_es_worker`, `esx.backends.mwt.fura_worker` (vLLM: `--worker-extension-cls esx.backends.mwt.fura_worker.WorkerExtension`) |
   | `mwt.fura` | `esx.backends.mwt.fura_blocks` |
   | AgentES-System `methods.flipout_dense`, `methods.engine` | `esx.backends.flipout_dense`, `esx.serving.flipout_engine` |
   | ES-SEAgent `methods.discovery.entropic` / `archive` | `esx.estimators.entropic` / `archive` |

4. **Verify** before using the switched code for results:
   - **(a) Code identity.** `python -m pytest third_party/esx/tests`. `test_legacy_parity` proves the moved code
     equals what the project ran before.
   - **(b) Numeric replay.** Take one finished run of the project. Rebuild its final centre on the new import path
     from the history (seeds and coefficients), and compare teacher-forced log π on a few recorded responses with the
     old path. They must be equal to the bf16 noise floor. Re-run a short evaluation of that centre.
   - **(c) A smoke run** of the project's own driver: 2–3 updates.
5. **Remove the project's private copy** only after (a)–(c), in a separate commit.

## Per project (plan only; not done)
| Project | ES code it uses now | Migrate to | Watch out for |
|---|---|---|---|
| **Agentic-ESOpt** | owns `algorithms/es` (+12 untracked files) | replace `algorithms/es` with the submodule. The untracked files there must first be committed or discarded by their owner. | `vllm_math_es_worker.py` duplicates `seeded_model_es` dense code. Keep it task-side or move its parity test into esx. |
| **AgentES** | `algorithms.es.fura_vllm_engine` (untracked in Agentic-ESOpt!), own `methods/es/lora_pop.py`, `train_flipout.py` | `esx.serving.slot_engine`; `lora_pop` to `esx.backends.lora_slots` | `lora_pop` uses a crc32 tensor id where `lora_slots` uses blake2b, so the same seed gives different noise. Keep `lora_pop`'s spec for replays of old runs. |
| **AgentES-System** | its own `methods/{engine,flipout_dense}.py` (now in esx) + SlotEngine | `esx.serving.flipout_engine`, `esx.backends.flipout_dense` | The repo has no commits; commit it first. Benches (`tasks/bench`) stay in the project. |
| **specialists_team** | `algorithms.es.{fura_native,structured_es,fura_vllm_engine}` + private copies (`specialists/perturb/structured_es.py`, `batched_es.py`, `steer_slots.py`, `struct_slots.py`) | esx backends; per-member sigma becomes an option of `batched_es` | FuraNative tensor id without `.weight` (audit R1); `--vec-mode scalar` is a runtime monkeypatch (R9). |
| **TreeES** | drives the specialists `es_server` over HTTP | follows specialists_team; estimators to `esx.estimators` (tree) | `kappa * sigma / ‖w‖` step option (R5). |
| **ForkES** | own `methods/fork_es/estimators.py` (classic 1/sigma gradient) | `esx.estimators.fork` (new) | Repo has no commits. |
| **HarnessES** | own `methods/steer_es/` (numpy RNG, `alpha/pairs`), vLLM 0.30 | `esx.backends.steer_slots` with HarnessES's scale as an option | 2× step versus `alpha/N` (R5); vLLM 0.30 pin. |
| **ES-SEAgent** | done: `esx.serving.collective_client`, `esx.backends.mwt.*`, `esx.estimators.*` | — | E4 `methods/structes/spec_worker.py` (placement-spec worker) is still project code, extending `esx.backends.mwt.fura_worker`. Candidate for `esx.backends` in phase 1. |
