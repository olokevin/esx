# AgentES-System

System track of the "Rethinking ES training" paper. Docs: `~/Project/docs/AgentES-System/`.
Outputs: a100 `/raid0-data/yequan_zhao/AgentES-System/{runs,logs}`. GPUs: a100 0, 1.

- `methods/flipout_dense.py`: full-parameter Flipout component (one perturbation per sequence, exact ES update).
  Dense or low-rank (`rank`) base noise; `extras` adds embedding, LM head, norm gains and biases.
- `methods/engine.py`: in-process vLLM engine with the component installed before graph capture; `prefix_cache`
  turns vLLM prefix caching on (callers must salt requests per slot to stay exact).
- `tests/test_flipout_parity.py`, `tests/test_extras_parity.py`: slot vs materialized member weights, update check.
- `tasks/bench/throughput.py`: fixed-workload throughput (plain / flipout / centre).
- `tasks/bench/layer_micro.py`: one-layer timing of the Flipout formulations.
- `tasks/bench/rollout.py`: real WebShop rollouts; schedulers `sync` / `stream`, prefix-cache modes; `analyze_rollout.py`.
- `tasks/bench/memory.py`: peak memory against sequence length, backprop step vs forward-only (S1).
- `tasks/bench/grad_quality.py`: ES direction quality of dense vs low-rank base noise against the true gradient.
- `tasks/bench/multi_gpu.py`: seed-communication ES across GPUs (S2).
- `scripts/sync.sh` pushes to a100; `scripts/a100_run.sh <gpu> <name> <script> [args]` runs detached (tmux
  `agsys-<name>`); `scripts/a100_queue.sh <gpu> <name> <list>` runs a list sequentially;
  `scripts/a100_env.sh` starts the WebShop env servers (ports 38400.., tmux `agsys-wsenv`).

Depends on `~/Project/ZO/Agentic-ESOpt` (`algorithms/es/fura_vllm_engine.SlotEngine`) and, for the WebShop
rollouts, on `~/Project/AgentES/tasks/webshop/{agent,webshop_server}.py`.
