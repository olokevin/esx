# Shared Agentic-ESOpt layer

`es/` contains task-independent model-weight evolution code used by Sudoku,
Math, DocVQA, WebArena, and AHD.

```text
model_es_client.py   HTTP client for /es/* routes
seeded_model_es.py   deterministic noise and model update implementation
structured_es.py     fura / fura_s / iso / isobtt scopes (shared with the vLLM worker)
batched_es.py        fura_s / isobtt run from their factors, one perturbation per sequence
run_state.py         sigma schedules, atomic history, validation, replay
registry.py          method metadata
test_run_state.py    schedule and history unit tests
test_structured_es.py  structured-scope and vLLM-worker parity tests
```

## Sigma contract

Task runners supply an explicit start and end:

```python
from es.run_state import sigma_at_step

sigma = sigma_at_step(
    sigma_start=1e-3,
    sigma_end=1e-4,
    step=generation,
    total_steps=20,
    schedule="cosine",  # constant, linear, or cosine
    warmup_steps=0,
)
```

For linear and cosine schedules, the first scheduled step is exactly
`sigma_start` and the last is exactly `sigma_end`. Warmup keeps the start value
fixed before the decay. A decay needs at least two generations; with one
generation, the sole perturbation uses `sigma_start`.

## Endpoint protocol

Initialize the selected parameter scope:

```json
POST /es/init
{"parameter_scope": "full", "target_modules": null, "verbose": true}
```

The structured scopes also accept `"scope_options": {...}`; see
[Structured scopes](#structured-scopes).

Calling `/es/init` again first restores all tracked updates and any active
perturbation, so replay on a reused server starts from the same base weights.

Apply and revert deterministic Gaussian noise:

```json
POST /es/apply
{"seed": 123, "sigma": 0.001}

POST /es/revert
{"seed": 123, "sigma": 0.001}
```

Apply the seed-replay update:

```json
POST /es/update
{
  "seeds": [123, 456],
  "rewards": [0.4, 0.8],
  "alpha": 0.001,
  "reward_normalization": "zscore",
  "reward_normalization_ddof": 0,
  "reward_normalization_eps": 1e-8
}
```

The server regenerates each noise direction and applies:

```text
theta <- theta + (alpha / N) * sum_i weight_i * epsilon(seed_i)
```

`/es/reset` undoes the ES state accumulated after initialization, and
`/es/status` reports the selected parameter set and active state.

Supported scopes are `full`, `all_linear`, `lora`, and the structured scopes
`fura`, `fura_s`, `iso`, and `isobtt`. LoRA scope requires the server model to have LoRA
parameters before `/es/init`.

## Structured scopes

`fura`, `fura_s`, `iso`, and `isobtt` restrict ES to a structured family of each
linear weight. They select the same tensors as `all_linear` (2-D weights outside the
embeddings, LM head, and norms, optionally filtered by `target_modules`) and
keep an fp32 *state* per tensor, from which the live weight is rendered:

| Scope | Family | Trained state | Meaning of `sigma` |
| --- | --- | --- | --- |
| `fura` | Block-wise SVD per input block, `W[:, blk_j] = A_j (R0_j + C_j)` with `A_j = U_j diag(S_j)` frozen | additive core `C_j` | absolute noise std on the core |
| `fura_s` | Same factorisation with both frames frozen, `W[:, blk_j] = U_j diag(S0_j + c_j) Vh_j` | singular-value offsets `c_j` | absolute noise std on the singular values |
| `isobtt` | Same factorisation, core kept orthogonal: `R_j <- Cay(sigma * Omega_j) R_j` | orthogonal core `R_j` | relative displacement `‖dW‖ / ‖W‖` |
| `iso` | Fixed spectrum of the whole matrix: `W <- C_L W C_R^T`, `C` exactly orthogonal | fp32 master of `W` | relative displacement `‖dW‖ / ‖W‖` |

`iso` preserves every singular value of each weight, `isobtt` every singular
value of each input block, and `fura` preserves neither. `fura_s` is the
complement of `isobtt`: it keeps each block's singular vectors and moves only
its singular values, which makes it by far the smallest scope (one coefficient
per singular value). Input blocks come from
the factor pair of `in_features` closest to its square root. The update is

```text
fura, fura_s:  C <- C + (alpha / N) * sum_i weight_i * epsilon(seed_i)
iso / isobtt:  state <- prod_i Cay((alpha / N) * weight_i * Omega(seed_i)) state
```

so the request and history formats are unchanged, and a history replays to the
same weights as long as `/es/init` is called with the same scope and options.
Differences from the dense scopes that matter when running them:

- **Step sizes are not interchangeable.** For `iso` and `isobtt`, `sigma` is a
  relative footprint, so the dense default of `1e-3` is far too small; the
  reference runs used `sigma = 5e-2`, `alpha = sigma / 2`. For `fura` they used
  `sigma = 1.25e-2`, `alpha = sigma / 2`. Treat these as starting points.
- **Revert is exact.** `/es/revert` re-renders the weight from the unperturbed
  state instead of subtracting noise.
- **`fura`, `fura_s`, and `isobtt` change the base model slightly.** `/es/init` overwrites
  each weight with its block-SVD reconstruction, which differs from the loaded
  weight by rounding in the model dtype (about `1.6e-3` relative in bfloat16).
  The model keeps that reconstruction after `/es/reset`; reload the checkpoint
  to get the original weights back. `iso` starts from the loaded weights
  exactly.
- **Memory.** `iso` holds an fp32 copy of every selected weight (twice the
  bfloat16 model); `fura`, `fura_s`, and `isobtt` hold a frozen frame the size of the
  selected weights in the model dtype plus a small fp32 core. Lower the GPU
  memory fraction given to the inference engine accordingly. The `batched`
  option below removes this overhead for `fura_s` and `isobtt`.
- **Fused weights.** vLLM fuses projections such as q/k/v into one matrix. `iso`
  reads the split from the layer's `output_sizes` and never rotates rows across
  it, so the spectrum of each original matrix is the one preserved.

`/es/update` additionally returns `frob_drift` for `iso` (relative change of
`‖W‖_F` at that update, before it is corrected) and `orth_err` for `isobtt`
(largest entry of `|R R^T - I|`); both should stay near `1e-6`.

Options, passed as `scope_options` to `/es/init`:

| Option | Default | Applies to | Effect |
| --- | --- | --- | --- |
| `iso_block_size` | `128` | `iso` | block size of the skew generator; cost per perturbation is `O(b * |W|)` |
| `iso_perm` | `true` | `iso` | re-draw the block basis for every seed |
| `swap_blocks` | `false` | `fura`, `fura_s`, `isobtt` | swap the number of input blocks and the block size |
| `batched` | `false` | `fura_s`, `isobtt` | run the layer from its factors and perturb each sequence separately (in-process vLLM only; see below) |

### Batched perturbation (`"batched": true`)

By default a structured scope renders one perturbed dense weight, so an engine
evaluates one population member at a time and holds the frozen frame next to
the dense weight. With `"batched": true`, `fura_s` and `isobtt` are served by
`batched_es.py` instead:

- **No dense weight.** Each selected linear layer is replaced by its block-SVD
  factors and computes `y = frame @ perturb(core, x)` in the model dtype. Memory
  is that of the unmodified model, so the engine keeps its full KV cache.
- **One perturbation per sequence.** The perturbation is applied to the small
  intermediate activation, per token, following Flipout (Wen et al., ICLR 2018).
  A request names its member in `SamplingParams.extra_args["es_member"]`, and
  the whole population shares one engine batch.
  - `isobtt`: a generation shares one skew generator `Omega` (seeded from all of
    its seeds) and member `n` flips its sign pattern,
    `Omega_n = D_n Omega D_n`. Since `Cay(D Omega D) = D Cay(Omega) D`, each
    member's rotation is exactly orthogonal. The update is
    `R <- Cay(alpha * Omega o M) R`, `M = (1/N) sum_n weight_n d_n d_n^T`.
  - `fura_s`: the core is a vector, so each member gets its own Gaussian draw
    -- the same draws the unbatched scope makes for those seeds.
- **Cores live in the model dtype.** One ES step is below a bfloat16 ulp for
  many entries, so updates are written back with seeded stochastic rounding
  (unbiased, and identical on replay). This costs precision: over 10 updates at
  `alpha = 1e-2` on Qwen3.5-4B the stored update had cosine 0.83-0.84 with the
  exact one. `orth_err` for `isobtt` is correspondingly about `1e-2`, not `1e-6`.
- **Different `isobtt` trajectories.** The sign-shared generator is not the
  per-seed generator of the unbatched scope, so an `isobtt` history replays only
  under the option it was written with (the runner checks this on resume).
- **Runner support.** The vLLM worker exposes `set_population_math_es` /
  `clear_population_math_es`; `apply_math_es` is refused. DocVQA rolls a whole
  generation out through one continuously refilled engine pool
  (`--inference-batch-size` live rollouts, `--rollout-token-budget` tokens
  reserved in the engine at once). The Math runner and the HTTP servers do not
  support the option yet.
- **Prefix caching** (`MATH_VLLM_PREFIX_CACHING=1`, off by default) is safe
  here and worth about 2x on multi-turn rollouts, because a turn then prefills
  only its new tokens. Cached blocks are valid only for the weights that wrote
  them, so the pool salts each request's cache with its member and the weight
  version, and the actor drops the cache at every population change and update.

## Durable history

Every active task stores the seeds, raw rewards, alpha, normalization settings,
schedule metadata, and server update responses in `history.json`. Writes use a
temporary sibling followed by `os.replace`, so an interrupted process does not
leave half-written JSON.

On resume, runners:

1. initialize every fresh model endpoint;
2. select completed update records;
3. validate the deterministic seed stream and population;
4. replay every update on every endpoint; and
5. continue from the next generation.

AHD additionally records the EoH generation and operator for each model update.
Its model history is separate from EoH's saved population JSON.

Run the unit tests with:

```bash
python -m unittest algorithms.es.test_run_state -v
python -m unittest algorithms.es.test_structured_es -v      # needs torch
```
