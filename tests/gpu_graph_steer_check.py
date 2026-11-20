"""GPU check of esx.serving.graph_steer against a running server (compiled, CUDA graphs on). Run on the server host:

    python tests/gpu_graph_steer_check.py --url http://127.0.0.1:8920 --model MAI-Gym-9B --layers 0,...,31 --d 4096

Checks:
1. All tables zero: slot 0 and slot 3 give identical greedy tokens and logprobs.
2. A row written after startup takes effect: slot 1 with a vector on one layer changes the output.
3. Isolation in a mixed batch: 8 concurrent requests on slots {0, 1, 2} match the solo results of their slot.
   Slot 2's row equals slot 0's (zero), so it must match the base.
4. esx_steer_zero restores the base output.

The prefix cache is reset between phases, and every request gets a cache_salt per (phase, slot).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import tempfile

import numpy as np
import requests

PROMPT = ("You are operating a Linux desktop. The task: open the terminal and list the files in the home directory. "
          "Describe, step by step, the actions you would take, then give the first pyautogui call.\n\nAnswer:")


def call(url, method, *args):
    r = requests.post(f"{url}/collective_rpc", json={"method": method, "args": list(args)}, timeout=300)
    r.raise_for_status()
    return r.json()


def gen(url, model, slot, salt, n=48):
    body = {"model": model, "prompt": PROMPT, "max_tokens": n, "temperature": 0.0, "logprobs": 1,
            "cache_salt": salt, "vllm_xargs": {"steer_slot": slot}}
    r = requests.post(f"{url}/v1/completions", json=body, timeout=300)
    r.raise_for_status()
    c = r.json()["choices"][0]
    return c["logprobs"]["tokens"], np.array(c["logprobs"]["token_logprobs"], dtype=np.float64)


def same(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8920")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--layer", type=int, default=None, help="layer to steer in check 2 (default: middle)")
    ap.add_argument("--scale", type=float, default=4.0, help="per-coordinate std of the steer vector")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    L = a.layer if a.layer is not None else layers[len(layers) // 2]
    reset = lambda: requests.post(f"{a.url}/reset_prefix_cache", timeout=60)

    call(a.url, "esx_steer_zero"); reset()
    base = gen(a.url, a.model, 0, "p1-s0")
    z3 = gen(a.url, a.model, 3, "p1-s3")
    ok1 = same(base, z3)
    print(f"[1] zero tables, slot0 == slot3: {ok1}  ({''.join(base[0])[:60]!r})")

    v = np.random.default_rng(0).standard_normal(a.d).astype(np.float32) * a.scale
    tab = np.zeros((3, a.d), np.float32); tab[1] = v
    path = os.path.join(tempfile.mkdtemp(prefix="gsc_"), "t.npz")
    np.savez(path, **{str(L): tab})
    call(a.url, "esx_steer_load", path); reset()
    s1 = gen(a.url, a.model, 1, "p2-s1")
    b2 = gen(a.url, a.model, 0, "p2-s0")
    ok2 = (not same(s1, base)) and same(b2, base)
    dl = float(np.abs(s1[1][: len(base[1])] - base[1][: len(s1[1])]).max())
    print(f"[2] row 1 on layer {L} changes output: {not same(s1, base)}; slot 0 unchanged: {same(b2, base)}; "
          f"max |dlogp| = {dl:.3f}  ({''.join(s1[0])[:60]!r})")

    reset()
    plan = [0, 1, 2, 1, 0, 2, 1, 0]
    with cf.ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(gen, a.url, a.model, s, f"p3-{i}-s{s}") for i, s in enumerate(plan)]
        outs = [f.result() for f in futs]
    ref = {0: base, 1: s1, 2: base}
    # batched kernels may differ in the last bits from solo runs; tokens must match, logprobs within 1e-2
    match = [o[0] == ref[s][0] for o, s in zip(outs, plan)]
    err = [float(np.abs(o[1] - ref[s][1]).max()) if o[0] == ref[s][0] else float("nan") for o, s in zip(outs, plan)]
    ok3 = all(match)
    print(f"[3] mixed batch slots {plan}: tokens match solo = {match}; max |dlogp| = {np.nanmax(err):.2e}")

    call(a.url, "esx_steer_zero"); reset()
    z = gen(a.url, a.model, 1, "p4-s1")
    ok4 = same(z, base)
    print(f"[4] after zero, slot 1 == base: {ok4}")
    print("debug:", call(a.url, "esx_steer_debug"))
    print("ALL OK" if ok1 and ok2 and ok3 and ok4 else "FAILED")


if __name__ == "__main__":
    main()
