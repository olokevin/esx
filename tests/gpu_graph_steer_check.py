"""GPU check of esx.serving.graph_steer against a running server (compiled, CUDA graphs on). Run on the server host:

    python tests/gpu_graph_steer_check.py --url http://127.0.0.1:8920 --model MAI-Gym-9B --layers 0,...,31 --d 4096

Checks:
1. All tables zero: slot 0 and slot 3 give identical greedy tokens and logprobs.
2. A row written after startup takes effect: slot 1 with a vector on one layer changes the output.
3. Isolation in a mixed batch: 8 concurrent requests on slots {0, 1, 2} (slot 2's row is zero, like slot 0's).
   Batched and solo runs use different kernel shapes, so their bf16 logits differ by ~0.1 nat; greedy decoding then
   flips wherever the solo top-2 margin is that small. So check 3 is judged by teacher forcing, not by greedy text:
   a. teacher-forced logprobs of each slot's solo continuation, scored inside a mixed batch, are within --tol of the
      solo scores (max over tokens), and every request is at least 4x closer (sum |dlogp|) to the solo score of its
      own slot than to the solo score of the other row (slot 0 vs 1) on the same text;
   b. greedy outputs match solo up to their first divergence, and any divergence is at a near-tie (solo top-2
      margin < --tol);
   c. with ESX_STEER_CHECK=1 on the server: no step had a wrong token -> slot row (check_bad_slot, check_bad_pos).
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


def score(url, model, slot, salt, text):
    """Teacher-forced logprobs of `text` (prompt_logprobs) under `slot`."""
    body = {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0,
            "cache_salt": salt, "vllm_xargs": {"steer_slot": slot}}
    r = requests.post(f"{url}/v1/completions", json=body, timeout=300)
    r.raise_for_status()
    pl = r.json()["choices"][0]["prompt_logprobs"]
    # prompt_logprobs=0: each entry holds just the prompt token's own logprob (the first entry is None)
    return np.array([next(iter(d.values()))["logprob"] for d in pl[1:]], dtype=np.float64)


def gen_margin(url, model, slot, salt, n=48):
    body = {"model": model, "prompt": PROMPT, "max_tokens": n, "temperature": 0.0, "logprobs": 2,
            "cache_salt": salt, "vllm_xargs": {"steer_slot": slot}}
    r = requests.post(f"{url}/v1/completions", json=body, timeout=300)
    r.raise_for_status()
    lp = r.json()["choices"][0]["logprobs"]
    m = [(lambda v: v[0] - v[1] if len(v) > 1 else np.inf)(sorted(t.values(), reverse=True)) for t in lp["top_logprobs"]]
    return lp["tokens"], np.array(m)


def first_div(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


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
    ap.add_argument("--tol", type=float, default=0.25, help="check 3: batched-vs-solo logprob tolerance (nat)")
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

    plan = [0, 1, 2, 1, 0, 2, 1, 0]
    text = {0: PROMPT + "".join(base[0]), 1: PROMPT + "".join(s1[0]), 2: PROMPT + "".join(base[0])}
    reset()
    tf_solo = {s: score(a.url, a.model, s, f"p3s-{s}", text[s]) for s in (0, 1)}
    # the other row's solo score of each text: slot 1's text without steering, the base text with row 1
    tf_wrong = {1: score(a.url, a.model, 0, "p3s-x1", text[1]), 0: score(a.url, a.model, 1, "p3s-x0", text[0])}
    m_solo = {0: gen_margin(a.url, a.model, 0, "p3g-0"), 1: gen_margin(a.url, a.model, 1, "p3g-1")}
    reset()
    with cf.ThreadPoolExecutor(2 * len(plan)) as ex:
        tf_f = [ex.submit(score, a.url, a.model, s, f"p3m-{i}-s{s}", text[s]) for i, s in enumerate(plan)]
        g_f = [ex.submit(gen_margin, a.url, a.model, s, f"p3n-{i}-s{s}") for i, s in enumerate(plan)]
        tf_mix, g_mix = [f.result() for f in tf_f], [f.result() for f in g_f]
    R = {0: 0, 1: 1, 2: 0}  # the solo reference of each slot (slot 2's row is zero, like slot 0's)
    tf_err = [float(np.abs(t - tf_solo[R[s]]).max()) for t, s in zip(tf_mix, plan)]
    d_own = [float(np.abs(t - tf_solo[R[s]]).sum()) for t, s in zip(tf_mix, plan)]
    d_wrong = [float(np.abs(t - tf_wrong[R[s]]).sum()) for t, s in zip(tf_mix, plan)]
    ok3a = max(tf_err) <= a.tol and all(o * 4 <= w for o, w in zip(d_own, d_wrong))
    divs = [first_div(g[0], m_solo[R[s]][0]) for g, s in zip(g_mix, plan)]
    at_tie = [d is None or (d < len(m_solo[R[s]][1]) and m_solo[R[s]][1][d] < a.tol) for d, s in zip(divs, plan)]
    ok3b = all(at_tie)
    dbg = call(a.url, "esx_steer_debug")
    dbg = dbg["results"][0] if isinstance(dbg, dict) and "results" in dbg else (dbg[0] if isinstance(dbg, list) else {})
    chk = (dbg.get("check_steps"), dbg.get("check_bad_slot"), dbg.get("check_bad_pos"))
    ok3c = chk[0] is None or (chk[1] == 0 and chk[2] == 0)
    ok3 = ok3a and ok3b and ok3c
    print(f"[3a] teacher-forced, mixed batch {plan}: max |dlogp| vs solo per request = "
          f"{[round(e, 3) for e in tf_err]} (tol {a.tol}); sum|d| to own slot {[round(x, 2) for x in d_own]} vs "
          f"to the other row {[round(x, 1) for x in d_wrong]}: {ok3a}")
    print(f"[3b] greedy first divergence vs solo = {divs}; all at near-ties (margin < {a.tol}): {ok3b}; "
          f"token-exact: {all(d is None for d in divs)}")
    print(f"[3c] slot audit (ESX_STEER_CHECK): steps, bad slot, bad pos = {chk}: {ok3c}"
          + ("" if chk[0] is not None else " (server without ESX_STEER_CHECK=1: skipped)"))

    call(a.url, "esx_steer_zero"); reset()
    z = gen(a.url, a.model, 1, "p4-s1")
    ok4 = same(z, base)
    print(f"[4] after zero, slot 1 == base: {ok4}")
    print("debug:", call(a.url, "esx_steer_debug"))
    print("ALL OK" if ok1 and ok2 and ok3 and ok4 else "FAILED")


if __name__ == "__main__":
    main()
