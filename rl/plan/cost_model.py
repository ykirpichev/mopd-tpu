"""Monte Carlo cost/time model for the v2 sync-vs-async plan. Anchored on the first run's measurements
(v1: 128 seqs at a 3,072-token cap, 76 s generation, 47 s learner step on ~70k tokens, $5.40/h v6e-4
Flex-start); every extrapolated quantity is a triangular (low, mode, high) distribution.

    python plan/cost_model.py            # prints the tier menu as markdown + JSON
"""
import json
import math

import numpy as np

PRICE_VM_H = 5.40          # ct6e-standard-4t Flex-start, asia-northeast1
CHIPS = 4
V1 = dict(cap=3072, gen_s=76.0, train_s=47.0, train_tok=70.5e3, opt_s=5.0, publish_s=3.0, load_s=2.0)
V1_TRUNC, V1_FIN = 0.89, 1750                               # v1 rollout truncation rate, finished-sample length
THR_EAGER = V1["train_tok"] / (V1["train_s"] - V1["opt_s"])  # learner tok/s, eager, chunk 64, mb 2

PLAN = dict(steps=150, warm_steps=30, warm_cap=4096, cap=8192, prompts=16, group=8,
            eval_problems=136, eval_samples_mid=16, eval_samples_final=32, prompt_tok=250)

DISTS = {
    "ctx_slowdown_8k": (1.0, 1.12, 1.35),     # decode s/step at 8k context vs <=3k (6 of 24 layers attend)
    "learner_speedup": (1.3, 3.0, 4.5),       # chunk 256 + token-budget microbatches vs v1 eager
    "informative_groups": (5.0, 8.5, 11.0),   # of 16, after the difficulty filter, 8k cap
    "trunc_run_8k": (0.15, 0.40, 0.70),       # run-average truncation rate at the 8k cap
    "trunc_run_4k": (0.45, 0.65, 0.85),       # during the 4k warm-up
    "trunc_6k_over_8k": (1.1, 1.25, 1.5),     # truncation at a 6k cap relative to 8k
    "finished_len": (1500, 1900, 2600),       # mean length of samples that finish
    "decode_overhead_share": (0.3, 0.6, 0.9),  # share of a 128-seq decode step that does not shrink with fewer seqs
    "eval_tok_s": (4000, 6500, 9000),         # per chip, 256 concurrent, long caps
    "async_overhead": (1.0, 1.04, 1.10),
    "vm_setup_h": (0.3, 0.5, 0.9),            # boot, docker pull, model + compile cache from GCS, smoke
    "trunc_base_8k": (0.30, 0.55, 0.85),      # pilot profile of the base model
    "entropy_overhead": (1.02, 1.05, 1.10),   # --log-entropy: one extra no-grad lm_head pass, every arm
}


def draw(rng, n):
    return {k: rng.triangular(*v, n) for k, v in DISTS.items()}


def mean_alive(cap, trunc, fin):
    """Average share of a batch's sequences still decoding over the cap steps (finished lengths
    ~ uniform on [0.3, 1.7] x mean, the rest run to the cap)."""
    lo, hi = 0.3 * fin, np.minimum(1.7 * fin, cap)
    finished_area = (1 - trunc) * (lo + (hi - lo) / 2)  # sum over finished seqs of their length
    return (trunc * cap + finished_area) / cap


def decode_factor(p, alive, conc_scale=1.0):
    a = p["decode_overhead_share"]
    return a + (1 - a) * conc_scale * alive


def t_step128(p):
    """s per decode step with 128 live seqs at <=3k context, calibrated on the v1 rollout time."""
    return V1["gen_s"] / (V1["cap"] * decode_factor(p, mean_alive(V1["cap"], V1_TRUNC, V1_FIN)))


def step_times(p, cap, trunc, rollout_scale=1.0):
    """rollout_scale: per-engine concurrency relative to 128 seqs (0.5 = batch split over 2 engines)."""
    f = 1 + (p["ctx_slowdown_8k"] - 1) * max(0, cap - V1["cap"]) / (8192 - V1["cap"])
    alive = mean_alive(cap, trunc, p["finished_len"])
    t128 = p["t128"] if "t128" in p else t_step128(p)
    R = cap * t128 * f * decode_factor(p, alive, rollout_scale) + 3.0
    seq_len = PLAN["prompt_tok"] + trunc * cap + (1 - trunc) * p["finished_len"]
    tok = p["informative_groups"] * PLAN["group"] * seq_len
    L = V1["opt_s"] + tok / (THR_EAGER * p["learner_speedup"]) * p["entropy_overhead"]
    sync = R + L + V1["publish_s"] + V1["load_s"]
    asyn = np.maximum(R + V1["load_s"], L + V1["publish_s"]) * p["async_overhead"]
    return R, L, sync, asyn


def trunc_at(p, cap):
    return p["trunc_run_8k"] if cap >= 8192 else np.minimum(0.95, p["trunc_run_8k"] * p["trunc_6k_over_8k"])


def run_hours(p, mode, rollout_scale=1.0):
    w, n = PLAN["warm_steps"], PLAN["steps"] - PLAN["warm_steps"]
    *_, s4, a4 = step_times(p, PLAN["warm_cap"], p["trunc_run_4k"], rollout_scale)
    *_, s8, a8 = step_times(p, PLAN["cap"], trunc_at(p, PLAN["cap"]), rollout_scale)
    sec = (w * s4 + n * s8) if mode == "sync" else (w * a4 + n * a8)
    return 0.05 + sec / 3600


def eval_hours(p, samples, n_ckpts):
    """Chip-hours to evaluate n_ckpts checkpoints with `samples` per problem on the v2 set."""
    tr = trunc_at(p, PLAN["cap"])
    mean_len = tr * PLAN["cap"] + (1 - tr) * p["finished_len"]
    return n_ckpts * PLAN["eval_problems"] * samples * mean_len / p["eval_tok_s"] / 3600 + 0.04


def vm_hours_pair(p, run_a, run_b, eval_chip_h):
    """One v6e-4 hosting two 1:1 runs (chips 0/1 and 2/3). Chips freed by the shorter run evaluate;
    leftover eval work spreads over all 4 chips after both runs finish."""
    long_, short = np.maximum(run_a, run_b), np.minimum(run_a, run_b)
    free = 2 * (long_ - short)
    return long_ + np.maximum(0, eval_chip_h - free) / CHIPS


def run_evals(p):
    return (eval_hours(p, PLAN["eval_samples_mid"], 2) + eval_hours(p, PLAN["eval_samples_final"], 1))


def pilot(p):
    tok = 919 * 8 * (p["trunc_base_8k"] * 8192 + (1 - p["trunc_base_8k"]) * p["finished_len"])
    profile_h = tok / (3 * p["eval_tok_s"]) / 3600 + 0.05       # chips 0-2; chip 3 runs benches + timing probe
    base_eval_h = eval_hours({**p, "trunc_run_8k": p["trunc_base_8k"]}, PLAN["eval_samples_final"], 1) / CHIPS
    return p["vm_setup_h"] + np.maximum(profile_h, 0.6) + base_eval_h


def tiers(n=20000, seed=0, cap=8192, steps=150):
    PLAN.update(cap=cap, steps=steps)
    rng = np.random.default_rng(seed)
    p = draw(rng, n)
    sync, asyn = run_hours(p, "sync"), run_hours(p, "async")
    ev = run_evals(p)
    core_vm = 3 * (p["vm_setup_h"] + vm_hours_pair(p, sync, asyn, 2 * ev))           # VM i: sync-s_i + async-s_i
    # Sweep: fixed-lag k = 1, 2, 4, 2 seeds each, two async runs per VM (3 VMs).
    sweep_vm = 3 * (p["vm_setup_h"] + vm_hours_pair(p, asyn, asyn, 2 * ev))
    # 2:1 split: one run per VM on chips 0-2 (2 rollout engines + learner), chip 3 evaluates as checkpoints land.
    s2, a2 = run_hours(p, "sync", 0.5), run_hours(p, "async", 0.5)
    split_vm = 2 * (2 * p["vm_setup_h"] + s2 + a2 + np.maximum(0, ev - (s2 + a2)) / CHIPS)
    pil = pilot(p)
    out = {
        "a": {"vm_h": pil + core_vm, "vms_parallel": 3, "wall_h": pil + p["vm_setup_h"] + vm_hours_pair(p, sync, asyn, 2 * ev)},
        "b": {"vm_h": pil + core_vm + sweep_vm, "vms_parallel": 6,
              "wall_h": pil + p["vm_setup_h"] + vm_hours_pair(p, sync, asyn, 2 * ev)},
        "c": {"vm_h": pil + core_vm + sweep_vm + split_vm, "vms_parallel": 8,
              "wall_h": pil + p["vm_setup_h"] + np.maximum(vm_hours_pair(p, sync, asyn, 2 * ev),
                                                           p["vm_setup_h"] + s2 + a2)},
    }
    # No-IS ablation (--behavior-logprobs none). Async runs only, so every ablation VM finishes before
    # the core sync/async VMs; with quota for all VMs at once it adds no wall-clock.
    core_wall = p["vm_setup_h"] + vm_hours_pair(p, sync, asyn, 2 * ev)
    one = p["vm_setup_h"] + vm_hours_pair(p, asyn, 0 * asyn, ev)       # one async run, chips 2/3 evaluate
    two = p["vm_setup_h"] + vm_hours_pair(p, asyn, asyn, 2 * ev)       # two async runs
    abl = {
        # noIS-k2-s{0,1,2}, bounded k=2 like the core async arm: VM (s0, s1) + VM (s2)
        "abl_k2": {"vm_h": two + one, "vms": 2, "wall": np.maximum(two, one)},
        # without (b): noIS-fixed-k4 x3 + IS-fixed-k4 x3, VM i = (noIS-k4-s_i, IS-k4-s_i)
        "abl_k4_standalone": {"vm_h": 3 * two, "vms": 3, "wall": two},
        # with (b), which already has IS-fixed-k4-s{0,1}: noIS-fixed-k4 x3 + IS-fixed-k4-s2 -> 2 VMs
        "abl_k4_in_b": {"vm_h": 2 * two, "vms": 2, "wall": two},
    }
    out["a+abl"] = {"vm_h": out["a"]["vm_h"] + abl["abl_k2"]["vm_h"], "vms_parallel": 5,
                    "wall_h": pil + np.maximum(core_wall, abl["abl_k2"]["wall"])}
    out["a+abl+k4"] = {"vm_h": out["a+abl"]["vm_h"] + abl["abl_k4_standalone"]["vm_h"], "vms_parallel": 8,
                       "wall_h": pil + np.maximum(core_wall, abl["abl_k4_standalone"]["wall"])}
    out["b+abl"] = {"vm_h": out["b"]["vm_h"] + abl["abl_k2"]["vm_h"] + abl["abl_k4_in_b"]["vm_h"],
                    "vms_parallel": 10, "wall_h": out["b"]["wall_h"]}
    out["c+abl"] = {"vm_h": out["c"]["vm_h"] + abl["abl_k2"]["vm_h"] + abl["abl_k4_in_b"]["vm_h"],
                    "vms_parallel": 12, "wall_h": out["c"]["wall_h"]}
    q = lambda x: [float(np.quantile(x, z)) for z in (0.1, 0.5, 0.9)]  # noqa: E731
    cap_of = lambda cost: 5 * math.ceil(float(np.quantile(cost, 0.9)) * 1.15 / 5)  # noqa: E731
    res = {"per_run_h": {"sync": q(sync), "async": q(asyn), "sync_2to1": q(s2), "async_2to1": q(a2)},
           "speedup": q(sync / asyn), "speedup_2to1_async_vs_1to1_sync": q(sync / a2),
           "pilot_vm_h": q(pil), "eval_chip_h_per_run": q(ev)}
    for k, (R, L, s, a) in {"4k": step_times(p, 4096, p["trunc_run_4k"]),
                            "8k": step_times(p, PLAN["cap"], trunc_at(p, PLAN["cap"]))}.items():
        res[f"step_s_{k}"] = {"rollout": q(R), "learner": q(L), "sync": q(s), "async": q(a)}
    for t, v in out.items():
        cost = v["vm_h"] * PRICE_VM_H
        res[t] = {"vm_h": q(v["vm_h"]), "chip_h": q(v["vm_h"] * CHIPS), "usd": q(cost), "wall_h": q(v["wall_h"]),
                  "vms_parallel": v["vms_parallel"],
                  "hard_cap_usd": cap_of(cost)}
    for t, v in abl.items():
        cost = v["vm_h"] * PRICE_VM_H
        res[t] = {"extra_vms": v["vms"], "chip_h": q(v["vm_h"] * CHIPS), "usd": q(cost), "wall_h": q(v["wall"]),
                  "hard_cap_usd": cap_of(cost), "p_extends_core_wall": float(np.mean(v["wall"] > core_wall))}
    return res


def plan50(cap=3072, steps=60, n=20000, seed=0, vms=1, learner_speedup=None, filt=True, trunc=None,
           finished_len=None, eval_tok_s=None, t128=None, informative=None, spent_usd=None, setups_left=None,
           price_vm_h=PRICE_VM_H, samples=False):
    """$50-total plan: one v6e-4 runs three 1:1 pairs back to back -- (sync-s0, async-s0), (sync-s1, async-s1),
    (noIS-s0, noIS-s1); async arms are fixed-lag k=2. Mini-pilot first: 3-chip prompt profile at the training
    cap, a 2-config learner bench on chip 3, then a 16-sample base eval. Final-step eval only (16 samples).

    Measured pilot inputs replace the priors: trunc (base truncation rate at the cap), finished_len (mean length
    of finished samples), eval_tok_s (bulk decode tok/s per chip), t128 (s per decode step, 128 live seqs),
    informative (expected informative groups of 16 after the filter). With spent_usd, the pilot is replaced by
    what was actually spent and only setups_left VM setups are added; "usd" is then the projected total."""
    PLAN.update(cap=cap, steps=steps, warm_steps=0, warm_cap=cap)
    rng = np.random.default_rng(seed)
    p = draw(rng, n)
    p["learner_speedup"] = (rng.triangular(1.0, 1.5, 3.0, n)      # chunk-256 patch untested on TPU; 1.0 = eager
                            if learner_speedup is None else np.full(n, float(learner_speedup)))
    if not filt:                                                   # v1 measured 3.6/16 without the filter
        p["informative_groups"] = rng.triangular(3.0, 4.0, 5.5, n)
    if informative is not None:
        p["informative_groups"] = rng.triangular(0.75 * informative, informative, 1.15 * informative, n)
    if trunc is None:
        tr = np.minimum(0.95, rng.triangular(0.70, 0.83, 0.92, n) * (3072 / cap) ** 0.5)
    else:                                   # training moves truncation a little either way over 60 steps at lr 1e-6
        tr = np.minimum(0.97, rng.triangular(0.8 * trunc, trunc, min(0.98, 1.1 * trunc) + 1e-9, n))
    if finished_len is not None:
        p["finished_len"] = rng.triangular(0.85 * finished_len, finished_len, 1.15 * finished_len, n)
    if eval_tok_s is not None:
        p["eval_tok_s"] = rng.triangular(0.9 * eval_tok_s, eval_tok_s, 1.1 * eval_tok_s, n)
    if t128 is not None:
        p["t128"] = rng.triangular(0.95 * t128, t128, 1.15 * t128, n)
    p["trunc_run_8k"] = tr                                         # trunc_at() reads this key for cap >= 8192 only
    _, _, s, a = step_times(p, cap, tr)
    sync, asyn = 0.05 + steps * s / 3600, 0.05 + steps * a / 3600
    mean_len = tr * cap + (1 - tr) * p["finished_len"]
    ev = PLAN["eval_problems"] * 16 * mean_len / p["eval_tok_s"] / 3600 + 0.04
    prof = 919 * 8 * mean_len / (3 * p["eval_tok_s"]) / 3600 + 0.05
    pil = np.maximum(prof, 0.35) + ev / CHIPS
    pairs = vm_hours_pair(p, sync, asyn, 2 * ev) * 2 + vm_hours_pair(p, asyn, asyn, 2 * ev)
    vm_h = vms * p["vm_setup_h"] + pil + pairs
    usd = vm_h * price_vm_h
    if spent_usd is not None:
        usd = spent_usd + ((vms if setups_left is None else setups_left) * p["vm_setup_h"] + pairs) * price_vm_h
    q = lambda x: [float(np.quantile(x, z)) for z in (0.1, 0.5, 0.9)]  # noqa: E731
    pair_sa = vm_hours_pair(p, sync, asyn, 2 * ev)
    pair_aa = vm_hours_pair(p, asyn, asyn, 2 * ev)
    extra = {"_samples": {"pair_sa": pair_sa, "pair_aa": pair_aa, "setup_h": p["vm_setup_h"]}} if samples else {}
    return {**extra, "cap": cap, "steps": steps, "vm_h": q(vm_h), "usd": q(usd), "price_vm_h": price_vm_h,
            "wall_h_1vm": q(vm_h), "sync_h": q(sync), "async_h": q(asyn), "speedup": q(sync / asyn),
            "pair_sync_async_h": q(pair_sa), "pair_async_async_h": q(pair_aa), "pilot_vm_h": q(pil),
            "p_over_50": float(np.mean(usd > 50)), "p_over_45": float(np.mean(usd > 45)),
            "p_over_80": float(np.mean(usd > 80)), "p_over_90": float(np.mean(usd > 90))}


TIERS = ("a", "a+abl", "a+abl+k4", "b", "b+abl", "c", "c+abl")
ABL = ("abl_k2", "abl_k4_standalone", "abl_k4_in_b")


def menu():
    rows = []
    for cap in (6144, 8192):
        for steps in (150, 200):
            r = tiers(cap=cap, steps=steps)
            rows.append({"cap": cap, "steps": steps, **{t: r[t] for t in TIERS},
                         "speedup": r["speedup"], "per_run_h": r["per_run_h"], "step_s_8k": r["step_s_8k"]})
    return rows


def fmt(q, money=False):
    f = (lambda x: f"${x:,.0f}") if money else (lambda x: f"{x:,.1f}")
    return f"{f(q[1])} ({f(q[0])}-{f(q[2])})"


if __name__ == "__main__":
    import sys

    if "--json" in sys.argv:
        print(json.dumps(tiers(), indent=1))
    else:
        print("| cap | steps | tier | VMs in parallel | chip-h P50 (P10-P90) | wall-clock h | $ P50 (P10-P90) | hard cap |")
        print("|---|---|---|---|---|---|---|---|")
        for row in menu():
            for t in TIERS:
                v = row[t]
                print(f"| {row['cap']} | {row['steps']} | {t} | {v['vms_parallel']} | {fmt(v['chip_h'])} | "
                      f"{fmt(v['wall_h'])} | {fmt(v['usd'], True)} | ${v['hard_cap_usd']} |")
        r = tiers()
        print("\n8k/150 ablation increments:\n")
        print("| option | extra VMs | chip-h | own wall h | $ P50 (P10-P90) | hard cap | P(extends core wall) |")
        print("|---|---|---|---|---|---|---|")
        for t in ABL:
            v = r[t]
            print(f"| {t} | {v['extra_vms']} | {fmt(v['chip_h'])} | {fmt(v['wall_h'])} | {fmt(v['usd'], True)} | "
                  f"${v['hard_cap_usd']} | {v['p_extends_core_wall']:.3f} |")
        print("\n8k/150 detail:", json.dumps({k: r[k] for k in ("per_run_h", "speedup", "speedup_2to1_async_vs_1to1_sync",
                                                               "step_s_4k", "step_s_8k", "pilot_vm_h",
                                                               "eval_chip_h_per_run")}, indent=1))
