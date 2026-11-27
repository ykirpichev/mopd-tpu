"""Monte Carlo VM-hours, wall-clock and cost for the MOPD plan (docs/mopd-tpu-plan.md) on v6e-4 VMs, max 2 at once.

Anchors (measured, plan65k1long / plan55sync60 on Flex v6e-4, europe-west4-a): async lag-1 + IS GRPO on AIME,
128 seqs/step at a 6,144-token cap, 147.6 s generation per batch, 176.4 s train per step on ~320-340k trained tokens
(about 2,000 tok/s fwd+bwd with grad checkpointing, host AdamW), 186.6 s/step over 60 steps; one eval of 136 x 16 at
6,144 tokens takes 0.37-0.45 h on 2 chips. The math-teacher job reuses cost_long.sim (that same run, re-anchored).
Everything else is a triangular (low, mode, high) prior: mixed math/IF batches, teacher prefill throughput, IF lengths.

Layouts (chips of one v6e-4):
  teachers  0/1 math GRPO teacher, 2/3 IF GRPO teacher (one VM, both at once)
  mixrl     0/1 Mix-RL GRPO seed 0, 2/3 seed 1 (one VM, both at once)
  mopd      0 student vLLM, 1 learner, 2 both 2B teachers (prefill service), 3 evals as checkpoints land
  mopd_dist same as mopd; the teacher ships final hidden states, the learner also holds both teachers' lm_heads
  opd9b     0 student vLLM, 1 learner, 2/3 Qwen3.5-9B teacher (prefill split over two chips), evals after training

    cd rl && python -m plan.cost_mopd                 # full plan and pilot, at Flex and Spot rates, JSON
    cd rl && python -m plan.cost_mopd --markdown      # the same as tables
"""
import argparse
import json

import numpy as np

from plan.cost_long import D as LONG, sim as long_sim

RATES = {                                    # USD per v6e-4 VM-hour, rl/tpu/plan65k1long.json
    "flex": 5.40,                            # Flex-start, every CT6E region with quota
    "spot_asia_southeast1": 1.07,            # cheapest Spot list price; capacity never tried
    "spot_us_south1": 2.15,
    "spot_granted_before": 6.67,             # mean of asia-south1 6.25 and europe-west4 7.13, where plan90 got Spot
}
HAZARD_H = (0.0, 0.05, 0.15)                 # Spot preemptions per VM-hour (plan.reprice)
REBOOT_H = 0.1

P = {
    "gen_mix_s": (110.0, 135.0, 165.0),      # 8 math + 8 IF prompts x 8: math tails set the time (147.6 s all-math)
    "gen_if_s": (25.0, 45.0, 80.0),          # all-IF batch, answers end well before the cap
    "tok_math": (4000.0, 4700.0, 5500.0),    # prompt + completion per math sequence, 60-step mean
    "tok_if": (450.0, 900.0, 1600.0),
    "inf_math": (0.45, 0.56, 0.65),          # share of math groups with nonzero GRPO advantage (~9/16 measured)
    "inf_if": (0.50, 0.75, 0.90),
    "learn_tok_s": (1800.0, 2000.0, 2200.0),  # learner fwd+bwd tokens/s, one chip
    "opt_s": (4.0, 6.0, 9.0),                # host AdamW + transfers per step
    "teach_tok_s": (5000.0, 7500.0, 11000.0),  # 2B teacher forward-only (prefill) tokens/s, one chip
    "r9b": (3.2, 3.9, 4.6),                  # 9B / 2B prefill time per token
    "async_oh": (1.0, 1.04, 1.10),
    "drift": (0.85, 0.95, 1.15),             # s/step multiplier for steps 21..N
    "startup_h": (0.03, 0.05, 0.12),         # engines + teacher service up to steady state
    "aime_eval_h": LONG["eval_h"],           # one checkpoint, 2 chips
    "eval_drift": LONG["eval_drift"],
    "if_eval_h": (0.08, 0.13, 0.25),         # IFEval 541 + IFBench 300 prompts x 4 samples, 2 chips
    "setup_h": LONG["setup_h"],
    "probe_h": (0.6, 0.9, 1.5),              # base evals, 4B/9B teacher probe (KL, prefill tok/s, 9B AIME x4)
    "debug_h": (0.3, 1.0, 3.0),              # first bring-up of the teacher service and OPD loss on TPU
    "tail_h": LONG["sync_h"],
    # Full-distribution arm (MOPD-dist): the teacher ships its final hidden state, the learner rebuilds the teacher
    # logits with the teacher's lm_head and trains on the exact per-position reverse KL over the vocabulary.
    "dist_learn_x": (1.05, 1.15, 1.35),      # learner time: teacher lm_head fwd (+1.0 of ~18 GFLOP/token) and
                                             # the full-vocab KL + its backward (memory-bound, fp32, chunked)
    "dist_xfer_s": (1.0, 3.0, 8.0),          # per step, not overlapped: ~1.5 GB of hidden states D2H, /dev/shm, H2D
    "dist_debug_h": (0.2, 0.5, 1.5),         # bring-up of the second loss and the hidden-state channel
}
RECREATE_GAP_H = 0.5                         # a VM that would idle longer waiting for the teachers is deleted, then
                                             # re-created (one more setup) when they are ready

VOCAB = 248320                               # Qwen3.5 text_config.vocab_size (2B and 9B)
HIDDEN_2B = 2048                             # Qwen3.5-2B hidden size; lm_head tied to embed_tokens


def draw(rng, n):
    return {k: rng.triangular(*v, n) for k, v in P.items()}


def q(x):
    return [round(float(v), 2) for v in np.percentile(x, [10, 50, 90])]


def step_s(p, arm):
    """Steady-state seconds per lag-1 step for a mixed 50/50 math/IF batch of 128 sequences."""
    tok_all = 64 * p["tok_math"] + 64 * p["tok_if"]
    if arm == "mixrl":
        learn = p["opt_s"] + (64 * p["tok_math"] * p["inf_math"] + 64 * p["tok_if"] * p["inf_if"]) / p["learn_tok_s"]
        return np.maximum(p["gen_mix_s"] + 2, learn + 3) * p["async_oh"]
    learn = p["opt_s"] + tok_all / p["learn_tok_s"]      # distillation trains every sequence
    if arm == "mopd_dist":
        learn = learn * p["dist_learn_x"] + p["dist_xfer_s"]
    if arm in ("mopd", "mopd_dist"):
        teach = tok_all / p["teach_tok_s"]
    else:                                                # opd9b: prefill split over two chips
        teach = tok_all * p["r9b"] / (2 * p["teach_tok_s"])
    # Teacher prefill sits between generation and training; with lag 1 it is hidden unless it is the slowest stage.
    return np.maximum.reduce([p["gen_mix_s"] + 2, learn + 3, teach]) * p["async_oh"]


def train_h(p, arm, steps):
    s = step_s(p, arm)
    return p["startup_h"] + (min(steps, 20) + max(0, steps - 20) * p["drift"]) * s / 3600


def eval_ckpt_h(p, k):
    """k checkpoints, AIME + IF evals, on 2 chips."""
    return (p["aime_eval_h"] * (1 + (k - 1) * p["eval_drift"])) + k * p["if_eval_h"]


def transfer_table(tokens=64 * 4700 + 64 * 900):
    """Bytes per token and per 128-sequence mixed batch for each way of shipping the teacher's signal."""
    rows = {
        "sampled-token logprob, fp32": 4,
        "top-64: int32 id + bf16 logprob": 64 * 6,
        "top-256: int32 id + bf16 logprob": 256 * 6,
        "final hidden state, bf16 (2B teacher)": HIDDEN_2B * 2,
        "full logits, bf16": VOCAB * 2,
        "full logits, fp32": VOCAB * 4,
    }
    return {"tokens_per_batch": tokens,
            "rows": {k: {"bytes_per_token": b, "gb_per_batch": round(b * tokens / 1e9, 4)} for k, b in rows.items()},
            "teacher_lm_head_gb_bf16": round(VOCAB * HIDDEN_2B * 2 / 1e9, 3),
            "fp32_logits_gb_per_512_token_chunk": round(512 * VOCAB * 4 / 1e9, 3)}


def jobs(p, n, steps, teacher_steps, seed, pilot, opd9b_arm=True):
    """Jobs as (lane, name, hours, needs_teachers), run in list order per lane. Lane 0 and 1 are the two VMs."""
    pair, _, _ = long_sim(teacher_steps, n=n, seed=seed)            # math teacher, incl. its AIME evals
    k_teacher = teacher_steps // 20 if teacher_steps % 20 == 0 else teacher_steps // 20 + 1
    if_teacher = p["startup_h"] + teacher_steps * np.maximum(p["gen_if_s"] + 2, p["opt_s"] + 3 + 128 * p["tok_if"]
                                                             * p["inf_if"] / p["learn_tok_s"]) * p["async_oh"] / 3600
    # Teachers also get IF evals (math teacher) and AIME + IF evals of the final IF teacher.
    teachers = np.maximum(pair + p["if_eval_h"], if_teacher + eval_ckpt_h(p, 1)) + p["tail_h"]
    k = 1 if pilot else (steps // 20 if steps % 20 == 0 else steps // 20 + 1)
    mix_one = train_h(p, "mixrl", steps) + eval_ckpt_h(p, k)
    # MOPD: checkpoints before the last are evaluated on chip 3 during training (hidden); the last on 4 chips.
    mopd = train_h(p, "mopd", steps) + eval_ckpt_h(p, 1) / 2 + p["tail_h"]
    dist = train_h(p, "mopd_dist", steps) + eval_ckpt_h(p, 1) / 2 + p["tail_h"]
    opd9b = train_h(p, "opd9b", steps) + eval_ckpt_h(p, k) / 2 + p["tail_h"]
    out = [(0, "setup", p["setup_h"], False), (0, "debug", p["debug_h"] + p["dist_debug_h"], False),
           (0, "teachers", teachers, False), (1, "setup", p["setup_h"], False), (1, "probe", p["probe_h"], False)]
    if pilot:
        # VM1: teachers, MOPD s0. VM2: probe, Mix-RL s0 (chips 0/1, no partner), then MOPD-dist s0 once the
        # teachers exist (VM2 is usually deleted and re-created for it).
        out += [(0, "mopd_s0", mopd, True), (1, "mixrl_s0", mix_one + p["tail_h"], False),
                (1, "mopd_dist_s0", dist, True)]
    else:
        # VM1: teachers, MOPD s0, MOPD-dist s0, [OPD-9B s0]. VM2: probe, Mix-RL s0+s1 pair, [OPD-9B s1, which
        # needs no trained teacher, so VM2 does not wait], MOPD s1, MOPD-dist s1.
        out += [(1, "mixrl_pair", mix_one + p["tail_h"], False)]
        if opd9b_arm:
            out += [(1, "opd9b_s1", opd9b, False)]
        out += [(0, "mopd_s0", mopd, True), (0, "mopd_dist_s0", dist, True),
                (1, "mopd_s1", mopd, True), (1, "mopd_dist_s1", dist, True)]
        if opd9b_arm:
            out += [(0, "opd9b_s0", opd9b, False)]
    return out


def price(js, rate, n, rng, spot):
    """Cost and wall samples. Jobs run in list order on their lane; a job that needs the teachers starts when
    the teachers job has ended. A wait up to RECREATE_GAP_H is billed as idle; a longer one deletes the VM and
    re-creates it (one setup, billed) when the teachers are ready. On Spot a job is preempted at most once (hazard
    HAZARD_H); it does not resume, so the elapsed part is lost and the job re-runs after a reboot and setup."""
    vmh, t = np.zeros(n), [np.zeros(n), np.zeros(n)]
    teach_end = np.full(n, np.inf)
    lam = rng.triangular(*HAZARD_H, n) if spot else np.zeros(n)
    for lane, name, h, needs in js:
        h = np.asarray(h, dtype=float) * np.ones(n)
        if needs:
            gap = np.maximum(0.0, teach_end - t[lane])
            recreate = gap > RECREATE_GAP_H
            setup = rng.triangular(*P["setup_h"], n)
            vmh += np.where(recreate, setup, gap)
            t[lane] = np.where(recreate, teach_end + setup, t[lane] + gap)
        extra = np.zeros(n)
        if spot and name != "setup":
            hit = rng.random(n) < 1 - np.exp(-lam * h)
            extra = np.where(hit, rng.random(n) * h + REBOOT_H + rng.triangular(*P["setup_h"], n), 0.0)
        vmh += h + extra
        t[lane] = t[lane] + h + extra
        if name == "teachers":
            teach_end = t[lane].copy()
    cost = vmh * rate + rng.triangular(*LONG["extra_usd"], n)
    return {"usd": q(cost), "vm_h": q(vmh), "wall_h": q(np.maximum(*t))}


def plan(steps, teacher_steps, pilot, n=50000, seed=0, opd9b_arm=True):
    rng = np.random.default_rng(seed)
    p = draw(rng, n)
    js = jobs(p, n, steps, teacher_steps, seed, pilot, opd9b_arm)
    res = {"steps": steps, "teacher_steps": teacher_steps, "pilot": pilot, "opd9b_arm": opd9b_arm,
           "s_step": {a: q(step_s(p, a)) for a in ("mixrl", "mopd", "mopd_dist", "opd9b")},
           "job_h": {}}
    for _, name, h, _ in js:
        if name not in res["job_h"]:
            res["job_h"][name] = q(np.asarray(h) * np.ones(n))
    for r, rate in RATES.items():
        res[r] = price(js, rate, n, np.random.default_rng(seed + 7), spot=r.startswith("spot"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--teacher-steps", type=int, default=60)
    ap.add_argument("--pilot-steps", type=int, default=30)
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    full = plan(a.steps, a.teacher_steps, pilot=False)
    core = plan(a.steps, a.teacher_steps, pilot=False, opd9b_arm=False)
    pil = plan(a.pilot_steps, a.pilot_steps, pilot=True)
    if not a.markdown:
        print(json.dumps({"full": full, "core_no_opd9b": core, "pilot": pil, "transfer": transfer_table()}, indent=1))
        return
    tt = transfer_table()
    print(f"| teacher signal ({tt['tokens_per_batch']:,} tokens/batch) | bytes/token | GB/batch |")
    print("|---|---|---|")
    for k, v in tt["rows"].items():
        print(f"| {k} | {v['bytes_per_token']:,} | {v['gb_per_batch']:,} |")
    print(f"\nteacher lm_head bf16 {tt['teacher_lm_head_gb_bf16']} GB; fp32 logits per 512-token chunk "
          f"{tt['fp32_logits_gb_per_512_token_chunk']} GB\n")
    print("| option | rate | $/VM-h | VM-h P50 (P10-P90) | wall h P50 (P90) | $ P50 | $ P90 |")
    print("|---|---|---|---|---|---|---|")
    for label, r in (("full", full), ("core (no OPD-9B)", core), ("pilot", pil)):
        for k, rate in RATES.items():
            v = r[k]
            print(f"| {label} | {k} | {rate:.2f} | {v['vm_h'][1]:.1f} ({v['vm_h'][0]:.1f}-{v['vm_h'][2]:.1f}) | "
                  f"{v['wall_h'][1]:.1f} ({v['wall_h'][2]:.1f}) | ${v['usd'][1]:.0f} | ${v['usd'][2]:.0f} |")
    for label, r in (("full", full), ("pilot", pil)):
        print(f"\n{label}: s/step P10/50/90 {json.dumps(r['s_step'])}")
        print(f"{label}: job hours P10/50/90 {json.dumps(r['job_h'])}")


if __name__ == "__main__":
    main()
