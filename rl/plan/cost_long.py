"""Monte Carlo VM-hours and cost for plan65k1long: two async k=1 pairs (IS / no IS, seeds 0 and 1) at N steps,
evals at steps 20, 40, ..., N (16 samples, v2). Anchored on plan25k1's measurements on a Flex v6e-4
(europe-west4-a, $5.40/h), not on the a-priori model in cost_model.py:

  claim -> end of training, 20 steps   70.6 and 72.7 min  (pA, pB)
  steady s/step                        204.0-210.7 s
  eval of one checkpoint, 2 chips      23.0 and 24.9 min  (run B of pA, pB)
  VM setup (docker, 18 GB cache)       0.14 h
  extra short-lived boots              $0.29 (a Spot grant deleted by the guard)

Each run evaluates its checkpoints on its own two chips right after its training (EVAL_SPLIT=1), so a pair's eval
time is that of its slower run. Steps past 20 have an extra triangular drift factor on s/step (response lengths can
grow or shrink as reward rises), and later checkpoints a longer eval.

    python plan/cost_long.py [--steps 60] [--rate 5.40] [--cap 65]

--mode sync prices one sync pair (sync-s0 | sync-s1 on one VM, plan55sync60). Its s/step is anchored on plan90's
two sync runs (359.6 and 360.0 s/step over 20 steps, each sharing its VM with an async run). The drift range covers
plan65k1long's observed speed-up as answers shortened (steps 41-60 ran at 0.78-0.85x of steps 1-20) and a slow-down
of up to 1.2x if they grow instead.

    python plan/cost_long.py --mode sync --pairs 1 --steps 60 --rate 5.40 --cap 55
"""
import argparse
import json

import numpy as np

D = {
    "s_step": (200.0, 208.0, 216.0),        # steps 1-20, slower run of the pair
    "drift": (0.85, 1.0, 1.25),             # s/step multiplier for steps 21..N
    "train_overhead_h": (0.02, 0.04, 0.10),  # engine startup until steady state
    "eval_h": (0.37, 0.40, 0.45),           # one checkpoint on 2 chips, step 20
    "eval_drift": (0.9, 1.0, 1.3),          # later checkpoints
    "setup_h": (0.12, 0.15, 0.35),          # per VM boot
    "sync_h": (0.02, 0.03, 0.06),           # final rsync and marker writes
    "extra_usd": (0.0, 0.5, 3.0),           # stray grants deleted by the guard, a fast infra-failure release
}


D_SYNC = dict(D, **{
    "s_step": (350.0, 362.0, 380.0),        # plan90 sync-s0/s1: 359.6, 360.0
    "drift": (0.78, 0.95, 1.20),
})


def sim(steps, n=50000, seed=0, d=D):
    rng = np.random.default_rng(seed)
    p = {k: rng.triangular(*v, n) for k, v in d.items()}
    late = max(0, steps - 20)
    train = p["train_overhead_h"] + (min(steps, 20) + late * p["drift"]) * p["s_step"] / 3600
    k = steps // 20 if steps % 20 == 0 else len(range(20, steps + 1, 20)) + 1
    ev = p["eval_h"] * (1 + (k - 1) * p["eval_drift"])
    pair = train + ev + p["sync_h"]
    return pair, p["setup_h"], p["extra_usd"]


def q(x):
    return [round(float(v), 2) for v in np.percentile(x, [10, 50, 90])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--rate", type=float, default=5.40)
    ap.add_argument("--rate2", type=float, default=None, help="second VM's rate (default: --rate)")
    ap.add_argument("--cap", type=float, default=65.0)
    ap.add_argument("--mode", choices=("k1", "sync"), default="k1")
    ap.add_argument("--pairs", type=int, choices=(1, 2), default=2)
    a = ap.parse_args()
    d = D_SYNC if a.mode == "sync" else D
    r2 = a.rate if a.rate2 is None else a.rate2
    out = {"mode": a.mode, "pairs": a.pairs, "steps": a.steps, "rate": a.rate, "rate2": r2, "cap": a.cap}
    pair, setup, extra = sim(a.steps, d=d)
    pair_b, setup_b, _ = sim(a.steps, seed=1, d=d)
    out["pair_h_p10_50_90"] = q(pair)
    if a.pairs == 1:
        out["one_pair_1vm_usd"] = q((setup + pair) * a.rate + extra)
        out["one_pair_1vm_wall_h"] = q(setup + pair)
        out["fits_p90"] = out["one_pair_1vm_usd"][2] <= a.cap
        print(json.dumps(out))
        return
    out["parallel_2vm_usd"] = q((setup + pair) * a.rate + (setup_b + pair_b) * r2 + extra)
    out["sequential_1vm_usd"] = q((setup + pair + pair_b) * a.rate + extra)
    out["sequential_1vm_wall_h"] = q(setup + pair + pair_b)
    out["fits_p90"] = max(out["parallel_2vm_usd"][2], out["sequential_1vm_usd"][2]) <= a.cap
    print(json.dumps(out))


if __name__ == "__main__":
    main()
