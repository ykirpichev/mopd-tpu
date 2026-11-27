"""Descriptive tables for the MOPD paper that need the learner logs: T2 compute, T4 clip analysis,
seed spread per arm, and SFT forward-KL curves.

    python -m qrl.paper_extras --roots results/mopdfull/raw results/mopdpaper/raw \
        --paper results/mopdpaper/paper.json --out results/mopdpaper/extras.json
"""
import argparse
import glob
import json
import os
import statistics as st

from qrl.report_paper import arm_of

PRICE = 5.40
THRESH = ("adv_abs_gt0.25", "adv_abs_gt0.5", "adv_abs_gt1", "adv_abs_gt2")


def runs(roots):
    """{run: learner train rows} from the latest boot holding each run."""
    out = {}
    for root in roots:
        for f in sorted(glob.glob(os.path.join(root, "*", "*", "events", "learner.jsonl"))):
            run = f.split(os.sep)[-3]
            rows = [json.loads(l) for l in open(f) if '"train"' in l]
            rows = [r for r in rows if r.get("kind") == "train"]
            if rows:
                out[run] = sorted(rows, key=lambda r: r["step"])
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.fmean(xs) if xs else None


def compute(R, vmh):
    t2 = {}
    for run, rows in R.items():
        ends = [r["end"] for r in rows]
        dt = [b - a for a, b in zip(ends, ends[1:])]
        t2[run] = {
            "steps": len(rows),
            "s_per_step_median": st.median(dt) if dt else None,
            "train_tokens_M": sum(r.get("train_tokens", 0) for r in rows[:60]) / 1e6,
            "vm_h_step60": vmh.get(run, {}).get("step60"),
            "usd_step60": (vmh.get(run, {}).get("step60") or 0) * PRICE or None,
        }
        if "step120" in vmh.get(run, {}):
            t2[run]["train_tokens_M_120"] = sum(r.get("train_tokens", 0) for r in rows) / 1e6
            t2[run]["vm_h_step120"] = vmh[run]["step120"]
            t2[run]["usd_step120"] = vmh[run]["step120"] * PRICE
    t4 = {}
    for run, rows in R.items():
        if "adv_clip_frac" not in rows[0]:
            continue
        t4[run] = {"adv_clip_frac_mean": mean(r.get("adv_clip_frac") for r in rows),
                   "adv_clip_frac_max": max(r.get("adv_clip_frac", 0) for r in rows)}
        for k in THRESH:
            if k in rows[0]:
                v = [r[k] for r in rows if k in r]
                t4[run][k + "_mean"] = mean(v)
                t4[run][k + "_max"] = max(v)
    sft = {}
    for run, rows in R.items():
        if rows[0].get("objective") != "sft":
            continue
        sft[run] = {d: {"first5": mean(r.get(f"fkl_teacher{d}") for r in rows[:5]),
                        "last5_of_60": mean(r.get(f"fkl_teacher{d}") for r in rows[55:60]),
                        "last5_of_120": mean(r.get(f"fkl_teacher{d}") for r in rows[-5:])}
                    for d in ("", "_math", "_if")}
    return t2, t4, sft


def seed_spread(P):
    out = {}
    for kind, arms in P["evals"].items():
        for arm, e in arms.items():
            pr = e.get("per_row") or []
            if len(pr) > 1 and arm in P.get("seeds", {}):
                out.setdefault(arm, {})[kind] = {"sd_pp": 100 * st.stdev(pr), "range_pp": 100 * (max(pr) - min(pr)),
                                                 "rows": len(pr)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--paper", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    P = json.load(open(a.paper))
    t2, t4, sft = compute(runs(a.roots), P.get("vm_hours", {}))
    by_arm = {}
    for run, v in t2.items():
        by_arm.setdefault(arm_of(run), []).append(v)
    t2_arm = {arm: {k: mean(v.get(k) for v in vs) for k in vs[0]} | {"runs": len(vs)} for arm, vs in by_arm.items()}
    out = {"t2_runs": t2, "t2_arms": t2_arm, "t4_clip": t4, "sft_fkl": sft, "seed_spread": seed_spread(P)}
    json.dump(out, open(a.out, "w"), indent=1)
    for arm, v in sorted(t2_arm.items()):
        print(f"T2 {arm:12s} runs={v['runs']} s/step={v['s_per_step_median'] or 0:.0f} tokM={v['train_tokens_M']:.1f} "
              f"vmh60={v['vm_h_step60'] or 0:.2f} usd60={v['usd_step60'] or 0:.2f}")
    for run, v in sorted(t4.items()):
        print("T4", run, {k: f"{x:.2e}" for k, x in v.items()})
    for run, v in sorted(sft.items()):
        print("SFT", run, {d or "all": {k: round(x, 4) for k, x in s.items() if x is not None} for d, s in v.items()})
    for arm, v in sorted(out["seed_spread"].items()):
        print("SEED", arm, {k: f"sd {x['sd_pp']:.2f} range {x['range_pp']:.2f} n={x['rows']}" for k, x in v.items()})


if __name__ == "__main__":
    main()
