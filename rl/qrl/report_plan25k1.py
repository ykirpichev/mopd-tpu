"""plan25k1 report: lag 1 (k=1) arms next to plan90's sync and k=2 arms (20 steps, 6,144-token cap, same eval).

Runs report_plan90's build/charts over a combined root (plan90 and plan25k1 VM directories side by side), with
two more arms: k1is (async k=1 + truncated IS) and k1nois (async k=1, no IS). Pass@1 CIs and contrasts use the
same bootstrap as plan90; wall-clock is only compared inside a VM pair.

    python -m qrl.report_plan25k1 --roots results/plan90/raw results/plan25k1/raw \
        --out results/plan25k1/report.json --media DIR
"""
import argparse
import json
import os
import tempfile

import numpy as np

from . import report_plan90 as R90
from .report_v2 import train_series

ARMS = [("sync", "sync"), ("async", "async k=2 + IS"), ("nois", "async k=2, no IS"),
        ("k1is", "async k=1 + IS"), ("k1nois", "async k=1, no IS")]
CONTRASTS = [("k1is", "sync"), ("k1is", "async"), ("k1nois", "k1is"), ("k1nois", "nois"), ("async", "sync")]
PAIRS = {**R90.PAIRS, "pA": ("k1is-s0", "k1nois-s0"), "pB": ("k1is-s1", "k1nois-s1")}
COLORS = {**R90.COLORS, "k1is": "#2b6cb0", "k1nois": "#c53030"}
PRETTY = {"sync": "sync", "async": "k2-IS", "nois": "k2-noIS", "k1is": "k1-IS", "k1nois": "k1-noIS"}


def combined_root(roots):
    """One directory whose children are the VM directories of every root (find_runs globs root/*/*/config.json)."""
    d = tempfile.mkdtemp(prefix="k1root-")
    for r in roots:
        for vm in sorted(os.listdir(r)):
            src = os.path.abspath(os.path.join(r, vm))
            if os.path.isdir(src) and not os.path.exists(os.path.join(d, vm)):
                os.symlink(src, os.path.join(d, vm))
    return d


def patch():
    R90.ARMS, R90.CONTRASTS, R90.PAIRS, R90.COLORS, R90.PRETTY = ARMS, CONTRASTS, PAIRS, COLORS, PRETTY


def lag_kl_chart(rep, media, prefix="k1"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = rep["runs"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 3.8))
    for arm, title in ARMS:
        for n in sorted(x for x in runs if x.rsplit("-s", 1)[0] == arm):
            ls = "-" if n.endswith("s0") else "--"
            y = train_series(runs[n]["run_dir"], "staleness")
            axs[0].plot(np.arange(1, len(y) + 1), y + {"sync": -0.06, "async": 0.0, "nois": 0.06, "k1is": -0.03,
                                                       "k1nois": 0.03}[arm], ls, color=COLORS[arm], label=n)
            kl = train_series(runs[n]["run_dir"], "kl_behav_cur")
            ax = axs[1] if arm in ("nois", "k1nois") else axs[2]
            ax.plot(np.arange(1, len(kl) + 1), kl, ls, color=COLORS[arm], label=n)
    axs[0].set(title="policy versions behind (sampler vs learner), per step", xlabel="step",
               ylabel="versions behind (offset slightly per arm)", yticks=[0, 1, 2])
    axs[1].set(title="no IS: KL(sampler || trainer) on sampled tokens", xlabel="step", yscale="log")
    axs[2].set(title="with IS (and sync): KL(sampler || trainer)", xlabel="step", yscale="log")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(media, f"{prefix}_lag_kl.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def kl_summary(rep):
    """Per run: mean KL over steps 1-10 and 11-20, max KL, and max IS ratio (the plan90 drift signature)."""
    out = {}
    for n, r in rep["runs"].items():
        kl = train_series(r["run_dir"], "kl_behav_cur")
        mx = train_series(r["run_dir"], "behav_ratio_max")
        if len(kl):
            out[n] = {"kl_mean_1_10": float(np.nanmean(kl[:10])), "kl_mean_11_20": float(np.nanmean(kl[10:20])),
                      "kl_max": float(np.nanmax(kl)), "kl_max_step": int(np.nanargmax(kl)) + 1,
                      "behav_ratio_max": float(np.nanmax(mx)) if len(mx) else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--media", default=None)
    a = ap.parse_args()
    patch()
    root = combined_root(a.roots)
    rep = R90.build(root)
    rep["roots"] = a.roots
    rep["kl"] = kl_summary(rep)
    slim = json.loads(json.dumps(rep, default=float))
    json.dump(slim, open(a.out, "w"), indent=1)
    if a.media:
        os.makedirs(a.media, exist_ok=True)
        print("\n".join(R90.charts(rep, a.media, prefix="k1") + [lag_kl_chart(rep, a.media)]))
    for arm, A in rep["arms"].items():
        print(arm, {k: (round(A[k], 4) if isinstance(A.get(k), float) else A.get(k)) for k in
                    ("pass1", "ci95", "per_seed", "gain", "gain_ci95", "s_per_step", "s_per_step_seeds",
                     "staleness_mean", "kl_behav_cur_mean", "truncated_frac_gen")})
    for k, c in rep["contrasts"].items():
        print(k, round(c["diff"] * 100, 2), [round(x * 100, 2) for x in c["ci95"]], "equiv@3pp", c["equivalent"])
    for n, k in sorted(rep["kl"].items()):
        print(n, {x: round(v, 5) if isinstance(v, float) else v for x, v in k.items()})
    print(json.dumps(rep["pairs"], indent=1))


if __name__ == "__main__":
    main()
