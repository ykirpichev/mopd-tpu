"""Charts for the sync-vs-async comparison. Usage: python -m qrl.plots --runs DIR... --evals JSON... --out DIR"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from qrl.analyze import load_events, summarize  # noqa: E402

COLORS = {"sync": "#1f77b4", "async": "#d62728"}


def label(s):
    return f"{s['mode']}{'' if s['mode'] == 'sync' else f' k={s[chr(107)]}'} seed{s['seed']}"


def smooth(x, w=5):
    x = np.asarray(x, float)
    return np.array([x[max(0, i - w + 1):i + 1].mean() for i in range(len(x))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--evals", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    S = [summarize(d) for d in args.runs]
    ls = {0: "-", 1: "--"}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for s in S:
        r = s["reward_curve"]
        t = np.array(s["step_end_times"][-len(r):]) / 60
        kw = dict(color=COLORS[s["mode"]], ls=ls.get(s["seed"], ":"), label=label(s))
        ax[0].plot(range(1, len(r) + 1), smooth(r), **kw)
        ax[1].plot(t, smooth(r), **kw)
    ax[0].set(xlabel="optimizer step", ylabel="train reward (5-step mean)", title="Reward vs step")
    ax[1].set(xlabel="wall-clock (min, steady state)", ylabel="train reward (5-step mean)", title="Reward vs wall-clock")
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "reward_curves.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for s in S:
        t = np.array(s["step_end_times"][-s["steps"]:]) / 60
        ax.plot(range(1, len(t) + 1), t, color=COLORS[s["mode"]], ls=ls.get(s["seed"], ":"), label=label(s))
    ax.set(xlabel="optimizer step", ylabel="wall-clock (min)", title="Wall-clock to N steps")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "wallclock.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(S))
    ax.bar(x - .2, [s["rollout_idle_frac"] * 100 for s in S], .4, label="rollout engine idle", color="#8c564b")
    ax.bar(x + .2, [s["learner_idle_frac"] * 100 for s in S], .4, label="learner idle", color="#7f7f7f")
    ax.set_xticks(x, [label(s) for s in S], fontsize=8)
    ax.set(ylabel="% of steady-state wall-clock", title="Idle fractions")
    ax.legend()
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "idle_fractions.png"), dpi=130)

    fig, axs = plt.subplots(len(S), 1, figsize=(11, 1.3 * len(S) + .6), sharex=True, squeeze=False)
    for a, d, s in zip(axs[:, 0], args.runs, S):
        ev = load_events(d)
        t0 = min(e["start"] for e in ev if e["kind"] == "generate")
        for e in ev:
            if "start" not in e or e["start"] - t0 > 900 or e["end"] < t0:
                continue
            row = {"generate": 1, "load_weights": 1, "train": 0, "publish": 0}.get(e["kind"])
            if row is None:
                continue
            c = {"generate": "#2ca02c", "load_weights": "#ff7f0e", "train": "#9467bd", "publish": "#ff7f0e"}[e["kind"]]
            a.barh(row, e["end"] - e["start"], left=e["start"] - t0, color=c, height=.6)
        a.set_yticks([0, 1], ["learner", "rollout"])
        a.set_title(label(s), fontsize=9, loc="left")
    axs[-1, 0].set_xlabel("seconds since first generation (first 15 min)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "timeline.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, s in enumerate(S):
        h = s["staleness_hist"]
        ks = sorted(int(k) for k in h)
        ax.bar(np.array(ks) + (i - len(S) / 2) * .18, [h[str(k)] for k in ks], .18, label=label(s), color=COLORS[s["mode"]],
               alpha=.5 + .5 * (s["seed"] == 0))
    ax.set(xlabel="staleness (policy versions behind)", ylabel="steps", title="Staleness of trained batches")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "staleness.png"), dpi=130)

    if args.evals:
        rows = []
        for p in args.evals:
            ev = json.load(open(p))
            mode = "async" if "async" in os.path.basename(p) else "sync"
            for lab, r in ev.items():
                rows.append((mode, lab, r["pass@1"]))
        fig, ax = plt.subplots(figsize=(8, 4))
        names = [("base" if lab == "base" else f"{m} {lab}") for m, lab, _ in rows]
        cols = ["#555555" if lab == "base" else COLORS[m] for m, lab, _ in rows]
        ax.bar(range(len(rows)), [v * 100 for _, _, v in rows], color=cols)
        ax.set_xticks(range(len(rows)), names, rotation=30, ha="right", fontsize=8)
        ax.set(ylabel="AIME 2024+2025 pass@1 (%)", title="Eval pass@1 (60 problems x 8 samples)")
        ax.grid(alpha=.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "eval_pass1.png"), dpi=130)
    print("wrote", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
