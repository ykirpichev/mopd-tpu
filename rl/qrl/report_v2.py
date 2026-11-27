"""v2 report: summaries, pass@1 with hierarchical-bootstrap CIs, speedup CIs and charts for the two
budget-capped runs. Expects the GCS results layout: ROOT/<vm>/<run-name>/{events/,config.json,eval.stepN.json}
and ROOT/<vm>/pilot/eval_base.json. Run names are <arm>-s<seed>.

    python -m qrl.report_v2 --root results/v2 --out results/v2/report.json --media DIR
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from . import stats
from .analyze import load_events, summarize

ARMS = {  # $50 plan: one shared async-IS arm; run1 = longer sync vs async, run2 = IS ablation at the same lag
    "run1": [("sync", "sync"), ("async", "async fixed k=2, IS")],
    "run2": [("async", "async fixed k=2, IS"), ("nois", "async fixed k=2, no IS")],
}
TITLES = {"run1": "Longer run: sync vs async", "run2": "Ablation: IS vs no IS (fixed lag 2)"}
COLORS = {"sync": "#5a645c", "async": "#0f6b4c", "nois": "#b85c28"}


def find_runs(root):
    runs = {}
    for cfg in glob.glob(os.path.join(root, "*", "*", "config.json")):
        d = os.path.dirname(cfg)
        m = re.match(r"(.+)-s(\d+)$", os.path.basename(d))
        if m and os.path.isdir(os.path.join(d, "events")):
            runs.setdefault(m.group(1), {})[int(m.group(2))] = d
    return runs


def base_eval(root):
    p = sorted(glob.glob(os.path.join(root, "*", "pilot", "eval_base.json")))
    return p[0] if p else None


def train_series(run_dir, key):
    ev = sorted((e for e in load_events(run_dir) if e["kind"] == "train"), key=lambda e: e["step"])
    return np.array([e.get(key, np.nan) if key != "grad_norm_max_u" else max(e.get("grad_norms", [np.nan]))
                     for e in ev], float)


def arm_curve(dirs, key):
    xs = [train_series(d, key) for d in dirs]
    n = min(len(x) for x in xs)
    a = np.stack([x[:n] for x in xs])
    return a.mean(0), a.min(0), a.max(0)


def eval_paths(dirs, label):
    out = []
    for d in dirs:
        p = os.path.join(d, f"eval.{label}.json")
        if os.path.exists(p):
            out.append(f"{p}::{label}")
    return out


def arm_vs_base(paths, base_path, iters=4000):
    ids, cells = stats.load_cells({"arm": paths, "base": [f"{base_path}::base"]}, None)
    (ca, n), (cb, nb) = cells["arm"], cells["base"]
    pa, pb = ca / n, cb / nb
    rng = np.random.default_rng(1)
    S, P = pa.shape
    draws = np.empty(iters)
    for t in range(iters):
        prob = rng.integers(0, P, P)
        ra = rng.binomial(n, pa[rng.integers(0, S, S)][:, prob]) / n
        rb = rng.binomial(nb, pb[:, prob]) / nb
        draws[t] = ra.mean() - rb.mean()
    return {"pass1": float(pa.mean()), "base": float(pb.mean()), "gain": float(pa.mean() - pb.mean()),
            "gain_ci95": stats.ci(draws)}


def arm_ci(paths, iters=4000, seed=0):
    """CI of an arm's own pass@1 (problems, seeds, samples resampled)."""
    _, cells = stats.load_cells({"a": paths}, None)
    c, n = cells["a"]
    rng = np.random.default_rng(seed)
    p = c / n
    S, P = p.shape
    draws = np.empty(iters)
    for t in range(iters):
        r = p[rng.integers(0, S, S)][:, rng.integers(0, P, P)]
        draws[t] = (rng.binomial(n, r) / n).mean()
    return float(p.mean()), stats.ci(draws), [float(x) for x in p.mean(1)]


def build(root, margin=0.03):
    runs = find_runs(root)
    base = base_eval(root)
    rep = {"base_eval": base, "runs": {}}
    if base:
        b = json.load(open(base))["base"]
        rep["base_pass1"] = b["pass@1"]
        rep["base_truncated_frac"] = b["truncated_frac"]
    for run, arms in ARMS.items():
        R = {"arms": {}}
        for arm, title in arms:
            seeds = runs.get(arm, {})
            dirs = [seeds[s] for s in sorted(seeds)]
            if not dirs:
                continue
            summ = [summarize(d) for d in dirs]
            A = {"title": title, "seeds": sorted(seeds), "dirs": dirs,
                 "summaries": [{k: v for k, v in s.items() if not k.endswith("_curve") and k != "step_end_times"}
                               for s in summ]}
            for k in ("s_per_step", "wall_s", "rollout_idle_frac", "learner_idle_frac", "gen_tok_per_s",
                      "staleness_mean", "kl_behav_cur_mean", "grad_norm_mean", "grad_norm_max", "entropy_mean",
                      "behav_ratio_max", "behav_would_truncate_frac_mean", "clip_frac_mean", "truncated_frac_gen",
                      "informative_groups_mean", "train_s_mean", "gen_s_mean"):
                vals = [s[k] for s in summ if k in s]
                if vals:
                    A[k] = float(np.mean(vals))
                    A[k + "_seeds"] = [float(v) for v in vals]
            A["staleness_hist"] = {}
            for s in summ:
                for kk, v in s.get("staleness_hist", {}).items():
                    A["staleness_hist"][kk] = A["staleness_hist"].get(kk, 0) + v
            A["complete"] = [s["complete"] for s in summ]
            labels = sorted({os.path.basename(p)[5:-5] for d in dirs for p in glob.glob(os.path.join(d, "eval.step*.json"))
                             if re.match(r"eval\.step\d+\.json$", os.path.basename(p))}, key=lambda x: int(x[4:]))
            A["evals"] = {}
            for lab in labels:
                ps = eval_paths(dirs, lab)
                if not ps:
                    continue
                m, c, per_seed = arm_ci(ps)
                e = {"pass1": m, "ci95": c, "per_seed": per_seed, "n_seeds": len(ps)}
                if base:
                    e.update({k: v for k, v in arm_vs_base(ps, base).items() if k != "pass1"})
                A["evals"][lab] = e
            R["arms"][arm] = A
        names = [a for a, _ in arms if a in R["arms"]]
        if len(names) == 2:
            a, b = names[1], names[0]
            common = sorted(set(R["arms"][a]["evals"]) & set(R["arms"][b]["evals"]), key=lambda x: int(x[4:]))
            R["compare"] = {}
            for lab in common:
                pa, pb = eval_paths(R["arms"][a]["dirs"], lab), eval_paths(R["arms"][b]["dirs"], lab)
                R["compare"][lab] = stats.compare({a: pa, b: pb}, lab, a=a, b=b, margin=margin, iters=6000)
            sa, sb = R["arms"][a].get("s_per_step_seeds"), R["arms"][b].get("s_per_step_seeds")
            if sa and sb:
                r, c = stats.seed_ratio_ci(sa, sb)
                R["speedup_b_over_a"] = {"ratio": r, "ci95": c, "a": a, "b": b}
            for key in ("reward_mean", "kl_behav_cur", "entropy", "grad_norm"):
                xs = {}
                for arm in names:
                    s = [train_series(d, key) for d in R["arms"][arm]["dirs"]]
                    n = min(len(x) for x in s)
                    xs[arm] = np.stack([x[:n] for x in s])
                n = min(x.shape[1] for x in xs.values())
                tail = max(1, n // 3)
                d = xs[a][:, n - tail:n].mean(1).mean() - xs[b][:, n - tail:n].mean(1).mean()
                R.setdefault("last_third_diff", {})[key] = float(d)
        rep["runs"][run] = R
    return rep


def charts(rep, media, prefix="v2b"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(media, exist_ok=True)
    out = []

    def smooth(x, w=5):
        if len(x) < w:
            return x
        k = np.ones(w) / w
        return np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (run, R) in zip(axs, rep["runs"].items()):
        for arm, A in R["arms"].items():
            m, lo, hi = arm_curve(A["dirs"], "reward_mean")
            x = np.arange(1, len(m) + 1)
            ax.plot(x, smooth(m), color=COLORS[arm], label=f"{A['title']} (n={len(A['dirs'])})")
            ax.fill_between(x, smooth(lo), smooth(hi), color=COLORS[arm], alpha=0.15)
        ax.set_title(TITLES[run] + " (train reward)")
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axs[0].set_ylabel("mean reward (5-step smooth; band = seed min-max)")
    fig.tight_layout()
    p = os.path.join(media, f"{prefix}_reward_curves.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    xt, xl, xi = [], [], 0
    if "base_pass1" in rep:
        ax.bar(xi, rep["base_pass1"] * 100, color="#c9c0ae")
        xt.append(xi)
        xl.append("base")
        xi += 1.5
    shown = set()
    for run, R in rep["runs"].items():
        for arm, A in R["arms"].items():
            if arm in shown:
                continue
            shown.add(arm)
            for lab, e in A["evals"].items():
                lo, hi = e["ci95"]
                ax.bar(xi, e["pass1"] * 100, color=COLORS[arm], alpha=0.55 if lab != max(A["evals"], key=lambda s: int(s[4:])) else 0.95)
                ax.errorbar(xi, e["pass1"] * 100, yerr=[[100 * (e["pass1"] - lo)], [100 * (hi - e["pass1"])]], color="k", capsize=3)
                ax.scatter([xi] * len(e["per_seed"]), np.array(e["per_seed"]) * 100, color="k", s=8, zorder=3)
                xt.append(xi)
                xl.append(f"{arm}\n{lab}")
                xi += 1
            xi += 0.5
        xi += 0.5
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=7)
    ax.set_ylabel("pass@1 on eval v2 (%)\n95% hierarchical bootstrap CI")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(media, f"{prefix}_pass1.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    keys = [("kl_behav_cur", "KL(sampler || trainer) on sampled tokens"), ("behav_ratio_max", "max IS ratio (diagnostic)"),
            ("grad_norm", "grad norm (pre-clip)"), ("entropy", "policy entropy (nats/token)")]
    fig, axs = plt.subplots(2, 4, figsize=(15, 6))
    for r, (run, R) in enumerate(rep["runs"].items()):
        for c, (key, title) in enumerate(keys):
            ax = axs[r, c]
            for arm, A in R["arms"].items():
                try:
                    m, lo, hi = arm_curve(A["dirs"], key)
                except ValueError:
                    continue
                x = np.arange(1, len(m) + 1)
                ax.plot(x, smooth(m), color=COLORS[arm], label=A["title"])
                ax.fill_between(x, smooth(lo), smooth(hi), color=COLORS[arm], alpha=0.15)
            ax.set_title(f"{TITLES[run].split(':')[0]}: {title}", fontsize=9)
            ax.grid(alpha=0.3)
            if key in ("kl_behav_cur", "behav_ratio_max"):
                ax.set_yscale("log")
            if c == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(media, f"{prefix}_stability.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    names, sps, idle_r, idle_l, cols = [], [], [], [], []
    arms_once = {arm: A for R in rep["runs"].values() for arm, A in R["arms"].items()}
    for arm, A in arms_once.items():
        if True:
            names.append(A["title"].replace(", ", "\n"))
            sps.append(A.get("s_per_step_seeds", []))
            idle_r.append(A.get("rollout_idle_frac", np.nan) * 100)
            idle_l.append(A.get("learner_idle_frac", np.nan) * 100)
            cols.append(COLORS[arm])
    x = np.arange(len(names))
    axs[0].bar(x, [np.mean(s) for s in sps], color=cols)
    for i, s in enumerate(sps):
        axs[0].scatter([i] * len(s), s, color="k", s=10, zorder=3)
    axs[0].set_title("seconds per step (dots = seeds)")
    axs[1].bar(x - 0.2, idle_r, 0.4, color=cols, label="rollout chip")
    axs[1].bar(x + 0.2, idle_l, 0.4, color=cols, alpha=0.45, label="learner chip")
    axs[1].set_title("idle fraction (%)")
    axs[1].legend(fontsize=8)
    for ax in axs[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    for arm, A in arms_once.items():
        if True:
            h = A["staleness_hist"]
            ks = sorted(int(k) for k in h)
            tot = sum(h.values())
            axs[2].plot(ks, [h[str(k)] / tot for k in ks], "o-", color=COLORS[arm], label=A["title"],
                        linestyle="--" if arm == "nois" else "-")
    axs[2].set_title("realized staleness (share of steps)")
    axs[2].set_xlabel("policy versions behind")
    axs[2].legend(fontsize=7)
    axs[2].grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(media, f"{prefix}_throughput_staleness.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--media", default=None)
    a = ap.parse_args()
    rep = build(a.root)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    if a.media:
        print("\n".join(charts(rep, a.media)))
    brief = {run: {"arms": {arm: {k: A.get(k) for k in ("s_per_step", "rollout_idle_frac", "staleness_mean",
                                                        "kl_behav_cur_mean", "entropy_mean", "grad_norm_mean")}
                            | {"evals": {l: (round(e["pass1"], 4), [round(x, 4) for x in e["ci95"]]) for l, e in A["evals"].items()}}
                            for arm, A in R["arms"].items()},
                   "compare": {l: {k: c[k] for k in ("diff", "ci95", "ci90", "equivalent")} for l, c in R.get("compare", {}).items()},
                   "speedup": R.get("speedup_b_over_a"), "last_third_diff": R.get("last_third_diff")}
             for run, R in rep["runs"].items()}
    print(json.dumps(brief, indent=1, default=float))


if __name__ == "__main__":
    main()
