"""plan90 report: 3 arms (sync, async k=2 + IS, async k=2 without IS) x 2 seeds, 20 steps, 6,144-token cap.

Reuses the v2 pieces (report_v2.find_runs/arm_ci/arm_vs_base, stats.compare, analyze.summarize). Wall-clock
comparisons are only made inside a VM pair: the two runs of a pair trained concurrently on chips 0/1 and 2/3
of the same v6e-4, while different pairs ran on different Spot VMs and regions.

    python -m qrl.report_plan90 --root results/plan90/raw --out results/plan90/report.json --media DIR
"""
import argparse
import json
import os
import re

import numpy as np

from . import stats
from .analyze import summarize
from .report_v2 import arm_ci, arm_vs_base, base_eval, eval_paths, find_runs, train_series

ARMS = [("sync", "sync"), ("async", "async k=2 + IS"), ("nois", "async k=2, no IS")]
CONTRASTS = [("async", "sync"), ("nois", "async"), ("nois", "sync")]
PAIRS = {"p1": ("sync-s0", "async-s0"), "p2": ("sync-s1", "nois-s0"), "p3": ("async-s1", "nois-s1")}
COLORS = {"sync": "#5a645c", "async": "#0f6b4c", "nois": "#b85c28"}
PRETTY = {"sync": "sync", "async": "async-IS", "nois": "no-IS"}
LABEL = "step20"


def run_table(runs):
    out = {}
    for arm, seeds in runs.items():
        for s, d in seeds.items():
            S = summarize(d)
            if "step_end_times" not in S:
                continue
            ends = np.array(S.get("step_end_times", []))
            S["step_s"] = np.diff(np.concatenate([[0.0], ends])).tolist() if len(ends) else []
            S["vm"] = os.path.basename(os.path.dirname(d))
            S["acc_curve"] = S.pop("reward_curve", [])
            S["informative_curve"] = train_series(d, "informative_groups").tolist()
            S["truncated_curve"] = gen_series(d, "truncated_frac")
            ev = os.path.join(d, f"eval.{LABEL}.json")
            if os.path.exists(ev):
                e = json.load(open(ev))[LABEL]
                S["eval"] = {k: v for k, v in e.items() if k != "per_problem"}
            out[f"{arm}-s{s}"] = S
    return out


def gen_series(d, key):
    from .analyze import load_events

    g = sorted((e for e in load_events(d) if e["kind"] == "generate"), key=lambda e: e["batch"])
    return [e.get(key, np.nan) for e in g]


def build(root, margin=0.03):
    runs = find_runs(root)
    base = base_eval(root)
    rep = {"root": root, "label": LABEL, "base_eval": base, "runs": run_table(runs), "arms": {}, "contrasts": {},
           "pairs": {}}
    if base:
        b = json.load(open(base))["base"]
        rep["base_pass1"], rep["base_truncated_frac"] = b["pass@1"], b["truncated_frac"]
    paths = {}
    for arm, title in ARMS:
        dirs = [runs[arm][s] for s in sorted(runs.get(arm, {}))]
        ps = eval_paths(dirs, LABEL)
        A = {"title": title, "seeds": sorted(runs.get(arm, {})), "n_evals": len(ps)}
        names = [f"{arm}-s{s}" for s in A["seeds"] if f"{arm}-s{s}" in rep["runs"]]
        for k in ("s_per_step", "wall_s", "rollout_idle_frac", "learner_idle_frac", "gen_tok_per_s", "gen_tokens",
                  "truncated_frac_gen", "informative_groups_mean", "staleness_mean", "kl_behav_cur_mean",
                  "entropy_mean", "grad_norm_mean", "behav_ratio_max", "gen_s_mean", "train_s_mean"):
            v = [rep["runs"][n][k] for n in names if k in rep["runs"][n]]
            if v:
                A[k], A[k + "_seeds"] = float(np.mean(v)), [float(x) for x in v]
        if ps:
            m, c, per_seed = arm_ci(ps)
            A["pass1"], A["ci95"], A["per_seed"] = m, c, per_seed
            if base:
                A.update({k: v for k, v in arm_vs_base(ps, base).items() if k != "pass1"})
            paths[arm] = ps
        rep["arms"][arm] = A
    for a, b in CONTRASTS:
        if a in paths and b in paths:
            rep["contrasts"][f"{a}-{b}"] = stats.compare({a: paths[a], b: paths[b]}, LABEL, a=a, b=b,
                                                         margin=margin, iters=10000)
    for p, (ra, rb) in PAIRS.items():
        A, B = rep["runs"].get(ra), rep["runs"].get(rb)
        if A and B and A.get("complete") and B.get("complete"):
            rep["pairs"][p] = {"vm": A["vm"], "a": ra, "b": rb, "s_per_step": [A["s_per_step"], B["s_per_step"]],
                               "wall_s": [A["wall_s"], B["wall_s"]], "ratio_a_over_b": A["s_per_step"] / B["s_per_step"]}
    asyncish = [rep["runs"][n]["s_per_step"] for n in rep["runs"] if not n.startswith("sync") and rep["runs"][n]["complete"]]
    syncs = [rep["runs"][n]["s_per_step"] for n in rep["runs"] if n.startswith("sync") and rep["runs"][n]["complete"]]
    if asyncish and syncs:
        r, c = stats.seed_ratio_ci(asyncish, syncs)
        rep["speedup_async_family_over_sync"] = {"ratio": r, "ci95": c, "note": "pooled across VMs; see pairs"}
    return rep


def smooth(x, w=3):
    x = np.asarray(x, float)
    if len(x) < w:
        return x
    return np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")


def vm_tag(vm):
    return re.sub(r"-\d{8}T\d{6}$", "", vm).removeprefix("qrl-bc4a-")


def charts(rep, media, prefix="p90"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(media, exist_ok=True)
    out, R = [], rep["runs"]

    fig, axs = plt.subplots(1, 3, figsize=(15, 3.9))
    for arm, title in ARMS:
        names = [n for n in R if n.rsplit("-s", 1)[0] == arm]
        if not names:
            continue
        curves = [np.array(R[n]["acc_curve"]) for n in names]
        n = min(len(c) for c in curves)
        a = np.stack([c[:n] for c in curves])
        x = np.arange(1, n + 1)
        axs[0].plot(x, smooth(a.mean(0)), color=COLORS[arm], label=f"{title} (n={len(names)})")
        axs[0].fill_between(x, smooth(a.min(0)), smooth(a.max(0)), color=COLORS[arm], alpha=0.15)
        for nm in names:
            r = R[nm]
            t = (np.array(r["step_end_times"]) + 0) / 60
            ls = "-" if nm.endswith("s0") else "--"
            axs[1].plot(t[:len(r["acc_curve"])], smooth(r["acc_curve"]), ls, color=COLORS[arm], label=nm)
            axs[2].plot(np.arange(1, len(t) + 1), t, ls, color=COLORS[arm], label=f"{nm} ({vm_tag(r['vm'])})")
    axs[0].set(title="train reward (= accuracy) per step, 3-step smooth", xlabel="step", ylabel="mean reward")
    axs[1].set(title="train reward vs wall-clock", xlabel="minutes since first generation")
    axs[2].set(title="wall-clock to N steps", xlabel="step", ylabel="minutes")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(media, f"{prefix}_reward_wallclock.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axs[0]
    xs, xl = [], []
    if "base_pass1" in rep:
        ax.bar(0, rep["base_pass1"] * 100, color="#c9c0ae")
        xs.append(0)
        xl.append("base")
    for i, (arm, title) in enumerate(ARMS, 1):
        A = rep["arms"].get(arm, {})
        if "pass1" not in A:
            continue
        lo, hi = A["ci95"]
        ax.bar(i, A["pass1"] * 100, color=COLORS[arm])
        ax.errorbar(i, A["pass1"] * 100, yerr=[[100 * (A["pass1"] - lo)], [100 * (hi - A["pass1"])]], color="k", capsize=4)
        ax.scatter([i] * len(A["per_seed"]), np.array(A["per_seed"]) * 100, color="k", s=12, zorder=3)
        xs.append(i)
        xl.append(title.replace(" + ", "\n+ ").replace(", ", "\n"))
    ax.set_xticks(xs)
    ax.set_xticklabels(xl, fontsize=8)
    ax.set_ylabel("step-20 pass@1 (%), 136 problems x 16 samples\n95% bootstrap CI (problems x seeds x samples)")
    ax.grid(axis="y", alpha=0.3)
    ax = axs[1]
    names = list(rep["contrasts"])
    for j, k in enumerate(names):
        c = rep["contrasts"][k]
        lo, hi = c["ci95"]
        ax.errorbar(c["diff"] * 100, j, xerr=[[100 * (c["diff"] - lo)], [100 * (hi - c["diff"])]], fmt="o", color="k", capsize=4)
    ax.axvline(0, color="#999", lw=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([" − ".join(PRETTY[x] for x in k.split("-")) for k in names])
    ax.invert_yaxis()
    ax.set(title="paired differences at step 20 (pp, 95% CI)", xlabel="pass@1 difference (percentage points)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(media, f"{prefix}_pass1.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.9))
    order = [n for p_ in PAIRS.values() for n in p_ if n in R]
    x = np.arange(len(order))
    cols = [COLORS[n.rsplit("-s", 1)[0]] for n in order]
    axs[0].bar(x, [R[n]["s_per_step"] for n in order], color=cols)
    for i, n in enumerate(order):
        axs[0].text(i, R[n]["s_per_step"] + 3, f"{R[n]['s_per_step']:.0f}", ha="center", fontsize=8)
    axs[0].set_title("steady-state seconds per step (pairs share a VM)")
    axs[1].bar(x - 0.2, [100 * R[n]["rollout_idle_frac"] for n in order], 0.4, color=cols, label="rollout chip idle")
    axs[1].bar(x + 0.2, [100 * R[n]["learner_idle_frac"] for n in order], 0.4, color=cols, alpha=0.45, label="learner chip idle")
    axs[1].set_title("idle fraction (%)")
    axs[1].legend(fontsize=8)
    pair_of = {n: p_ for p_, ns in PAIRS.items() for n in ns}
    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n{pair_of[n]} {vm_tag(R[n]['vm'])}" for n in order], fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        for b in range(1, len(order) // 2):
            ax.axvline(2 * b - 0.5, color="#bbb", lw=1)
    fig.tight_layout()
    p = os.path.join(media, f"{prefix}_speed_idle.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    keys = [("kl_behav_cur", "KL(sampler || trainer), sampled tokens"), ("entropy", "policy entropy (nats/token)"),
            ("grad_norm", "grad norm (pre-clip)"), ("staleness", "policy versions behind")]
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.4))
    for ax, (key, title) in zip(axs, keys):
        for arm, t in ARMS:
            for n in [n for n in R if n.rsplit("-s", 1)[0] == arm]:
                y = train_series(R[n]["run_dir"], key)
                ax.plot(np.arange(1, len(y) + 1), y, "-" if n.endswith("s0") else "--", color=COLORS[arm], label=n)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        if key == "kl_behav_cur":
            ax.set_yscale("log")
    axs[0].legend(fontsize=6)
    fig.tight_layout()
    p = os.path.join(media, f"{prefix}_stability.png")
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
    slim = json.loads(json.dumps(rep, default=float))
    json.dump(slim, open(a.out, "w"), indent=1)
    if a.media:
        print("\n".join(charts(rep, a.media)))
    for arm, A in rep["arms"].items():
        print(arm, {k: (round(A[k], 4) if isinstance(A.get(k), float) else A.get(k)) for k in
                    ("pass1", "ci95", "per_seed", "gain", "gain_ci95", "s_per_step", "rollout_idle_frac",
                     "learner_idle_frac", "staleness_mean", "kl_behav_cur_mean", "truncated_frac_gen")})
    for k, c in rep["contrasts"].items():
        print(k, round(c["diff"] * 100, 2), [round(x * 100, 2) for x in c["ci95"]], "ci90",
              [round(x * 100, 2) for x in c["ci90"]], "equiv@3pp", c["equivalent"])
    print(json.dumps(rep["pairs"], indent=1), rep.get("speedup_async_family_over_sync"))


if __name__ == "__main__":
    main()
