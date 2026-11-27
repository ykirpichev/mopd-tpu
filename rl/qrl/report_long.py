"""plan65k1long report: async k=1 with and without truncated IS, 2 seeds each, 60 steps, evals at steps 20/40/60.

The key question is whether the no-IS drift seen at 20 steps (plan90 k=2, plan25k1 k=1) turns into instability
or keeps learning. Per run and step: KL(sampler || trainer) on sampled tokens, max IS ratio (applied with IS,
diagnostic only without), pre-clip grad norm vs the 1.0 clip, entropy, train reward, lag and s/step. Pass@1 CIs and
the IS vs no-IS contrasts use report_plan90's hierarchical bootstrap at each eval step. The first 20 steps repeat
plan25k1's k=1 runs (same seeds, flags and constant lr), so the step-20 numbers are also a replication.

plan55sync60 adds the sync arm (lsync-s0/s1: plan90's sync config, 60 steps, same evals) from a second results root.
With it, the contrasts include k=1 + IS and k=1 no IS against sync at each eval step, and the replication compares
lsync at step 20 with plan90's 20-step sync runs.

    python -m qrl.report_long --root results/plan65k1long/raw results/plan55sync60/raw \
        --k1-root results/plan25k1/raw --p90-root results/plan90/raw \
        --out results/plan55sync60/report.json --media DIR
"""
import argparse
import json
import os
import tempfile

import numpy as np

from . import report_plan90 as R90
from . import stats
from .report_v2 import train_series

ARMS = [("lsync", "sync"), ("lk1is", "k=1 + IS"), ("lk1nois", "k=1, no IS")]
CONTRASTS = [("lk1nois", "lk1is"), ("lk1is", "lsync"), ("lk1nois", "lsync")]
PAIRS = {"pA": ("lk1is-s0", "lk1nois-s0"), "pB": ("lk1is-s1", "lk1nois-s1"), "pS": ("lsync-s0", "lsync-s1")}
COLORS = {"lsync": "#5a645c", "lk1is": "#2b6cb0", "lk1nois": "#c53030"}
PRETTY = {"lsync": "sync", "lk1is": "IS", "lk1nois": "no-IS"}
LABELS = ["step20", "step40", "step60"]
CLIP = 1.0
SERIES = ["kl_behav_cur", "behav_ratio_max", "grad_norm", "entropy", "reward_mean", "staleness", "train_tokens",
          "informative_groups"]


def patch():
    R90.ARMS, R90.CONTRASTS, R90.PAIRS, R90.COLORS, R90.PRETTY = ARMS, CONTRASTS, PAIRS, COLORS, PRETTY


def windows(x, edges=((0, 10), (10, 20), (20, 40), (40, 60))):
    return {f"{a + 1}-{b}": float(np.nanmean(x[a:b])) if len(x) > a else None for a, b in edges}


def run_diag(d):
    s = {k: train_series(d, k) for k in SERIES}
    kl, gn, rmax = s["kl_behav_cur"], s["grad_norm"], s["behav_ratio_max"]
    out = {"steps": int(len(kl)), "kl": windows(kl), "kl_max": float(np.nanmax(kl)), "kl_max_step": int(np.nanargmax(kl)) + 1,
           "kl_last": float(kl[-1]), "ratio_max": float(np.nanmax(rmax)), "ratio_max_step": int(np.nanargmax(rmax)) + 1,
           "ratio_max_by_window": {k: float(np.nanmax(rmax[a:b])) for k, (a, b) in
                                   {"1-20": (0, 20), "21-40": (20, 40), "41-60": (40, 60)}.items() if len(rmax) > a},
           "clip_hits": int(np.sum(gn > CLIP)), "clip_hit_steps": [int(i) + 1 for i in np.where(gn > CLIP)[0]],
           "grad_norm": windows(gn), "grad_norm_max": float(np.nanmax(gn)), "entropy": windows(s["entropy"]),
           "reward": windows(s["reward_mean"]), "staleness_hist": np.bincount(s["staleness"].astype(int)).tolist(),
           "train_tokens": windows(s["train_tokens"]), "informative": windows(s["informative_groups"])}
    return out, s


def combined_root(roots):
    """One directory whose children are the VM directories of every root (find_runs globs root/*/*/config.json)."""
    if isinstance(roots, str):
        return roots
    d = tempfile.mkdtemp(prefix="longroot-")
    for r in roots:
        for vm in sorted(os.listdir(r)):
            src = os.path.abspath(os.path.join(r, vm))
            if os.path.isdir(src) and not os.path.exists(os.path.join(d, vm)):
                os.symlink(src, os.path.join(d, vm))
    return d


def build(roots, k1_root=None, p90_root=None):
    patch()
    root = combined_root(roots)
    rep = {"root": roots, "evals": {}, "diag": {}}
    for lab in LABELS:
        R90.LABEL = lab
        r = R90.build(root)
        rep["evals"][lab] = {"arms": {a: {k: r["arms"][a].get(k) for k in ("pass1", "ci95", "per_seed", "gain", "gain_ci95")}
                                      for a in r["arms"]},
                             "contrasts": r["contrasts"],
                             "runs": {n: json.loads(json.dumps(v.get("eval", {}), default=float)) for n, v in r["runs"].items()}}
        if lab == LABELS[-1]:
            rep["runs"] = r["runs"]
            rep["pairs"] = r["pairs"]
            rep["base_pass1"] = r.get("base_pass1")
            rep["speed"] = {a: {k: r["arms"][a].get(k) for k in ("s_per_step", "s_per_step_seeds", "wall_s_seeds",
                                                                  "rollout_idle_frac", "learner_idle_frac",
                                                                  "gen_s_mean", "train_s_mean", "truncated_frac_gen_seeds")}
                            for a in r["arms"]}
    rep["series"] = {}
    for n, r in rep["runs"].items():
        rep["diag"][n], s = run_diag(r["run_dir"])
        rep["series"][n] = {k: v.tolist() for k, v in s.items()}
        st = np.array(r.get("step_s", []))
        rep["diag"][n]["s_per_step_windows"] = windows(st)
    rep["replication_step20"] = {}
    if k1_root:
        rep["replication_step20"].update(replication(root, k1_root, (("lk1is", "k1is"), ("lk1nois", "k1nois"))))
    if p90_root:
        rep["replication_step20"].update(replication(root, p90_root, (("lsync", "sync"),)))
    return rep


def replication(root, k1_root, pairs):
    """Step-20 pass@1 of the 60-step runs vs the 20-step runs of the same arm (same seeds and flags up to step 20)."""
    from .report_v2 import eval_paths, find_runs

    long_runs, k1_runs = find_runs(root), find_runs(k1_root)
    out = {}
    for la, ka in pairs:
        pl = eval_paths([long_runs[la][s] for s in sorted(long_runs.get(la, {}))], "step20")
        pk = eval_paths([k1_runs[ka][s] for s in sorted(k1_runs.get(ka, {}))], "step20")
        if pl and pk:
            out[f"{la}-vs-{ka}"] = stats.compare({la: pl, ka: pk}, "step20", a=la, b=ka, margin=0.03, iters=10000)
        for s in sorted(long_runs.get(la, {})):
            if s in k1_runs.get(ka, {}):
                a = train_series(long_runs[la][s], "kl_behav_cur")[:20]
                b = train_series(k1_runs[ka][s], "kl_behav_cur")[:20]
                out[f"kl20-{la}-s{s}"] = {"long": a.tolist(), "k1_20step": b.tolist()}
    return out


def charts(rep, media, prefix="long"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(media, exist_ok=True)
    S, paths = rep["series"], []

    def each():
        for arm, title in ARMS:
            for n in sorted(x for x in S if x.rsplit("-s", 1)[0] == arm):
                yield n, COLORS[arm], "-" if n.endswith("s0") else "--"

    fig, axs = plt.subplots(1, 3, figsize=(16, 3.9))
    for n, c, ls in each():
        x = np.arange(1, len(S[n]["kl_behav_cur"]) + 1)
        axs[0].plot(x, S[n]["kl_behav_cur"], ls, color=c, label=n)
        axs[1].plot(x, S[n]["behav_ratio_max"], ls, color=c, label=n)
        axs[2].plot(x, S[n]["grad_norm"], ls, color=c, label=n)
    axs[0].set(title="KL(sampler || trainer), sampled tokens", xlabel="step", yscale="log")
    axs[1].set(title="max IS ratio per batch (no-IS: diagnostic only)", xlabel="step", yscale="log")
    axs[2].axhline(CLIP, color="#999", lw=1, ls=":")
    axs[2].set(title="grad norm, pre-clip (dotted: clip at 1.0)", xlabel="step")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    paths.append(os.path.join(media, f"{prefix}_kl_ratio_grad.png"))
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(18, 3.8))
    for n, c, ls in each():
        x = np.arange(1, len(S[n]["entropy"]) + 1)
        axs[0].plot(x, R90.smooth(np.array(S[n]["reward_mean"]), 5), ls, color=c, label=n)
        axs[1].plot(x, S[n]["entropy"], ls, color=c, label=n)
        axs[2].plot(x, np.array(S[n]["train_tokens"]) / 1e3, ls, color=c, label=n)
        axs[3].plot(x, np.array(S[n]["staleness"]) + (0.03 if "nois" in n else 0.0 if "sync" in n else -0.03), ls, color=c, label=n)
    axs[0].set(title="train reward (batch accuracy), 5-step smooth", xlabel="step")
    axs[1].set(title="policy entropy (nats/token)", xlabel="step")
    axs[2].set(title="trained tokens per step (k)", xlabel="step")
    axs[3].set(title="policy versions behind", xlabel="step", yticks=[0, 1, 2], ylim=(-0.3, 2.3))
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    paths.append(os.path.join(media, f"{prefix}_reward_entropy_lag.png"))
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4))
    steps = [0] + [int(lab[4:]) for lab in LABELS]
    arms = [(a, t) for a, t in ARMS if any(rep["evals"][lab]["arms"].get(a, {}).get("pass1") is not None for lab in LABELS)]
    for i, (arm, title) in enumerate(arms):
        off = (i - (len(arms) - 1) / 2) * 1.2
        ys, lo, hi, seeds = [rep.get("base_pass1")], [0], [0], []
        for lab in LABELS:
            A = rep["evals"][lab]["arms"].get(arm, {})
            ys.append(A.get("pass1"))
            lo.append(A["pass1"] - A["ci95"][0] if A.get("pass1") is not None else 0)
            hi.append(A["ci95"][1] - A["pass1"] if A.get("pass1") is not None else 0)
            seeds.append(A.get("per_seed") or [])
        y = np.array([v if v is not None else np.nan for v in ys]) * 100
        axs[0].errorbar(np.array(steps) + off, y, yerr=[np.array(lo) * 100, np.array(hi) * 100], color=COLORS[arm],
                        marker="o", capsize=4, label=title)
        for st, sd in zip(steps[1:], seeds):
            axs[0].scatter([st + off] * len(sd), np.array(sd) * 100, color=COLORS[arm], s=14, marker="x")
    axs[0].set(title="pass@1 vs step (136 problems x 16 samples; 95% CI; x = seeds)", xlabel="step",
               ylabel="pass@1 (%)", xticks=steps)
    axs[0].grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    rows = []
    for a, b in CONTRASTS:
        for lab in LABELS:
            c = rep["evals"][lab]["contrasts"].get(f"{a}-{b}")
            if c:
                rows.append((f"{PRETTY[a]} − {PRETTY[b]}, {lab}", c, COLORS[a]))
    for j, (name, c, col) in enumerate(rows):
        axs[1].errorbar(c["diff"] * 100, j, xerr=[[100 * (c["diff"] - c["ci95"][0])], [100 * (c["ci95"][1] - c["diff"])]],
                        fmt="o", color=col, capsize=4)
    axs[1].axvline(0, color="#999", lw=1)
    axs[1].set_yticks(range(len(rows)))
    axs[1].set_yticklabels([r[0] for r in rows], fontsize=8)
    axs[1].invert_yaxis()
    axs[1].set(title="paired differences (pp, 95% CI)", xlabel="pass@1 difference (percentage points)")
    axs[1].grid(axis="x", alpha=0.3)
    fig.tight_layout()
    paths.append(os.path.join(media, f"{prefix}_pass1.png"))
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
    for n, c, ls in each():
        r = rep["runs"][n]
        st = np.array(r.get("step_s", []))
        axs[0].plot(np.arange(1, len(st) + 1), st, ls, color=c, label=f"{n} ({R90.vm_tag(r['vm'])})")
        t = np.array(r["step_end_times"]) / 60
        axs[1].plot(np.arange(1, len(t) + 1), t, ls, color=c, label=n)
    axs[0].set(title="seconds per step", xlabel="step", ylabel="s")
    axs[1].set(title="wall-clock to N steps", xlabel="step", ylabel="minutes since first generation")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    paths.append(os.path.join(media, f"{prefix}_speed.png"))
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="+", required=True)
    ap.add_argument("--k1-root", default=None)
    ap.add_argument("--p90-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--media", default=None)
    a = ap.parse_args()
    rep = build(a.root if len(a.root) > 1 else a.root[0], a.k1_root, a.p90_root)
    slim = json.loads(json.dumps(rep, default=float))
    json.dump(slim, open(a.out, "w"), indent=1)
    if a.media:
        print("\n".join(charts(rep, a.media)))
    for lab in LABELS:
        E = rep["evals"][lab]
        for arm, A in E["arms"].items():
            if A.get("pass1") is not None:
                print(lab, arm, round(A["pass1"] * 100, 2), [round(x * 100, 2) for x in A["ci95"]],
                      [round(x * 100, 2) for x in A["per_seed"]])
        for k, c in E["contrasts"].items():
            print(lab, k, round(c["diff"] * 100, 2), [round(x * 100, 2) for x in c["ci95"]],
                  [round(x * 100, 2) for x in c["ci90"]], "equiv@3pp", c["equivalent"])
    for n, d in sorted(rep["diag"].items()):
        print(n, json.dumps(d, default=float))
    for k, v in rep.get("replication_step20", {}).items():
        if "diff" in v:
            print("replication", k, round(v["diff"] * 100, 2), [round(x * 100, 2) for x in v["ci95"]])


if __name__ == "__main__":
    main()
