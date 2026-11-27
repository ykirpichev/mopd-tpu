"""Paper report (docs/mopd-paper-plan.md section 6): every arm with all its seeds, the SFT and merge baselines and the
re-evaluated reference rows, pooled from several campaign roots (mopdfull: seeds 0/1, teachers, base; mopdpaper:
seeds 2/3, SFT, merges, reference re-evals).

Statistics:
  - hierarchical bootstrap (problems x rows x samples), one shared problem draw per benchmark, so every difference
    and headroom ratio is paired by problem; rows are seeds for trained arms and eval repeats for reference rows;
  - TOST equivalence at +-MARGIN with 90% CIs; two-sided bootstrap p-values with Holm correction per benchmark over
    the primary family;
  - a per-benchmark binomial GLMM, correct ~ arm + (1|problem) + (1|run), for the primary contrasts (statsmodels
    BinomialBayesMixedGLM, variational Bayes; reported as a robustness check);
  - VM-hours per seed from the learner events (whole VM for the distillation arms, half a VM for Mix-RL and SFT).

    python -m qrl.report_paper --roots results/mopdfull/raw results/mopdpaper/raw --out paper.json --media DIR
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from .analyze import load_events

KINDS = (("v2", "AIME pass@1"), ("ifeval", "IFEval strict"), ("ifbench", "IFBench strict"))
TEACHER_OF = {"v2": "tmath", "ifeval": "tif", "ifbench": "tif"}
MARGIN = 0.03
STEP = "step60"
SFT_LONG = "step120"
REF = ("base", "tmath", "tif")
SEEDED = ("mixrl", "mopd", "mopd-dist", "opd-tmath", "sft")
MERGES = ("merge-m1-linear", "merge-m2-ta", "merge-m3-ties20", "merge-m4-ties50")
VM_SHARE = {"mixrl": 0.5, "sft": 0.5, "mopd": 1.0, "mopd-dist": 1.0, "opd-tmath": 1.0}
STYLE = {
    "base": ("base 2B", "#a0aec0"), "tmath": ("T-math", "#2f855a"), "tif": ("T-IF", "#b83280"),
    "mixrl": ("Mix-RL", "#5a645c"), "mopd": ("MOPD", "#2b6cb0"), "mopd-dist": ("MOPD-dist", "#c05621"),
    "opd-tmath": ("OPD (T-math only)", "#6b46c1"), "sft": ("SFT (offline KD)", "#d69e2e"),
    "sft@120": ("SFT step 120", "#975a16"), "merge-m1-linear": ("M1 linear", "#81e6d9"),
    "merge-m2-ta": ("M2 task arith.", "#38b2ac"), "merge-m3-ties20": ("M3 TIES 20%", "#2c7a7b"),
    "merge-m4-ties50": ("M4 TIES 50%", "#234e52"), "mopd#step20": ("MOPD step 20", "#2b6cb0"),
}
# (a, b): primary family, Holm-corrected per benchmark; H1 rows compare against the domain's teacher
PRIMARY = (("mopd", "mixrl"), ("mopd", "sft"), ("mopd", "sft@120"), ("mopd", "merge-m3-ties20"),
           ("mopd-dist", "mopd"), ("mopd", "opd-tmath"), ("mopd", "teacher"))
SECONDARY = (("mopd", "merge-m1-linear"), ("mopd", "merge-m2-ta"), ("mopd", "merge-m4-ties50"),
             ("sft", "mixrl"), ("merge-m3-ties20", "teacher"), ("merge-m2-ta", "teacher"), ("sft", "teacher"),
             ("sft@120", "sft"), ("merge-m3-ties20", "base"), ("mopd-dist", "mixrl"), ("mopd#step20", "mixrl"))
EVAL_RE = re.compile(r"^eval(?:_(ifeval|ifbench))?\.([^.]+)\.json$")


def arm_of(run):
    tail = run.rsplit("-s", 1)
    return tail[0] if len(tail) == 2 and tail[1].isdigit() else run


def collect(roots):
    """{run: {(kind, label): [(boot, result), ...]}} over every boot dir of every root, plus the pilot base AIME."""
    out = {}
    for root in roots:
        f = os.path.join(root, os.pardir, "pilot", "eval_base.json")
        if os.path.exists(f):
            out.setdefault("base", {}).setdefault(("v2", "final"), []).append(("pilot", json.load(open(f))["base"]))
        for d in sorted(glob.glob(os.path.join(root, "*", "*"))):
            run = os.path.basename(d)
            if not os.path.isdir(d) or run in ("teachers", "smoke", "sft-smoke", "roundtrip", "probe"):
                continue
            boot = os.path.basename(os.path.dirname(d))
            if run == "pilot":
                f = os.path.join(d, "eval_base.json")
                if os.path.exists(f):
                    out.setdefault("base", {}).setdefault(("v2", "final"), []).append((boot, json.load(open(f))["base"]))
                continue
            for f in os.listdir(d):
                m = EVAL_RE.match(f)
                if not m:
                    continue
                r = json.load(open(os.path.join(d, f))).get(m.group(2))
                if r and r.get("per_problem"):
                    out.setdefault(run, {}).setdefault((m.group(1) or "v2", m.group(2)), []).append((boot, r))
    # one base AIME eval per pilot (they are the same eval copied to every boot): keep the first
    if ("v2", "final") in out.get("base", {}):
        seen, keep = set(), []
        for b, r in out["base"][("v2", "final")]:
            k = (r["pass@1"], r["samples"])
            if k not in seen:
                seen.add(k); keep.append((b, r))
        out["base"][("v2", "final")] = keep
    return out


def problems(r):
    return {q["id"]: (float(q["correct"]), r["samples"]) for q in r["per_problem"]}


def row_evals(C, row, kind):
    """Eval results forming one table row: one per seed (latest boot) or every repeat of a reference model."""
    if row in REF:
        ev = [r for _, r in C.get(row, {}).get((kind, "final"), [])]
        ev += [r for k, lst in C.get(f"refs-{row}", {}).items() if k[0] == kind for _, r in lst]
        return ev
    if row.startswith("merge-"):
        return [lst[-1][1] for k, lst in C.get(row, {}).items() if k == (kind, "final")]
    arm, label = (row.split("@")[0], f"step{row.split('@')[1]}") if "@" in row else (row, STEP)
    runs = sorted(r for r in C if arm_of(r) == arm and r != arm)
    return [C[r][(kind, label)][-1][1] for r in runs if (kind, label) in C[r]]


def seeds_of(C, row, kind):
    arm, label = (row.split("@")[0], f"step{row.split('@')[1]}") if "@" in row else (row, STEP)
    return sorted(r for r in C if arm_of(r) == arm and r != arm and (kind, label) in C[r])


def by_step(C, arm, kind):
    labels = sorted({k[1] for r in C if arm_of(r) == arm and r != arm for k in C[r] if k[0] == kind and
                     k[1].startswith("step")}, key=lambda s: int(s[4:]))
    return labels


def matrix(evals, ids):
    pp = [problems(r) for r in evals]
    return np.array([[p[i][0] for i in ids] for p in pp]), np.array([pp[k][ids[0]][1] for k in range(len(pp))])


def hier(c, n, prob, rng):
    rows = rng.integers(0, c.shape[0], c.shape[0])
    p = c[rows][:, prob] / n[rows][:, None]
    return (rng.binomial(n[rows][:, None], p) / n[rows][:, None]).mean()


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj


def glmm(C, a, b, kind, ids):
    """correct ~ arm + (1|problem) + (1|run) on Bernoulli rows; returns the arm log-odds coefficient (a vs b)."""
    try:
        import pandas as pd
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError:
        return None
    recs = []
    for arm, is_a in ((a, 1), (b, 0)):
        for k, r in enumerate(row_evals(C, arm, kind)):
            p = problems(r)
            for i in ids:
                c, n = p[i]
                c = int(round(c))
                recs += [(is_a, str(i), f"{arm}#{k}", 1)] * c + [(is_a, str(i), f"{arm}#{k}", 0)] * (int(n) - c)
    if not recs:
        return None
    df = pd.DataFrame(recs, columns=["arm", "problem", "run", "y"])
    m = BinomialBayesMixedGLM.from_formula("y ~ arm", {"problem": "0 + C(problem)", "run": "0 + C(run)"}, df)
    f = m.fit_vb()
    j = list(m.exog_names).index("arm")
    return {"logodds": float(f.fe_mean[j]), "sd": float(f.fe_sd[j]),
            "ci95": [float(f.fe_mean[j] - 1.96 * f.fe_sd[j]), float(f.fe_mean[j] + 1.96 * f.fe_sd[j])],
            "rows": len(df)}


def vm_hours(C_dirs, run, upto):
    """Training VM-hours of RUN to step UPTO: learner wall time from its first train start, times the VM share."""
    for d in reversed(C_dirs.get(run, [])):
        try:
            tr = sorted((e for e in load_events(d) if e["kind"] == "train"), key=lambda e: e["step"])
        except (FileNotFoundError, KeyError, ValueError):
            continue
        tr = [e for e in tr if e["step"] < upto]
        if len(tr) >= upto * 0.9:
            ev = load_events(d)
            t0 = min([e["start"] for e in ev if e["kind"] in ("startup", "generate", "train")] or [tr[0]["start"]])
            return (tr[-1]["end"] - t0) / 3600 * VM_SHARE.get(arm_of(run), 1.0)
    return None


def build(roots, n_boot=10000, seed=0, with_glmm=True):
    C = collect(roots)
    dirs = {}
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "*", "*"))):
            if os.path.isdir(d):
                dirs.setdefault(os.path.basename(d), []).append(d)
    rows = list(REF) + list(SEEDED) + ["sft@120"] + list(MERGES)
    out = {"roots": roots, "bootstrap": {"draws": n_boot, "levels": "problems, rows (seeds or eval repeats), samples"},
           "margin": MARGIN, "label": STEP, "evals": {}, "contrasts": {}, "headroom": {}, "by_step": {},
           "vm_hours": {}, "glmm": {}, "seeds": {}}
    for kind, _ in KINDS:
        ev = {r: row_evals(C, r, kind) for r in rows}
        ev = {r: e for r, e in ev.items() if e}
        common = sorted(set.intersection(*(set(problems(x)) for e in ev.values() for x in e)))
        mats = {r: matrix(e, common) for r, e in ev.items()}
        steps = {a: by_step(C, a, kind) for a in SEEDED}
        for a, labs in steps.items():
            for lab in labs:
                runs = sorted(r for r in C if arm_of(r) == a and r != a and (kind, lab) in C[r])
                if runs:
                    mats[f"{a}#{lab}"] = matrix([C[r][(kind, lab)][-1][1] for r in runs], common)
        rng = np.random.default_rng(seed)
        P = len(common)
        draws = {k: np.empty(n_boot) for k in mats}
        for t in range(n_boot):
            prob = rng.integers(0, P, P)
            for k, (c, n) in mats.items():
                draws[k][t] = hier(c, n, prob, rng)
        point = {k: float((c / n[:, None]).mean()) for k, (c, n) in mats.items()}
        ci = lambda x, q=(.025, .975): [float(np.quantile(x, q[0])), float(np.quantile(x, q[1]))]
        E = out["evals"][kind] = {}
        for r in ev:
            c, n = mats[r]
            E[r] = {"mean": point[r], "ci95": ci(draws[r]), "rows": int(c.shape[0]), "problems": P,
                    "samples": [int(x) for x in n], "per_row": [float(x) for x in (c / n[:, None]).mean(1)]}
            if r in SEEDED or "@" in r:
                out["seeds"].setdefault(r, {})[kind] = seeds_of(C, r, kind)
        out["by_step"][kind] = {a: {lab: {"mean": point[f"{a}#{lab}"], "ci95": ci(draws[f"{a}#{lab}"]),
                                          "rows": int(mats[f"{a}#{lab}"][0].shape[0])} for lab in labs}
                                for a, labs in steps.items() if labs}
        res = {}
        for fam, pairs in (("primary", PRIMARY), ("secondary", SECONDARY)):
            for a, b in pairs:
                b_ = TEACHER_OF[kind] if b == "teacher" else b
                if a not in draws or b_ not in draws:
                    continue
                d = draws[a] - draws[b_]
                p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
                c90 = ci(d, (.05, .95))
                res[f"{a} - {b_}"] = {"family": fam, "diff": point[a] - point[b_], "ci95": ci(d), "ci90": c90,
                                      "p": p, "equivalent": c90[0] > -MARGIN and c90[1] < MARGIN,
                                      "significant": ci(d)[0] > 0 or ci(d)[1] < 0}
        prim = [k for k, v in res.items() if v["family"] == "primary"]
        for k, ph in zip(prim, holm(np.array([res[k]["p"] for k in prim]))):
            res[k]["p_holm"] = float(ph)
            res[k]["significant_holm"] = bool(ph < 0.05)
        out["contrasts"][kind] = res
        t = TEACHER_OF[kind]
        if t in draws and "base" in draws:
            gap = draws[t] - draws["base"]
            ok = np.abs(gap) > 1e-9
            for a in [r for r in ev if r not in REF]:
                ratio = (draws[a][ok] - draws["base"][ok]) / gap[ok]
                out["headroom"].setdefault(kind, {})[a] = {
                    "closed": (point[a] - point["base"]) / (point[t] - point["base"]), "ci95": ci(ratio),
                    "teacher_gap": point[t] - point["base"], "gap_ci95": ci(gap)}
        if with_glmm:
            for a, b in PRIMARY:
                b_ = TEACHER_OF[kind] if b == "teacher" else b
                if a in ev and b_ in ev:
                    try:
                        g = glmm(C, a, b_, kind, common)
                    except Exception as e:  # VB can fail to converge on degenerate rows; report, don't abort
                        g = {"error": f"{type(e).__name__}: {e}"}
                    if g:
                        out["glmm"].setdefault(kind, {})[f"{a} - {b_}"] = g
    for r in sorted(C):
        a = arm_of(r)
        if a in SEEDED and r != a:
            out["vm_hours"][r] = {"step60": vm_hours(dirs, r, 60)}
            if a == "sft":
                out["vm_hours"][r]["step120"] = vm_hours(dirs, r, 120)
    return out


def charts(rep, media, prefix="paper"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(media, exist_ok=True)
    name = lambda r: STYLE.get(r, (r,))[0]
    color = lambda r: STYLE.get(r, ("", "#718096"))[1]
    files = []

    order = list(REF) + list(SEEDED) + ["sft@120"] + list(MERGES)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for i, (kind, title) in enumerate(KINDS):
        E = rep["evals"][kind]
        rows = [r for r in order if r in E]
        for k, r in enumerate(rows):
            m = E[r]
            ax[i].bar(k, m["mean"], color=color(r), alpha=0.55 if r in REF else 1.0)
            ax[i].errorbar(k, m["mean"], yerr=[[m["mean"] - m["ci95"][0]], [m["ci95"][1] - m["mean"]]], color="k",
                           capsize=3, lw=1)
            ax[i].text(k, m["ci95"][1] + 0.003, f"{m['mean']:.3f}\nn={m['rows']}", ha="center", va="bottom", fontsize=6.5)
        ax[i].set_xticks(range(len(rows)), [name(r) for r in rows], rotation=40, ha="right", fontsize=8)
        ax[i].set_ylim(max(0, min(E[r]["ci95"][0] for r in rows) - 0.05), None)
        ax[i].set_title(f"{title}, step 60 (n = seeds or eval repeats; 95% CI)", fontsize=10)
        ax[i].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    f = os.path.join(media, f"{prefix}_t1_evals.png"); fig.savefig(f, dpi=140); plt.close(fig); files.append(f)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for i, (kind, title) in enumerate(KINDS):
        S = rep["by_step"][kind]
        b = rep["evals"][kind]["base"]["mean"]
        for a, labs in S.items():
            xs = [0] + [int(l[4:]) for l in labs]
            ys = [b] + [labs[l]["mean"] for l in labs]
            lo = [b] + [labs[l]["ci95"][0] for l in labs]
            hi = [b] + [labs[l]["ci95"][1] for l in labs]
            ax[i].plot(xs, ys, "o-", color=color(a), label=name(a), lw=1.6)
            ax[i].fill_between(xs, lo, hi, color=color(a), alpha=0.1)
        t = TEACHER_OF[kind]
        ax[i].axhline(rep["evals"][kind][t]["mean"], color=color(t), ls="--", lw=1.2, label=f"{name(t)} teacher")
        m3 = rep["evals"][kind].get("merge-m3-ties20")
        if m3:
            ax[i].axhline(m3["mean"], color=color("merge-m3-ties20"), ls=":", lw=1.4, label="M3 TIES merge")
        ax[i].set_title(title + " by training step (seeds pooled, 95% CI)", fontsize=10)
        ax[i].set_xlabel("training step"); ax[i].grid(alpha=0.3)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    f = os.path.join(media, f"{prefix}_f2_by_step.png"); fig.savefig(f, dpi=140); plt.close(fig); files.append(f)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for i, (kind, title) in enumerate(KINDS):
        E = rep["evals"][kind]
        for r in [x for x in order if x in E and x not in REF]:
            if r.startswith("merge-"):
                vh = 0.04 * (MERGES.index(r) - 1.5)
            else:
                arm, st = (r.split("@")[0], "step120") if "@" in r else (r, "step60")
                v = [rep["vm_hours"][s][st] for s in rep["vm_hours"] if arm_of(s) == arm and rep["vm_hours"][s].get(st)]
                if not v:
                    continue
                vh = float(np.mean(v))
            m = E[r]
            ax[i].errorbar(vh, m["mean"], yerr=[[m["mean"] - m["ci95"][0]], [m["ci95"][1] - m["mean"]]], fmt="o",
                           color=color(r), capsize=3, label=name(r))
        for t, ls in ((TEACHER_OF[kind], "--"), ("base", ":")):
            ax[i].axhline(E[t]["mean"], color=color(t), ls=ls, lw=1.1, label=name(t))
        ax[i].set_xlabel("training VM-hours per seed (v6e-4; merges ~0)")
        ax[i].set_title(f"{title} vs compute", fontsize=10); ax[i].grid(alpha=0.3)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    f = os.path.join(media, f"{prefix}_f3_accuracy_vs_vmh.png"); fig.savefig(f, dpi=140); plt.close(fig); files.append(f)

    comps = [k for k in rep["contrasts"]["v2"]]
    fig, ax = plt.subplots(1, 3, figsize=(17, 0.42 * len(comps) + 1.8), sharey=True)
    for i, (kind, title) in enumerate(KINDS):
        C = rep["contrasts"][kind]
        keys = [k for k in C]
        for j, k in enumerate(keys):
            d = C[k]
            col = "#2b6cb0" if d["ci95"][0] > 0 else "#c53030" if d["ci95"][1] < 0 else "#4a5568"
            ax[i].errorbar(100 * d["diff"], j, xerr=[[100 * (d["diff"] - d["ci95"][0])], [100 * (d["ci95"][1] - d["diff"])]],
                           fmt="o" if d["family"] == "primary" else "s", color=col, capsize=3, ms=5)
        ax[i].axvline(0, color="k", lw=0.8)
        ax[i].axvspan(-100 * MARGIN, 100 * MARGIN, color="#c6f6d5", alpha=0.4)
        tname = lambda p: "routed teacher" if p in ("tmath", "tif") else name(p)
        ax[i].set_yticks(range(len(keys)), [" − ".join(tname(p) for p in k.split(" - ")) for k in keys], fontsize=8)
        ax[i].invert_yaxis(); ax[i].set_xlabel("difference, pp (95% CI; green band ±3 pp)")
        ax[i].set_title(title, fontsize=10); ax[i].grid(axis="x", alpha=0.3)
    fig.tight_layout()
    f = os.path.join(media, f"{prefix}_f4_paired_diffs.png"); fig.savefig(f, dpi=140); plt.close(fig); files.append(f)

    arms = [r for r in order if r not in REF and r in rep["headroom"].get("v2", {})]
    fig, ax = plt.subplots(figsize=(14, 4.4))
    w = 0.8 / max(1, len(arms))
    for j, a in enumerate(arms):
        for i, (kind, _) in enumerate(KINDS):
            h = rep["headroom"].get(kind, {}).get(a)
            if not h:
                continue
            x = i + (j - (len(arms) - 1) / 2) * w
            ax.bar(x, 100 * h["closed"], w, color=color(a), label=name(a) if i == 0 else None)
            ax.errorbar(x, 100 * h["closed"], yerr=[[100 * (h["closed"] - h["ci95"][0])], [100 * (h["ci95"][1] - h["closed"])]],
                        color="k", capsize=1.5, lw=0.7)
    ax.axhline(100, color="k", ls="--", lw=1); ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(3), [f"{t}\n(gap {100 * rep['headroom'][k]['mopd']['teacher_gap']:+.1f} pp)" for k, t in KINDS])
    ax.set_ylim(-100, 200); ax.set_ylabel("% of teacher − base gap closed (95% CI)")
    ax.legend(frameon=False, ncol=6, fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Headroom closed at step 60 (CIs clipped to the axis)", fontsize=10)
    fig.tight_layout()
    f = os.path.join(media, f"{prefix}_f5_headroom.png"); fig.savefig(f, dpi=140); plt.close(fig); files.append(f)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    for i, (kind, title) in enumerate(KINDS):
        E = rep["evals"][kind]
        for k, a in enumerate(SEEDED + ("sft@120",)):
            if a in E:
                ys = E[a]["per_row"]
                ax[i].scatter([k] * len(ys), ys, color=color(a), s=28, zorder=3)
                ax[i].hlines(E[a]["mean"], k - 0.3, k + 0.3, color=color(a))
        ax[i].set_xticks(range(len(SEEDED) + 1), [name(a) for a in SEEDED + ("sft@120",)], rotation=30, ha="right", fontsize=8)
        ax[i].set_title(f"{title}: per-seed means", fontsize=10); ax[i].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    f = os.path.join(media, f"{prefix}_f7_per_seed.png"); fig.savefig(f, dpi=140); plt.close(fig); files.append(f)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--no-glmm", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--media", default=None)
    a = ap.parse_args(argv)
    rep = build(a.roots, a.n_boot, with_glmm=not a.no_glmm)
    if a.out:
        open(a.out, "w").write(json.dumps(rep, indent=1, default=float))
    if a.media:
        for f in charts(rep, a.media):
            print(f)
    for kind, title in KINDS:
        for r, m in rep["evals"][kind].items():
            print(f"{title:15s} {r:18s} {m['mean']:.4f} [{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}] n={m['rows']}")
        for k, d in rep["contrasts"][kind].items():
            print(f"{title:15s} {k:32s} {100 * d['diff']:+.2f} pp [{100 * d['ci95'][0]:+.2f}, {100 * d['ci95'][1]:+.2f}] "
                  f"p={d['p']:.3f}{' holm=%.3f' % d['p_holm'] if 'p_holm' in d else ''}{' EQUIV' if d['equivalent'] else ''}")


if __name__ == "__main__":
    main()
