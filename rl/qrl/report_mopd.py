"""MOPD plan report (docs/mopd-tpu-plan.md): pilot go/no-go gates, and per-arm tables for the study.

Reads a local copy of the plan's synced results ($B/results/<vm>-<boot>/<run>/...); a run that appears under several
boots (a re-run after a preemption or an infrastructure failure) is taken from the boot that has its final eval.

    python -m qrl.report_mopd gates --root /tmp/qrl-mopd/res --plan tpu/mopdpilot.json --spent 61.2 --out gates.json
"""
import argparse
import glob
import json
import math
import os
import re

import numpy as np

from .analyze import load_events, summarize

S_STEP_P90 = {"mopd-s0": 222.0, "mopd-dist-s0": 272.0}   # plan table "Expected s/step", modelled P90
HID_TOL = 1e-3
KL_9B_MAX_X = 5.0
EVAL_RE = re.compile(r"^eval(?:_(ifeval|ifbench))?\.([^.]+)\.json$")


def runs(root, final_label=None):
    """{run name: dir}, preferring the boot whose run has the final AIME eval, then the latest boot."""
    best = {}
    for d in sorted(glob.glob(os.path.join(root, "*", "*"))):
        if not os.path.isdir(d) or os.path.basename(d) in ("teachers", "pilot"):
            continue
        n = os.path.basename(d)
        has = bool(final_label) and os.path.exists(os.path.join(d, f"eval.{final_label}.json"))
        if n not in best or has or not best[n][1]:
            best[n] = (d, has)
    return {n: d for n, (d, _) in best.items()}


def eval_of(d, kind, label):
    f = os.path.join(d, ("eval" if kind == "v2" else f"eval_{kind}") + f".{label}.json")
    if not os.path.exists(f):
        return None
    return json.load(open(f)).get(label)


def boot_ci(r, n_boot=4000, seed=0):
    """Pass@1 (or strict IF accuracy) with a 95% bootstrap CI over problems and samples."""
    per = r.get("per_problem") or []
    if not per:
        return {"mean": r.get("pass@1"), "ci95": None}
    n = r["samples"]
    p = np.array([q["correct"] / n for q in per])
    rng = np.random.default_rng(seed)
    draws = [rng.binomial(n, p[rng.integers(0, len(p), len(p))]).mean() / n for _ in range(n_boot)]
    return {"mean": float(p.mean()), "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "problems": len(per), "samples": n}


def train_events(d):
    return sorted((e for e in load_events(d) if e["kind"] == "train"), key=lambda e: e["step"])


def first_last(xs, k=5):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(xs) < 2:
        return None, None
    k = min(k, len(xs) // 2)
    return float(np.mean(xs[:k])), float(np.mean(xs[-k:]))


def score_kl(path, prefix="kl_behav_teacher_", batches=None):
    """Token-weighted KL(behaviour || teacher) on sampled tokens from teacher score events."""
    sc = [json.loads(x) for x in open(path) if '"score"' in x]
    if batches is not None:
        sc = [e for e in sc if e.get("batch") in batches]
    num = den = 0.0
    for e in sc:
        for k, v in e.items():
            if k.startswith(prefix):
                n = e.get("tokens_" + k[len(prefix):], 0)
                num += v * n
                den += n
    return (num / den if den else None), (float(np.mean([e["tok_s"] for e in sc])) if sc else None)


def gates(root, plan, spent):
    steps = plan["steps"]
    R = runs(root, f"step{steps}")
    out = {"root": root, "runs": sorted(R), "gates": {}}
    G = out["gates"]
    arms = {"mopd-s0": "opd", "mopd-dist-s0": "opd_dist", "mixrl-s0": "grpo"}
    summ = {}
    for n in arms:
        if n in R and os.path.exists(os.path.join(R[n], "config.json")):
            try:
                summ[n] = summarize(R[n])
            except (KeyError, ValueError, ZeroDivisionError) as e:
                summ[n] = {"complete": False, "error": repr(e)}
    out["summary"] = {n: {k: v for k, v in s.items() if not isinstance(v, (list, dict))} for n, s in summ.items()}

    # 1. end to end; learner idle < 10% in the distillation arms
    g1 = {}
    for n in ("mopd-s0", "mopd-dist-s0"):
        s = summ.get(n, {})
        fin = n in R and eval_of(R[n], "v2", f"step{steps}") is not None
        g1[n] = {"steps": s.get("steps"), "final_eval": fin, "learner_idle_frac": s.get("learner_idle_frac"),
                 "rollout_idle_frac": s.get("rollout_idle_frac")}
    G["1_end_to_end_learner_idle"] = {"detail": g1, "pass": all(
        v["steps"] == steps and v["final_eval"] and v["learner_idle_frac"] is not None and v["learner_idle_frac"] < 0.10
        for v in g1.values())}

    # 2. s/step inside the modelled P90
    g2 = {n: {"s_per_step": summ.get(n, {}).get("s_per_step"), "p90": p90} for n, p90 in S_STEP_P90.items()}
    G["2_s_per_step"] = {"detail": g2, "pass": all(v["s_per_step"] is not None and v["s_per_step"] <= v["p90"]
                                                   for v in g2.values())}

    # 3. MOPD-dist at --mb-tokens 8192; shipped-hidden loss vs direct teacher logits within 1e-3
    g3 = {}
    if "mopd-dist-s0" in R:
        cfg = json.load(open(os.path.join(R["mopd-dist-s0"], "config.json")))
        tr = train_events(R["mopd-dist-s0"])
        hc = [e["hid_check_abs"] for e in tr if "hid_check_abs" in e]
        hm = [e["hid_check_max"] for e in tr if "hid_check_max" in e]
        g3 = {"mb_tokens_8192": "--mb-tokens 8192" in cfg.get("learner_args", ""), "steps": len(tr),
              "hid_check_mean_abs_max_over_steps": max(hc) if hc else None,
              "hid_check_max_abs_max_over_steps": max(hm) if hm else None}
    G["3_dist_fits_and_matches"] = {"detail": g3, "pass": bool(g3) and g3["mb_tokens_8192"] and g3["steps"] == steps
                                    and g3["hid_check_mean_abs_max_over_steps"] is not None
                                    and g3["hid_check_mean_abs_max_over_steps"] <= HID_TOL}

    # 4. reverse KL to the routed teacher falls in both arms; entropy does not collapse
    g4 = {}
    for n, key in (("mopd-s0", "kl_teacher"), ("mopd-dist-s0", "kl_exact")):
        if n not in R:
            continue
        tr = train_events(R[n])
        kf, kl_ = first_last([e.get(key) for e in tr])
        ef, el = first_last([e.get("entropy") for e in tr])
        dom = {}
        for d in ("math", "if"):
            a, b = first_last([e.get(f"{key}_{d}") for e in tr])
            c, f = first_last([e.get(f"entropy_{d}") for e in tr])
            dom[d] = {"kl_first5": a, "kl_last5": b, "entropy_first5": c, "entropy_last5": f}
        g4[n] = {"metric": key, "kl_first5": kf, "kl_last5": kl_, "entropy_first5": ef, "entropy_last5": el,
                 "by_domain": dom,
                 "kl_falls": kf is not None and kl_ < kf,
                 "entropy_ok": ef is not None and el >= 0.5 * ef}
    G["4_kl_falls_entropy_ok"] = {"detail": g4, "pass": len(g4) == 2 and all(v["kl_falls"] and v["entropy_ok"]
                                                                            for v in g4.values())}

    # 5. 9B starting KL measured; > 5x the same-origin KL -> switch the single teacher
    g5 = {}
    if "probe" in R:
        for tag in ("2B", "4B", "9B"):
            p = os.path.join(R["probe"], f"score_{tag}.jsonl")
            if os.path.exists(p):
                kl, tok_s = score_kl(p)
                g5[tag] = {"kl_start": kl, "prefill_tok_s": tok_s}
    so = None
    if "mopd-s0" in R and os.path.exists(os.path.join(R["mopd-s0"], "events", "teacher.jsonl")):
        so, _ = score_kl(os.path.join(R["mopd-s0"], "events", "teacher.jsonl"), batches={0})
    g5["same_origin_kl_start"] = so
    k9 = g5.get("9B", {}).get("kl_start")
    g5["ratio_9b_to_same_origin"] = (k9 / so) if (k9 is not None and so) else None
    g5["switch_single_teacher"] = g5["ratio_9b_to_same_origin"] is not None and g5["ratio_9b_to_same_origin"] > KL_9B_MAX_X
    if "base9b" in R:
        g5["base9b_aime"] = boot_ci(eval_of(R["base9b"], "v2", "final") or {})
        g5["base9b_ifeval"] = boot_ci(eval_of(R["base9b"], "ifeval", "final") or {})
    G["5_9b_starting_kl"] = {"detail": g5, "pass": k9 is not None and so is not None}

    # 6. teacher export round trip (the delete/hold checks are added by the caller)
    g6 = {}
    if "roundtrip" in R:
        for dom, run, kind in (("math", "tmath", "v2"), ("if", "tif", "ifeval")):
            a = eval_of(R[run], kind, "final") if run in R else None
            b = eval_of(R["roundtrip"], kind, run)
            if a and b:
                ca, cb = boot_ci(a), boot_ci(b)
                se = math.sqrt(sum(((c["ci95"][1] - c["ci95"][0]) / 3.92) ** 2 for c in (ca, cb)))
                g6[dom] = {"logged": ca, "reproduced": cb, "diff": cb["mean"] - ca["mean"], "se_diff": se,
                           "within_noise": abs(cb["mean"] - ca["mean"]) <= 2 * se}
    G["6_teacher_roundtrip"] = {"detail": g6, "pass": len(g6) == 2 and all(v["within_noise"] for v in g6.values())}

    # 7. re-price (filled by reprice())
    out["evals"] = arm_evals(R, steps)
    out["spent_usd"] = spent
    return out


def arm_evals(R, steps):
    ev = {}
    for n, d in sorted(R.items()):
        for f in sorted(os.listdir(d)):
            m = EVAL_RE.match(f)
            if m:
                kind, label = m.group(1) or "v2", m.group(2)
                r = json.load(open(os.path.join(d, f))).get(label)
                if r:
                    ev.setdefault(n, {}).setdefault(kind, {})[label] = boot_ci(r)
    return ev


ARM_STYLE = {"mixrl": ("Mix-RL", "#5a645c"), "mopd": ("MOPD", "#2b6cb0"), "mopd-dist": ("MOPD-dist", "#c05621"),
             "opd9b": ("OPD-9B", "#6b46c1"), "opd-tmath": ("OPD (T-math only)", "#6b46c1"), "tmath": ("T-math", "#2f855a"), "tif": ("T-IF", "#b83280")}


def arm_of(run):
    return run.rsplit("-s", 1)[0] if run.rsplit("-s", 1)[-1].isdigit() else run


def series(d, key):
    tr = train_events(d)
    return np.array([e["step"] + 1 for e in tr]), np.array([e.get(key, np.nan) for e in tr], dtype=float)


def smooth(x, w=5):
    x = np.asarray(x, float)
    return np.array([np.nanmean(x[max(0, i - w + 1):i + 1]) if np.isfinite(x[max(0, i - w + 1):i + 1]).any() else np.nan
                     for i in range(len(x))])


def charts(root, plan, media, prefix="mopd"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(media, exist_ok=True)
    steps = plan["steps"]
    R = runs(root, f"step{steps}")
    order = [r for a in ARM_STYLE for r in sorted(R) if arm_of(r) == a]
    out = []

    # training curves: reward, reverse KL to the routed teacher, entropy; per domain
    rows = [("reward", "train reward (5-step mean)"), ("kl", "reverse KL to routed teacher\n(sampled tokens; MOPD-dist: exact)"),
            ("entropy", "policy entropy (nats/token)")]
    fig, ax = plt.subplots(len(rows), 2, figsize=(12, 9), sharex=True)
    for r in order:
        name, color = ARM_STYLE[arm_of(r)]
        ls = "--" if r.endswith("-s1") else "-"
        for j, dom in enumerate(("math", "if")):
            for i, (what, _) in enumerate(rows):
                key = {"reward": f"reward_{dom}", "entropy": f"entropy_{dom}",
                       "kl": f"kl_exact_{dom}" if arm_of(r) == "mopd-dist" else f"kl_teacher_{dom}"}[what]
                x, y = series(R[r], key)
                if what == "entropy" and not np.isfinite(y).any():
                    x, y = series(R[r], "entropy") if arm_of(r) in ("tmath", "tif") and \
                        (arm_of(r) == "tmath") == (dom == "math") else (x, y)
                if np.isfinite(y).any():
                    ax[i, j].plot(x, smooth(y) if what == "reward" else y, color=color, ls=ls, lw=1.4,
                                  label=f"{name} {r.rsplit('-', 1)[-1] if r.rsplit('-s', 1)[-1].isdigit() else ''}")
    for j, dom in enumerate(("math (AIME-style prompts)", "instruction following (RLVR-IFeval)")):
        ax[0, j].set_title(dom)
        for i, (_, lab) in enumerate(rows):
            ax[i, j].set_ylabel(lab if j == 0 else "")
            ax[i, j].grid(alpha=0.3)
        ax[-1, j].set_xlabel("training step")
    seen = {}
    for a in ax.flat:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            seen.setdefault(ll, hh)
    h, lab = list(seen.values()), list(seen)
    if h:
        fig.legend(h, lab, loc="upper center", ncol=min(6, len(h)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    f = os.path.join(media, f"{prefix}_train_curves.png"); fig.savefig(f, dpi=130); plt.close(fig); out.append(f)

    # speed: s/step and learner/rollout idle
    summ = {}
    for r in order:
        try:
            summ[r] = summarize(R[r])
        except (KeyError, ValueError, ZeroDivisionError):
            pass
    summ = {r: s for r, s in summ.items() if s.get("s_per_step")}
    if summ:
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
        names = list(summ)
        cols = [ARM_STYLE[arm_of(r)][1] for r in names]
        ax[0].bar(names, [summ[r]["s_per_step"] for r in names], color=cols)
        for r, p90 in S_STEP_P90.items():
            if r in names:
                ax[0].plot([names.index(r) - 0.4, names.index(r) + 0.4], [p90, p90], "k--", lw=1)
        ax[0].set_ylabel("seconds per step (steady state)")
        ax[0].set_title("s/step (dashed: modelled P90)")
        w = 0.38
        xs = np.arange(len(names))
        ax[1].bar(xs - w / 2, [100 * summ[r]["learner_idle_frac"] for r in names], w, color=cols, label="learner idle")
        ax[1].bar(xs + w / 2, [100 * summ[r]["rollout_idle_frac"] for r in names], w, color=cols, alpha=0.45,
                  label="rollout idle")
        ax[1].axhline(10, color="k", ls="--", lw=1)
        ax[1].set_xticks(xs, names)
        ax[1].set_ylabel("% of steady-state wall time")
        ax[1].set_title("idle time (solid: learner, light: rollout; dashed: 10% gate)")
        for a in ax:
            a.tick_params(axis="x", rotation=30)
            a.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        f = os.path.join(media, f"{prefix}_speed.png"); fig.savefig(f, dpi=130); plt.close(fig); out.append(f)

    # evals: AIME pass@1, IFEval and IFBench strict prompt accuracy, 95% bootstrap CIs
    ev = arm_evals(R, steps)
    pick = []
    for r in ["base"] + order:
        lab = "final" if r in ("base", "tmath", "tif") else f"step{steps}"
        if r in ev and any(lab in ev[r].get(k, {}) for k in ("v2", "ifeval", "ifbench")):
            pick.append((r, lab))
    if pick:
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for i, (kind, title) in enumerate((("v2", "AIME pass@1"), ("ifeval", "IFEval (strict, all instructions)"),
                                           ("ifbench", "IFBench (strict, all instructions)"))):
            for k, (r, lab) in enumerate(pick):
                m = ev[r].get(kind, {}).get(lab)
                if not m:
                    continue
                c = ARM_STYLE.get(arm_of(r), ("base", "#a0aec0"))[1]
                ax[i].bar(k, m["mean"], color=c)
                if m.get("ci95"):
                    ax[i].errorbar(k, m["mean"], yerr=[[m["mean"] - m["ci95"][0]], [m["ci95"][1] - m["mean"]]],
                                   color="k", capsize=3, lw=1)
            ax[i].set_xticks(range(len(pick)), [r for r, _ in pick], rotation=35, ha="right")
            ax[i].set_title(title)
            ax[i].grid(axis="y", alpha=0.3)
        fig.tight_layout()
        f = os.path.join(media, f"{prefix}_evals.png"); fig.savefig(f, dpi=130); plt.close(fig); out.append(f)
    return out


STUDY_ARMS = ["mixrl", "opd-tmath", "opd9b", "mopd", "mopd-dist"]
KINDS = (("v2", "AIME pass@1"), ("ifeval", "IFEval strict"), ("ifbench", "IFBench strict"))
TEACHER_OF = {"v2": "tmath", "ifeval": "tif", "ifbench": "tif"}
COMPARE = (("mopd", "mixrl"), ("mopd-dist", "mixrl"), ("mopd", "opd-tmath"), ("mopd-dist", "mopd"),
           ("opd-tmath", "mixrl"), ("mopd", "opd9b"))


def problems(r):
    """{problem id: (correct, samples)} of one eval result."""
    return {q["id"]: (float(q["correct"]), r["samples"]) for q in (r or {}).get("per_problem") or []}


def base_eval(R, kind, base_aime=None):
    if "base" in R and eval_of(R["base"], kind, "final"):
        return eval_of(R["base"], kind, "final")
    if kind == "v2" and base_aime:  # $B/pilot/eval_base.json: the base model's AIME eval from the qwen38 study
        return json.load(open(base_aime)).get("base")
    return None


def pooled(R, arm, kind, label):
    """Seeds of ARM pooled per problem (problems every seed has): {id: (correct, samples)}, and the seeds used."""
    seeds = sorted(r for r in R if arm_of(r) == arm and r != arm)
    pp = [problems(eval_of(R[r], kind, label)) for r in seeds]
    pp = [p for p in pp if p]
    if not pp:
        return {}, []
    ids = set.intersection(*(set(p) for p in pp))
    return {i: (sum(p[i][0] for p in pp), sum(p[i][1] for p in pp)) for i in ids}, len(pp)


def boot_pooled(pp, n_boot=4000, seed=0):
    """Mean pass@1 over problems (equal weight per problem) with a 95% bootstrap CI over problems and samples."""
    if not pp:
        return None
    ids = sorted(pp)
    p = np.array([pp[i][0] / pp[i][1] for i in ids]); n = np.array([pp[i][1] for i in ids])
    rng = np.random.default_rng(seed)
    d = [(rng.binomial(n[ix], p[ix]) / n[ix]).mean() for ix in (rng.integers(0, len(ids), len(ids)) for _ in range(n_boot))]
    return {"mean": float(p.mean()), "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
            "problems": len(ids), "samples_per_problem": int(n.min())}


def boot_paired(pa, pb, n_boot=4000, seed=0):
    """Mean of A - B over the problems both have, with a paired 95% bootstrap CI (same problem draws for A and B)."""
    ids = sorted(set(pa) & set(pb))
    if not ids:
        return None
    a = np.array([pa[i][0] / pa[i][1] for i in ids]); na = np.array([pa[i][1] for i in ids])
    b = np.array([pb[i][0] / pb[i][1] for i in ids]); nb = np.array([pb[i][1] for i in ids])
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(ids), len(ids))
        d.append((rng.binomial(na[ix], a[ix]) / na[ix]).mean() - (rng.binomial(nb[ix], b[ix]) / nb[ix]).mean())
    lo, hi = float(np.quantile(d, .025)), float(np.quantile(d, .975))
    return {"diff": float(a.mean() - b.mean()), "ci95": [lo, hi], "problems": len(ids), "significant": lo > 0 or hi < 0}


def study(root, plan, base_aime=None):
    steps = plan["steps"]
    R = runs(root, f"step{steps}")
    labels = [f"step{s}" for s in str(plan.get("ckpt_steps", steps)).split(",")]
    arms = [a for a in STUDY_ARMS if any(arm_of(r) == a and r != a for r in R)]
    out = {"root": root, "runs": sorted(R), "arms": arms, "labels": labels, "evals": {}, "diffs": {}, "recovery": {},
           "speed": {}, "kl_start": {}}
    for kind, _ in KINDS:
        b = problems(base_eval(R, kind, base_aime))
        E = out["evals"].setdefault(kind, {})
        E["base"] = {"step0": boot_pooled(b)}
        for t in ("tmath", "tif"):
            if t in R:
                E[t] = {"final": boot_pooled(problems(eval_of(R[t], kind, "final")))}
        for a in arms:
            for lab in labels:
                pp, ns = pooled(R, a, kind, lab)
                if pp:
                    E.setdefault(a, {})[lab] = dict(boot_pooled(pp), seeds=ns)
                    for s in sorted(r for r in R if arm_of(r) == a and r != a):
                        one = eval_of(R[s], kind, lab)
                        if one:
                            E[a][lab].setdefault("per_seed", {})[s] = one["pass@1"]
        last = labels[-1]
        for x, y in COMPARE:
            px, _ = pooled(R, x, kind, last)
            py, _ = pooled(R, y, kind, last)
            d = boot_paired(px, py)
            if d:
                out["diffs"].setdefault(kind, {})[f"{x} - {y}"] = d
        t = E.get(TEACHER_OF[kind], {}).get("final")
        bm = E["base"]["step0"]
        if t and bm and t["mean"] != bm["mean"]:
            for a in arms:
                m = E.get(a, {}).get(last)
                if m:
                    out["recovery"].setdefault(kind, {})[a] = (m["mean"] - bm["mean"]) / (t["mean"] - bm["mean"])
    for r in sorted(R):
        if arm_of(r) in arms or r in ("tmath", "tif"):
            try:
                s = summarize(R[r])
                out["speed"][r] = {k: s.get(k) for k in ("s_per_step", "wall_s", "learner_idle_frac", "rollout_idle_frac",
                                                         "truncated_frac_gen", "steps")}
            except (KeyError, ValueError, ZeroDivisionError):
                pass
            tr = train_events(R[r])
            if tr:
                key = "kl_exact" if arm_of(r) == "mopd-dist" else "kl_teacher"
                ks = [e.get(key) for e in tr]
                if ks[0] is not None:
                    out["kl_start"][r] = {"first": ks[0], "last5": first_last(ks)[1],
                                          **{d: tr[0].get(f"{key}_{d}") for d in ("math", "if")}}
    return out


def study_charts(rep, media, prefix="mopd_full"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(media, exist_ok=True)
    out = []
    labels = rep["labels"]
    xs = [0] + [int(l[4:]) for l in labels]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for i, (kind, title) in enumerate(KINDS):
        E = rep["evals"].get(kind, {})
        b = E.get("base", {}).get("step0")
        for a in rep["arms"]:
            name, color = ARM_STYLE[a]
            ms = [b] + [E.get(a, {}).get(l) for l in labels]
            pts = [(x, m) for x, m in zip(xs, ms) if m]
            if len(pts) < 2:
                continue
            ax[i].plot([x for x, _ in pts], [m["mean"] for _, m in pts], "o-", color=color, label=name, lw=1.6)
            ax[i].fill_between([x for x, _ in pts], [m["ci95"][0] for _, m in pts], [m["ci95"][1] for _, m in pts],
                               color=color, alpha=0.10)
        t = E.get(TEACHER_OF[kind], {}).get("final")
        if t:
            tn, tc = ARM_STYLE[TEACHER_OF[kind]]
            ax[i].axhline(t["mean"], color=tc, ls="--", lw=1.2, label=f"{tn} teacher")
        ax[i].set_title(title + " (seeds pooled, 95% CI)")
        ax[i].set_xlabel("training step")
        ax[i].set_xticks(xs)
        ax[i].grid(alpha=0.3)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    f = os.path.join(media, f"{prefix}_evals_by_step.png"); fig.savefig(f, dpi=130); plt.close(fig); out.append(f)

    comps = [c for c in (f"{x} - {y}" for x, y in COMPARE) if any(c in rep["diffs"].get(k, {}) for k, _ in KINDS)]
    if comps:
        fig, ax = plt.subplots(1, 3, figsize=(15, 0.6 * len(comps) + 1.6), sharey=True)
        nice = lambda c: " − ".join(ARM_STYLE[p][0] for p in c.split(" - "))
        for i, (kind, title) in enumerate(KINDS):
            for j, c in enumerate(comps):
                d = rep["diffs"].get(kind, {}).get(c)
                if not d:
                    continue
                col = "#2b6cb0" if d["ci95"][0] > 0 else "#c53030" if d["ci95"][1] < 0 else "#4a5568"
                ax[i].errorbar(100 * d["diff"], j, xerr=[[100 * (d["diff"] - d["ci95"][0])], [100 * (d["ci95"][1] - d["diff"])]],
                               fmt="o", color=col, capsize=3)
            ax[i].axvline(0, color="k", lw=0.8)
            ax[i].set_yticks(range(len(comps)), [nice(c) for c in comps])
            ax[i].invert_yaxis()
            ax[i].set_xlabel("difference, percentage points")
            ax[i].set_title(f"{title}, step {labels[-1][4:]}")
            ax[i].grid(axis="x", alpha=0.3)
        fig.tight_layout()
        f = os.path.join(media, f"{prefix}_paired_diffs.png"); fig.savefig(f, dpi=130); plt.close(fig); out.append(f)
    return out


FINAL_ROWS = ["base", "tmath", "tif", "mixrl", "mopd", "mopd-dist", "opd-tmath"]
FINAL_DIFFS = (("mopd", "mixrl"), ("mopd-dist", "mopd"), ("mopd", "opd-tmath"), ("mopd-dist", "mixrl"),
               ("opd-tmath", "mixrl"))
TREND_KEYS = ("kl", "kl_math", "kl_if", "entropy", "entropy_math", "entropy_if")


def cell_matrix(evals):
    """Eval results (one per seed or repeat) -> problem ids, correct counts [rows, problems], samples per row."""
    pp = [problems(r) for r in evals if r]
    if not pp:
        return [], None, None
    ids = sorted(set.intersection(*(set(p) for p in pp)))
    return ids, np.array([[p[i][0] for i in ids] for p in pp]), np.array([pp[k][ids[0]][1] for k in range(len(pp))])


def final_cells(R, row, kind, label, base_extra=()):
    """{problem id: column} cells for one table row: base (study rerun + extra base evals), a teacher, or an arm's seeds."""
    if row == "base":
        ev = [base_eval(R, kind)]
        for p in base_extra:
            m = re.match(r"^eval_(ifeval|ifbench)\.", os.path.basename(p))
            if (m.group(1) if m else "v2") == kind:
                d = json.load(open(p))
                ev.append(d.get("final") or d.get("base"))
    elif row in ("tmath", "tif"):
        ev = [eval_of(R[row], kind, "final")] if row in R else []
    else:
        ev = [eval_of(R[r], kind, label) for r in sorted(R) if arm_of(r) == row and r != row]
    return cell_matrix(ev)


def hier_draws(c, n, prob, rng):
    """One hierarchical draw: the given problem draw, seeds (rows) resampled, then samples redrawn per cell."""
    rows = rng.integers(0, c.shape[0], c.shape[0])
    p = c[rows][:, prob] / n[rows][:, None]
    return (rng.binomial(n[rows][:, None], p) / n[rows][:, None]).mean()


def final(root, plan, base_extra=(), n_boot=10000, seed=0):
    """Step-final table with hierarchical 95% CIs (problems x seeds x samples), paired differences, headroom closed,
    speed and KL/entropy trends. Every quantity of one kind shares the same problem draws, so differences and
    headroom ratios are paired by problem."""
    steps = plan["steps"]
    R = runs(root, f"step{steps}")
    labels = [f"step{s}" for s in str(plan.get("ckpt_steps", steps)).split(",")]
    out = {"root": root, "runs": sorted(R), "labels": labels, "bootstrap": {"draws": n_boot, "levels":
           "problems, seeds (rows), samples (binomial)"}, "evals": {}, "diffs": {}, "headroom": {}, "speed": {},
           "trends": {}}
    for kind, _ in KINDS:
        cells = {}
        for row in FINAL_ROWS:
            for lab in (labels if row not in ("base", "tmath", "tif") else ["final"]):
                ids, c, n = final_cells(R, row, kind, lab, base_extra)
                if ids:
                    cells[(row, lab)] = (ids, c, n)
        common = sorted(set.intersection(*(set(v[0]) for v in cells.values())))
        idx = {k: np.array([v[0].index(i) for i in common]) for k, v in cells.items()}
        mat = {k: (v[1][:, idx[k]], v[2]) for k, v in cells.items()}
        rng = np.random.default_rng(seed)
        P = len(common)
        draws = {k: np.empty(n_boot) for k in mat}
        for t in range(n_boot):
            prob = rng.integers(0, P, P)
            for k, (c, n) in mat.items():
                draws[k][t] = hier_draws(c, n, prob, rng)
        point = {k: float((c / n[:, None]).mean()) for k, (c, n) in mat.items()}
        E = out["evals"].setdefault(kind, {})
        for (row, lab), m in point.items():
            c, n = mat[(row, lab)]
            E.setdefault(row, {})[lab] = {"mean": m, "ci95": [float(np.quantile(draws[(row, lab)], .025)),
                                                              float(np.quantile(draws[(row, lab)], .975))],
                                          "rows": int(c.shape[0]), "samples": [int(x) for x in n], "problems": P,
                                          "per_row": [float(x) for x in (c / n[:, None]).mean(1)]}
        last = labels[-1]
        for lab in labels:
            for a, b in FINAL_DIFFS:
                if (a, lab) in draws and (b, lab) in draws:
                    d = draws[(a, lab)] - draws[(b, lab)]
                    lo, hi = float(np.quantile(d, .025)), float(np.quantile(d, .975))
                    out["diffs"].setdefault(kind, {}).setdefault(lab, {})[f"{a} - {b}"] = {
                        "diff": point[(a, lab)] - point[(b, lab)], "ci95": [lo, hi],
                        "ci90": [float(np.quantile(d, .05)), float(np.quantile(d, .95))], "significant": lo > 0 or hi < 0}
        tk = (TEACHER_OF[kind], "final")
        if tk in draws and ("base", "final") in draws:
            for a in ("mixrl", "mopd", "mopd-dist", "opd-tmath"):
                if (a, last) in draws:
                    gap = draws[tk] - draws[("base", "final")]
                    ok = np.abs(gap) > 1e-9
                    r = (draws[(a, last)][ok] - draws[("base", "final")][ok]) / gap[ok]
                    out["headroom"].setdefault(kind, {})[a] = {
                        "closed": (point[(a, last)] - point[("base", "final")]) / (point[tk] - point[("base", "final")]),
                        "ci95": [float(np.quantile(r, .025)), float(np.quantile(r, .975))],
                        "teacher_gap": point[tk] - point[("base", "final")],
                        "gap_ci95": [float(np.quantile(gap, .025)), float(np.quantile(gap, .975))]}
    for r in sorted(R):
        if arm_of(r) not in FINAL_ROWS or r == "base":
            continue
        try:
            s = summarize(R[r])
            out["speed"][r] = {k: s.get(k) for k in ("s_per_step", "wall_s", "learner_idle_frac", "rollout_idle_frac",
                                                     "train_s_mean", "gen_s_mean", "train_tokens", "steps")}
        except (KeyError, ValueError, ZeroDivisionError, FileNotFoundError):
            pass
        tr = train_events(R[r])
        if tr:
            key = "kl_exact" if arm_of(r) == "mopd-dist" else "kl_teacher"
            T = {}
            for k in TREND_KEYS:
                src = k.replace("kl", key, 1) if k.startswith("kl") else k
                xs = [e.get(src) for e in tr]
                f5, l5 = first_last(xs)
                if f5 is not None:
                    T[k] = {"first5": f5, "last5": l5, "first": xs[0], "latest": xs[-1]}
            out["trends"][r] = T
    return out


def final_charts(rep, root, plan, media, prefix="mopd"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(media, exist_ok=True)
    out = []
    last = rep["labels"][-1]
    name = lambda r: {"base": "base 2B"}.get(r, ARM_STYLE.get(r, (r,))[0])
    color = lambda r: ARM_STYLE.get(r, ("", "#a0aec0"))[1]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for i, (kind, title) in enumerate(KINDS):
        E = rep["evals"][kind]
        rows = [(r, "final" if r in ("base", "tmath", "tif") else last) for r in FINAL_ROWS if r in E]
        for k, (r, lab) in enumerate(rows):
            m = E[r][lab]
            ax[i].bar(k, m["mean"], color=color(r), alpha=0.55 if r in ("base", "tmath", "tif") else 1.0)
            ax[i].errorbar(k, m["mean"], yerr=[[m["mean"] - m["ci95"][0]], [m["ci95"][1] - m["mean"]]], color="k",
                           capsize=3, lw=1)
            ax[i].text(k, m["ci95"][1] + 0.003, f"{m['mean']:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax[i].set_xticks(range(len(rows)), [name(r) for r, _ in rows], rotation=35, ha="right")
        lo = min(E[r][lab]["ci95"][0] for r, lab in rows)
        ax[i].set_ylim(max(0, lo - 0.05), None)
        ax[i].set_title(f"{title}, step {last[4:]} (2 seeds pooled, 95% CI)")
        ax[i].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    f = os.path.join(media, f"{prefix}_evals.png"); fig.savefig(f, dpi=130); plt.close(fig); out.append(f)

    xs = [0] + [int(l[4:]) for l in rep["labels"]]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for i, (kind, title) in enumerate(KINDS):
        E = rep["evals"][kind]
        b = E["base"]["final"]
        for a in ("mixrl", "mopd", "mopd-dist", "opd-tmath"):
            ms = [b] + [E.get(a, {}).get(l) for l in rep["labels"]]
            pts = [(x, m) for x, m in zip(xs, ms) if m]
            ax[i].plot([x for x, _ in pts], [m["mean"] for _, m in pts], "o-", color=color(a), label=name(a), lw=1.6)
            ax[i].fill_between([x for x, _ in pts], [m["ci95"][0] for _, m in pts], [m["ci95"][1] for _, m in pts],
                               color=color(a), alpha=0.08)
        t = TEACHER_OF[kind]
        ax[i].axhline(E[t]["final"]["mean"], color=color(t), ls="--", lw=1.2, label=f"{name(t)} teacher")
        ax[i].set_title(title + " (seeds pooled, 95% CI)")
        ax[i].set_xlabel("training step"); ax[i].set_xticks(xs); ax[i].grid(alpha=0.3)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    f = os.path.join(media, f"{prefix}_evals_by_step.png"); fig.savefig(f, dpi=130); plt.close(fig); out.append(f)

    comps = [f"{a} - {b}" for a, b in FINAL_DIFFS]
    fig, ax = plt.subplots(1, 3, figsize=(15, 0.55 * len(comps) + 1.6), sharey=True)
    for i, (kind, title) in enumerate(KINDS):
        for j, c in enumerate(comps):
            d = rep["diffs"].get(kind, {}).get(last, {}).get(c)
            if not d:
                continue
            col = "#2b6cb0" if d["ci95"][0] > 0 else "#c53030" if d["ci95"][1] < 0 else "#4a5568"
            ax[i].errorbar(100 * d["diff"], j, xerr=[[100 * (d["diff"] - d["ci95"][0])], [100 * (d["ci95"][1] - d["diff"])]],
                           fmt="o", color=col, capsize=3)
        ax[i].axvline(0, color="k", lw=0.8)
        ax[i].set_yticks(range(len(comps)), [" − ".join(name(p) for p in c.split(" - ")) for c in comps])
        ax[i].invert_yaxis(); ax[i].set_xlabel("difference, percentage points (95% CI)")
        ax[i].set_title(f"{title}, step {last[4:]}"); ax[i].grid(axis="x", alpha=0.3)
    fig.tight_layout()
    f = os.path.join(media, f"{prefix}_paired_diffs.png"); fig.savefig(f, dpi=130); plt.close(fig); out.append(f)

    arms = ("mixrl", "mopd", "mopd-dist", "opd-tmath")
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.2
    for j, a in enumerate(arms):
        for i, (kind, _) in enumerate(KINDS):
            h = rep["headroom"].get(kind, {}).get(a)
            if not h:
                continue
            x = i + (j - 1.5) * w
            ax.bar(x, 100 * h["closed"], w, color=color(a), label=name(a) if i == 0 else None)
            ax.errorbar(x, 100 * h["closed"], yerr=[[100 * (h["closed"] - h["ci95"][0])], [100 * (h["ci95"][1] - h["closed"])]],
                        color="k", capsize=2, lw=0.8)
    ax.axhline(100, color="k", ls="--", lw=1); ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(3), [f"{t}\n(teacher: {name(TEACHER_OF[k])}, gap "
                             f"{100 * rep['headroom'][k]['mopd']['teacher_gap']:+.1f} pp)" for k, t in KINDS])
    ax.set_ylabel("% of teacher − base gap closed (95% CI)")
    ax.set_ylim(-150, 250)
    ax.set_title(f"Share of teacher headroom closed at step {last[4:]} (CIs clipped to the axis)")
    ax.legend(frameon=False, ncol=4, loc="upper left"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    f = os.path.join(media, f"{prefix}_headroom.png"); fig.savefig(f, dpi=130); plt.close(fig); out.append(f)

    R = runs(root, f"step{plan['steps']}")
    order = [r for a in ("mopd", "mopd-dist", "opd-tmath", "mixrl") for r in sorted(R) if arm_of(r) == a]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    for r in order:
        a = arm_of(r)
        ls = "--" if r.endswith("-s1") else "-"
        key = "kl_exact" if a == "mopd-dist" else "kl_teacher"
        for j, sfx in enumerate(("", "_math", "_if")):
            if a != "mixrl":
                x, y = series(R[r], key + sfx)
                if np.isfinite(y).any():
                    ax[0, j].plot(x, y, color=color(a), ls=ls, lw=1.3, label=f"{name(a)} {r[-2:]}")
            x, y = series(R[r], "entropy" + sfx)
            if np.isfinite(y).any():
                ax[1, j].plot(x, smooth(y, 3), color=color(a), ls=ls, lw=1.3, label=f"{name(a)} {r[-2:]}")
    for j, t in enumerate(("all prompts", "math prompts", "IF prompts")):
        ax[0, j].set_yscale("log"); ax[0, j].set_title(f"reverse KL to the routed teacher, {t}")
        ax[1, j].set_title(f"policy entropy (3-step mean), {t}")
        ax[1, j].set_xlabel("training step")
        for i in range(2):
            ax[i, j].grid(alpha=0.3)
    ax[0, 0].set_ylabel("nats/token (MOPD-dist: exact full-vocab)")
    ax[1, 0].set_ylabel("nats/token")
    h, l = ax[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    f = os.path.join(media, f"{prefix}_kl_entropy.png"); fig.savefig(f, dpi=130); plt.close(fig); out.append(f)

    sp = rep["speed"]
    names = [r for a in ("tmath", "tif", "mixrl", "mopd", "mopd-dist", "opd-tmath") for r in sorted(sp) if arm_of(r) == a]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    cols = [color(arm_of(r)) for r in names]
    ax[0].bar(names, [sp[r]["s_per_step"] for r in names], color=cols)
    ax[0].set_ylabel("seconds per step (steady state)"); ax[0].set_title("s/step per run")
    ax[1].bar(names, [sp[r]["wall_s"] / 3600 for r in names], color=cols)
    ax[1].set_ylabel("hours"); ax[1].set_title(f"training wall-clock to step {plan['steps']} (excl. startup)")
    for a in ax:
        a.tick_params(axis="x", rotation=35); a.grid(axis="y", alpha=0.3)
        for lbl in a.get_xticklabels():
            lbl.set_ha("right")
    fig.tight_layout()
    f = os.path.join(media, f"{prefix}_speed.png"); fig.savefig(f, dpi=130); plt.close(fig); out.append(f)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gates", "charts", "study", "final"])
    ap.add_argument("--base-extra", nargs="*", default=[],
                    help="final: more base-model eval JSONs pooled into the base row (e.g. the pilot's base IF evals)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--media", default=None)
    ap.add_argument("--root", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--spent", type=float, default=0.0)
    ap.add_argument("--extra", default=None, help="JSON merged into gate 6/7 details (holds, delete test, reprice)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--base-aime", default=None, help="study: base-model AIME eval JSON ($B/pilot/eval_base.json)")
    a = ap.parse_args(argv)
    if a.what == "final":
        plan = json.load(open(a.plan))
        rep = final(a.root, plan, a.base_extra, a.n_boot)
        if a.out:
            open(a.out, "w").write(json.dumps(rep, indent=1, default=float))
        if a.media:
            for f in final_charts(rep, a.root, plan, a.media):
                print(f)
        last = rep["labels"][-1]
        for kind, title in KINDS:
            for row, v in rep["evals"][kind].items():
                m = v.get("final") or v.get(last)
                print(f"{title:15s} {row:10s} {m['mean']:.4f} [{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}] rows={m['rows']}")
            for c, d in rep["diffs"].get(kind, {}).get(last, {}).items():
                print(f"{title:15s} {c:22s} {100 * d['diff']:+.2f} pp [{100 * d['ci95'][0]:+.2f}, {100 * d['ci95'][1]:+.2f}]")
            for arm, h in rep["headroom"].get(kind, {}).items():
                print(f"{title:15s} headroom {arm:10s} {100 * h['closed']:.0f}% [{100 * h['ci95'][0]:.0f}, {100 * h['ci95'][1]:.0f}]")
        return
    if a.what == "charts":
        for f in charts(a.root, json.load(open(a.plan)), a.media):
            print(f)
        return
    if a.what == "study":
        rep = study(a.root, json.load(open(a.plan)), a.base_aime)
        if a.out:
            open(a.out, "w").write(json.dumps(rep, indent=1, default=float))
        if a.media:
            for f in study_charts(rep, a.media) + charts(a.root, json.load(open(a.plan)), a.media, prefix="mopd_full"):
                print(f)
        for kind, title in KINDS:
            for arm, v in rep["evals"].get(kind, {}).items():
                m = v.get(rep["labels"][-1]) or v.get("final") or v.get("step0")
                if m:
                    print(f"{title:15s} {arm:10s} {m['mean']:.4f} [{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}]")
            for c, d in rep["diffs"].get(kind, {}).items():
                print(f"{title:15s} {c:22s} {100 * d['diff']:+.2f} pp [{100 * d['ci95'][0]:+.2f}, {100 * d['ci95'][1]:+.2f}]")
        return
    rep = gates(a.root, json.load(open(a.plan)), a.spent)
    if a.extra:  # each gate passes only if both the computed part (when there is one) and the extra part pass
        for k, v in json.load(open(a.extra)).items():
            g = rep["gates"].setdefault(k, {"detail": {}, "pass": True})
            g["detail"].update(v.get("detail", {}))
            g["pass"] = g["pass"] and v.get("pass", True)
    txt = json.dumps(rep, indent=1, default=float)
    if a.out:
        open(a.out, "w").write(txt)
    for k, g in rep["gates"].items():
        print(f"{k}: {'PASS' if g['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
