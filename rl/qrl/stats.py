"""Confidence intervals for sync-vs-async comparisons.

pass@1 difference at a matched step: hierarchical bootstrap that resamples (1) problems, shared by both
modes so the comparison stays paired, (2) seeds within each mode, and (3) samples within each
problem x run cell (binomial redraw). Equivalence: two one-sided tests at margin delta via the 90% CI.
Run-level metrics (s/step, idle fractions): bootstrap over seeds, ratio CI for the speedup.
"""
import json

import numpy as np


def load_cells(eval_paths_by_mode, label, n_samples=None):
    """{mode: [eval.json or "eval.json::label", ...one per seed]} ->
    problem ids, {mode: (correct [seeds, problems], samples per problem)}."""
    ids, cells = None, {}
    for mode, paths in eval_paths_by_mode.items():
        rows, n = [], n_samples
        for p in paths:
            path, lab = (p.split("::", 1) + [label])[:2] if "::" in p else (p, label)
            ev = json.load(open(path))[lab]
            per = {q["id"]: q["correct"] for q in ev["per_problem"]}
            ids = ids or sorted(per)
            rows.append([per[i] for i in ids])
            if n is None and ev["per_problem"] and "lens" in ev["per_problem"][0]:
                n = len(ev["per_problem"][0]["lens"])
        if n is None:
            raise ValueError("samples per problem unknown: pass n_samples")
        cells[mode] = (np.array(rows, float), n)
    return ids, cells


def bootstrap_diff(correct_a, correct_b, n_samples, iters=10000, seed=0, resample_samples=True):
    """correct_*: [seeds, problems] correct counts out of n_samples. Returns mean diff (a-b) and draws."""
    rng = np.random.default_rng(seed)
    pa, pb = correct_a / n_samples, correct_b / n_samples
    sa, P = pa.shape
    sb = pb.shape[0]
    draws = np.empty(iters)
    for t in range(iters):
        prob = rng.integers(0, P, P)
        ra = pa[rng.integers(0, sa, sa)][:, prob]
        rb = pb[rng.integers(0, sb, sb)][:, prob]
        if resample_samples:
            ra = rng.binomial(n_samples, ra) / n_samples
            rb = rng.binomial(n_samples, rb) / n_samples
        draws[t] = ra.mean() - rb.mean()
    return float(pa.mean() - pb.mean()), draws


def ci(draws, level=0.95):
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def equivalence(draws, margin):
    """TOST at alpha=0.05: equivalent iff the 90% CI lies inside (-margin, +margin)."""
    lo, hi = ci(draws, 0.90)
    return {"ci90": (lo, hi), "margin": margin, "equivalent": bool(-margin < lo and hi < margin)}


def compare(eval_paths_by_mode, label, a="async", b="sync", margin=0.02, iters=10000, seed=0, n_samples=None):
    _, cells = load_cells(eval_paths_by_mode, label, n_samples)
    (ca, n), (cb, _) = cells[a], cells[b]
    d, draws = bootstrap_diff(ca, cb, n, iters=iters, seed=seed)
    return {"label": label, "diff": d, "ci95": ci(draws), **equivalence(draws, margin),
            f"pass@1_{a}": float((ca / n).mean()), f"pass@1_{b}": float((cb / n).mean()),
            "seeds": {a: int(ca.shape[0]), b: int(cb.shape[0])}, "problems": int(ca.shape[1]), "samples": n}


def seed_ratio_ci(values_a, values_b, iters=10000, seed=0, level=0.95):
    """CI for mean(b)/mean(a), e.g. sync s/step over async s/step = speedup, resampling seeds."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    draws = np.array([b[rng.integers(0, len(b), len(b))].mean() / a[rng.integers(0, len(a), len(a))].mean()
                      for _ in range(iters)])
    return float(b.mean() / a.mean()), ci(draws, level)


def simulate_halfwidth(problem_rates, n_problems, n_samples, n_seeds, seed_logit_sd, run_logit_sd=0.0,
                       iters=2000, reps=40, seed=0, return_coverage=False):
    """Expected 95% CI half-width of the async-sync pass@1 difference under the null (no true effect).
    problem_rates: per-problem pass rates to draw from; seed_logit_sd: run-to-run noise per problem;
    run_logit_sd: run-level shift shared by all problems of a run (a seed that trains better overall)."""
    rng = np.random.default_rng(seed)
    rates = np.clip(np.asarray(problem_rates, float), 1e-3, 1 - 1e-3)
    widths, cover = [], 0
    for _ in range(reps):
        p = rng.choice(rates, n_problems)
        logit = np.log(p / (1 - p))

        def runs():
            z = logit + rng.normal(0, run_logit_sd, (n_seeds, 1)) + rng.normal(0, seed_logit_sd, (n_seeds, n_problems))
            return rng.binomial(n_samples, 1 / (1 + np.exp(-z))).astype(float)
        _, draws = bootstrap_diff(runs(), runs(), n_samples, iters=iters, seed=int(rng.integers(1 << 30)))
        lo, hi = ci(draws)
        widths.append((hi - lo) / 2)
        cover += lo <= 0 <= hi
    return (float(np.mean(widths)), cover / reps) if return_coverage else float(np.mean(widths))


def estimate_seed_sd(correct_runs, n_samples):
    """Method-of-moments run-to-run logit sd per problem from >= 2 runs of the same condition,
    after removing binomial noise. correct_runs: [runs, problems]."""
    p = np.clip(correct_runs / n_samples, 0.5 / n_samples, 1 - 0.5 / n_samples)
    var_obs = p.var(axis=0, ddof=1)
    pbar = p.mean(axis=0)
    var_binom = pbar * (1 - pbar) / n_samples
    excess = np.clip(var_obs - var_binom, 0, None)
    slope = pbar * (1 - pbar)  # d p / d logit
    ok = slope > 0.02
    return float(np.sqrt(np.mean(excess[ok] / slope[ok] ** 2))) if ok.any() else 0.0


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="async-vs-sync pass@1 CI at a matched checkpoint label")
    ap.add_argument("--a", nargs="+", required=True, help="eval JSONs for mode a (one per seed)")
    ap.add_argument("--b", nargs="+", required=True, help="eval JSONs for mode b (one per seed)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--samples", type=int, default=None, help="samples per problem if the JSONs lack 'lens'")
    args = ap.parse_args(argv)
    print(json.dumps(compare({"async": args.a, "sync": args.b}, args.label, margin=args.margin,
                             n_samples=args.samples), indent=1))


if __name__ == "__main__":
    main()
