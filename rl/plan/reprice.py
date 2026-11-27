"""Post-pilot re-price for the $90 plan: measured truncation, finished length, decode speeds and learner speed
from the pilot directory, actual spend from the watchdog state, then cost_model.plan50 for each step count.
Picks the most steps (<= the plan's) whose projected-total median is <= the plan's reprice_median_max_usd
and whose P90 stays within the total cap.

With <pilot>/live_rates.json (written by auto90.sh from the guard's price table, or --rates), the remaining
cost is priced on the machines that are actually live, at their own rates, instead of plan50's flat $5.40:
see live_total(). A <pilot>/published.env (the plan.env pairs may already have claimed) is returned as is, so
every pair of the plan runs with one STEPS.

    python -m plan.reprice --pilot /tmp/qrl-v2/d90/pilot --plan /tmp/qrl-v2/plan.json \
        --watchdog /tmp/qrl-v2/watchdog.json --setups-left 1 --out /tmp/qrl-v2/d90/reprice.json
"""
import argparse
import datetime as dt
import json
import os

import numpy as np

from plan.cost_model import THR_EAGER, plan50

PRICES_DEFAULT = 5.40
HAZARD_H = (0.0, 0.05, 0.15)       # Spot preemptions per VM-hour (none seen yet on v6e Spot; an allowance)
REBOOT_H = 0.1                     # replacement boot + setup (measured ~4 min on as1c/ew4a, 11:22-11:32)
REGRANT_WAIT_H = (0.1, 0.5, 3.0)   # requester race until a replacement is granted (wall only: GKE fee)
TAIL_H = 0.05                      # final sync + self-delete after a machine's last pair


def r_h(q):
    """Hours to reserve for a pair: its P90 plus 15 minutes."""
    return round(q[2] + 0.25, 2)


def measured(pilot, cap):
    m = {}
    prof = json.load(open(os.path.join(pilot, "profile.json")))
    lens = [x for p in prof["prompts"].values() for x in p["lens"]]
    trunc = [t for p in prof["prompts"].values() for t in p["truncated"]]
    fin = [x for x, t in zip(lens, trunc) if not t]
    m["trunc"] = sum(trunc) / len(trunc)
    m["finished_len"] = sum(fin) / max(1, len(fin))
    if prof.get("gen_s"):
        m["eval_tok_s"] = sum(lens) / prof["gen_s"]          # per chip: shards ran concurrently, one per chip
    s = prof["summary"]
    m["informative_prompt_frac"] = s["informative_prompt_frac"]
    m["truncated_loop_frac"] = s["truncated_loop_frac"]
    m["base_reward"] = s["base_reward"]
    # Kept by the filter: 1..n-1 of n solved, plus 20% of never-solved; informative groups per 16 prompts.
    hist = {int(k): v for k, v in s["pass_count_hist"].items()}
    n = max(hist)
    mid = sum(v for k, v in hist.items() if 0 < k < n)
    kept = mid + 0.2 * hist.get(0, 0)
    m["kept_prompts"] = kept
    m["filter"] = kept >= 150
    m["informative"] = 16 * mid / max(1.0, kept if m["filter"] else sum(hist.values()))
    tp = os.path.join(pilot, "timing.json")
    if os.path.exists(tp):
        t = json.load(open(tp))
        m["decode_s_per_step"] = {k: v["s_per_step"] for k, v in t.items()}
        if "128" in t:
            m["t128"] = t["128"]["s_per_step"]
    for f in ("bench_c64_t0.json", "bench_c256_t16384.json"):
        p = os.path.join(pilot, f)
        if os.path.exists(p):
            b = json.load(open(p))
            if b.get("steps"):
                m[f[:-5] + "_tok_s"] = b["steps"][-1]["tok_s"]
    c64, c256 = m.get("bench_c64_t0_tok_s"), m.get("bench_c256_t16384_tok_s")
    m["bench_ratio"] = c256 / c64 if c64 and c256 else 1.0
    # One cap-length sequence per microbatch: the bench packs two (cap - 500)-token sequences into 16384, but two
    # real ones (prompt + 6144, padded to 6656) exhaust a v6e chip's HBM in the training learner at step 0.
    m["learner_flags"] = "--gdn-chunk 256 --mb-tokens 8192" if m["bench_ratio"] >= 1.15 else ""
    used = c256 if m["learner_flags"] else c64
    if used:
        m["learner_speedup"] = used / THR_EAGER
    return m


def spent(watchdog, prices):
    st = json.load(open(watchdog))
    now = dt.datetime.now(dt.timezone.utc)
    tot = 0.0
    for v in st["vms"].values():
        if not v.get("start"):
            continue
        end = dt.datetime.fromisoformat(v["end"].replace("Z", "+00:00")) if v.get("end") else now
        start = dt.datetime.fromisoformat(v["start"].replace("Z", "+00:00"))
        region = v["zone"].rsplit("-", 1)[0]
        tot += max(0.0, (end - start).total_seconds()) / 3600 * prices.get(region, prices.get("default", PRICES_DEFAULT))
    return tot


def live_total(s, spent_usd, rates, seed=1):
    """Projected total (USD and VM-h samples) of the three pairs on the live machines.

    rates: {"live": [{"name", "usd_h", "offset_h"}], "replacement_usd_h": [lo, mode, hi], "fee_usd_h"}.
    offset_h is how long a machine is still busy (billed) before its first pair, e.g. the pilot's base eval.
    A missing second machine is a replacement (REBOOT_H setup at the replacement rate). p1 and p2 (sync+async)
    start on the two machines, p3 (async+async) goes to the first one free; the other deletes itself (TAIL_H).
    Spot allowance: each pair is preempted at most once, with hazard HAZARD_H; pairs do not resume, so the elapsed
    part is lost and the whole pair re-runs on a replacement after REBOOT_H, at a rate drawn from
    replacement_usd_h. The GKE cluster fee runs until the last machine is done."""
    sa, aa = s["pair_sa"], s["pair_aa"]
    n = len(sa)
    rng = np.random.default_rng(seed)
    lam = rng.triangular(*HAZARD_H, n)
    lo, mode, hi = rates["replacement_usd_h"]
    live = sorted(rates["live"], key=lambda v: -v["usd_h"])[:2]
    cost, vmh = np.full(n, float(spent_usd)), np.zeros(n)
    t, rate = np.zeros((2, n)), np.zeros((2, n))
    for i in range(2):
        v = live[i] if i < len(live) else {"usd_h": mode, "offset_h": REBOOT_H}
        rate[i], t[i] = v["usd_h"], v.get("offset_h", 0.0)
        cost += t[i] * rate[i]; vmh += t[i]

    def run(i, d):
        nonlocal cost, vmh
        r_now, t_now = rate[i], t[i]
        pre = rng.random(n) < 1 - np.exp(-lam * d)
        u = rng.random(n) * d
        rr = rng.triangular(lo, mode, hi, n)
        cost = cost + np.where(pre, u * r_now + (REBOOT_H + d) * rr, d * r_now)
        vmh = vmh + np.where(pre, u + REBOOT_H + d, d)
        return np.where(pre, t_now + u + rng.triangular(*REGRANT_WAIT_H, n) + REBOOT_H + d, t_now + d), \
            np.where(pre, rr, r_now)

    for i in range(2):
        t[i], rate[i] = run(i, sa)
    first, idx = (t[1] < t[0]).astype(int), np.arange(n)
    t_other, r_other = t[1 - first, idx].copy(), rate[1 - first, idx].copy()
    t[0], rate[0] = t[first, idx].copy(), rate[first, idx].copy()   # slot 0 := the first-free machine
    t3, r3 = run(0, aa)
    end = np.maximum(t3, t_other) + TAIL_H
    cost += TAIL_H * (r3 + r_other) + rates.get("fee_usd_h", 0.0) * end
    vmh += 2 * TAIL_H
    q = lambda x: [float(np.quantile(x, z)) for z in (0.1, 0.5, 0.9)]  # noqa: E731
    return {"usd": q(cost), "vm_h_left": q(vmh), "wall_h_left": q(end),
            "p_over_80": float(np.mean(cost > 80)), "p_over_90": float(np.mean(cost > 90)),
            "p_preempt_any": float(np.mean(1 - np.exp(-lam * (2 * sa + aa))))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--watchdog", default=None, help="watchdog_v2 state file (or give --spent-usd)")
    ap.add_argument("--spent-usd", type=float, default=None)
    ap.add_argument("--setups-left", type=int, default=1)
    ap.add_argument("--learner-speedup", type=float, default=None)
    ap.add_argument("--rates", default=None, help="live-rates JSON (default <pilot>/live_rates.json when present)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    env_out = a.out[:-5] + ".env"
    pub = os.path.join(a.pilot, "published.env")
    if os.path.exists(pub):
        txt = open(pub).read()
        open(env_out, "w").write(txt)
        json.dump({"published": txt}, open(a.out, "w"), indent=1)
        print(f"using the already-published plan.env ({pub}): {' '.join(txt.split())}")
        return
    pl = json.load(open(a.plan))
    cap, max_steps = pl["cap"], pl["steps"]
    m = measured(a.pilot, cap)
    sp = a.spent_usd if a.spent_usd is not None else spent(a.watchdog, pl.get("price_usd_h", {}))
    ls = a.learner_speedup if a.learner_speedup is not None else m.get("learner_speedup")
    rates_path = a.rates or os.path.join(a.pilot, "live_rates.json")
    rates = json.load(open(rates_path)) if os.path.exists(rates_path) else None
    kw = dict(vms=2, learner_speedup=ls, trunc=m["trunc"], finished_len=m["finished_len"],
              eval_tok_s=m.get("eval_tok_s"), t128=m.get("t128"), informative=m["informative"])
    full = plan50(cap, max_steps, **kw)
    rows = {}
    choice = None
    for steps in range(max_steps, 9, -5):
        r = plan50(cap, steps, spent_usd=sp, setups_left=a.setups_left, samples=rates is not None, **kw)
        if rates is not None:
            r["flat_usd"] = r["usd"]
            r.update(live_total(r.pop("_samples"), sp, rates))
        rows[steps] = r
        if choice is None and r["usd"][1] <= pl["reprice_median_max_usd"] and r["usd"][2] <= pl["total_cap_usd"]:
            choice = steps
    if choice is None:
        choice = min(rows)
    res = {"measured": m, "spent_usd": sp, "setups_left": a.setups_left, "learner_speedup_used": ls,
           "rates": rates, "full_plan_measured": full, "by_steps": rows, "steps": choice,
           "shrunk": choice < max_steps}
    json.dump(res, open(a.out, "w"), indent=1)
    r = rows[choice]
    with open(env_out, "w") as f:
        f.write(f"STEPS={choice}\nTRAIN_FILTER={int(m['filter'])}\n"
                f"LEARNER_FLAGS='{m['learner_flags']}'\nBENCH_RATIO={m['bench_ratio']:.2f}\n"
                f"PAIR_SA_H={r_h(rows[choice]['pair_sync_async_h'])}\nPAIR_AA_H={r_h(rows[choice]['pair_async_async_h'])}\n")
    print(f"steps {choice} (plan {max_steps}); projected total ${r['usd'][1]:.1f} "
          f"(P10-P90 ${r['usd'][0]:.1f}-{r['usd'][2]:.1f}); spent ${sp:.2f}; trunc {m['trunc']:.2f}; "
          f"finished_len {m['finished_len']:.0f}; t128 {m.get('t128')}; learner x{ls}; "
          f"rates {'live ' + json.dumps([(v['name'], v['usd_h']) for v in rates['live']]) if rates else 'flat $%.2f' % r['price_vm_h']}")


if __name__ == "__main__":
    main()
