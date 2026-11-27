"""Off-ledger cross-region transfer for the mopdpaper campaign (the guard's ledger sees only TPU time).

Every fresh boot in europe-west4 pulls the compile cache (cache/) and the two teachers from the asia-northeast1
bucket; final_sync writes new cache keys back. Boots are read from the plan's per-boot ledger objects, the cache
size from the bucket, both at the time of the call. Prints the all-in estimate and the ledger allowance that keeps
the campaign under the hard cap:  allowance = cap - transfer so far - projected transfer of the boots left - margin.

    python rl/tpu/transfer_mopdpaper.py --plan rl/tpu/mopdpaper.json [--boots-left N] [--json]
"""
import argparse
import json
import subprocess

USD_GB = 0.08
TEACHERS_GB = 9.1
CONTROL_USD = 1.0


def gs(*a, timeout=300):
    r = subprocess.run(["gcloud", "storage", *a], capture_output=True, text=True, timeout=timeout)
    return r.stdout if r.returncode == 0 else ""


def du_gb(url):
    out = gs("du", "-s", url)
    return int(out.split()[0]) / 1e9 if out.strip() else 0.0


def boots(plan):
    """Distinct VM boots that reached setup: per-boot ledger objects written by each VM's guard."""
    urls = [u for u in gs("ls", plan["ledger_gs"] + "/").split() if u.endswith(".json")]
    return sorted({u.rsplit("/", 1)[-1][:-5] for u in urls})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--cap", type=float, default=320.0, help="all-in hard cap")
    ap.add_argument("--boots-left", type=int, default=None, help="default: 5 planned boots minus boots so far")
    ap.add_argument("--planned-boots", type=int, default=5)
    ap.add_argument("--cache-gb-at-start", type=float, default=None, help="cache size when the campaign started")
    ap.add_argument("--margin", type=float, default=5.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    plan = json.load(open(a.plan))
    bucket = plan["gs_prefix"].rsplit("/", 1)[0]
    cache = du_gb(bucket + "/cache/")
    c0 = a.cache_gb_at_start if a.cache_gb_at_start is not None else cache
    b = boots(plan)
    n = len(b)
    left = a.boots_left if a.boots_left is not None else max(0, a.planned_boots - n)
    pulled = n * ((c0 + cache) / 2 + TEACHERS_GB)
    written = max(0.0, cache - c0)
    so_far = (pulled + written) * USD_GB + CONTROL_USD
    per_boot_next = (cache + TEACHERS_GB) * USD_GB * 1.1
    projected = left * per_boot_next
    allowance = a.cap - so_far - projected - a.margin
    out = {"boots": n, "boot_ids": b, "cache_gb_now": round(cache, 1), "cache_gb_at_start": round(c0, 1),
           "transfer_so_far_usd": round(so_far, 2), "boots_left": left, "per_boot_next_usd": round(per_boot_next, 2),
           "projected_remaining_usd": round(projected, 2), "ledger_allowance_usd": round(allowance, 2),
           "plan_total_cap_usd": plan["total_cap_usd"]}
    print(json.dumps(out, indent=None if not a.json else 1))


if __name__ == "__main__":
    main()
