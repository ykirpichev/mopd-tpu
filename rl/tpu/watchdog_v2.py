"""Per-run spend watchdog for the v2 runs. Every minute it lists qrl-bc4a VMs, accrues billed time from
each VM's lastStartTimestamp (Flex-start bills from RUNNING), and keeps a state file so deleted VMs still
count. When a run's spend plus 15 minutes of its live VMs reaches that run's cap, it deletes every VM of
that run (results are already in GCS; remote_v2.sh rsyncs every 5 minutes).

    python rl/tpu/watchdog_v2.py --cap run1=100 --cap run2=100 --state /tmp/qrl-v2/watchdog.json

--total-cap prices every qrl-bc4a VM together (labels ignored) and deletes all of them at the cap;
--max-vms deletes the newest RUNNING VMs beyond that count as soon as they are seen.

--plan (default /tmp/qrl-v2/plan.json, when it exists) takes the total cap, max VMs, allowlist and per-region
prices from the plan file; any of --total-cap/--max-vms/--allow that disagrees with it makes the watchdog
refuse to start, and it exits if the plan file changes under it. Only one watchdog holds the lock at a time.

    python rl/tpu/watchdog_v2.py --plan /tmp/qrl-v2/plan.json --state /tmp/qrl-v2/watchdog.json
"""
import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import time

PRICE = 5.40
PRICES = {}
DEFAULT_PLAN = "/tmp/qrl-v2/plan.json"


def price(zone):
    region = zone.rsplit("-", 1)[0]
    return PRICES.get(region, PRICES.get("default", PRICE)) if PRICES else PRICE


def load_plan(a):
    """Fold the plan file into the args; return an error string on any disagreement."""
    path = a.plan or (DEFAULT_PLAN if os.path.exists(DEFAULT_PLAN) else None)
    if not path:
        return None
    pl = json.load(open(path))
    if pl.get("plan") == "plan90":
        return f"{path} is plan90, which is enforced by rl/tpu/guard90.py (owner {pl['owner']})"
    allow = sorted(n for n, _ in pl["slots"])
    if a.total_cap is not None and a.total_cap != pl["total_cap_usd"]:
        return f"--total-cap {a.total_cap} != plan {pl['total_cap_usd']} ({path}, owner {pl['owner']})"
    if a.max_vms is not None and a.max_vms != pl["max_vms"]:
        return f"--max-vms {a.max_vms} != plan {pl['max_vms']} ({path}, owner {pl['owner']})"
    if a.allow is not None and sorted(a.allow.split(",")) != allow:
        return f"--allow {a.allow} != plan {','.join(allow)} ({path}, owner {pl['owner']})"
    if a.cap:
        return f"--cap is not allowed with a plan file ({path})"
    a.total_cap, a.max_vms, a.allow = float(pl["total_cap_usd"]), int(pl["max_vms"]), ",".join(allow)
    PRICES.clear()
    PRICES.update(pl.get("price_usd_h", {}))
    a.plan, a.plan_text = path, open(path).read()
    return None
PROJECT = "tpu-builders-509006"
GCLOUD = os.path.expanduser("~/google-cloud-sdk/bin/gcloud")
TPU_ZONES = ("asia-northeast1-b", "us-east5-a", "us-east5-b")


def now():
    return dt.datetime.now(dt.timezone.utc)


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def list_vms():
    out = subprocess.run([GCLOUD, "compute", "instances", "list", f"--project={PROJECT}",
                          "--filter=labels.owner=qrl-bc4a AND name~^qrl-bc4a-", "--format=json"],
                         capture_output=True, text=True, timeout=120)
    if out.returncode:
        raise RuntimeError(out.stderr[-300:])
    vms = json.loads(out.stdout)
    for z in TPU_ZONES:
        out = subprocess.run([GCLOUD, "compute", "tpus", "tpu-vm", "list", f"--zone={z}", f"--project={PROJECT}",
                              "--filter=labels.owner=qrl-bc4a", "--format=json"], capture_output=True, text=True, timeout=120)
        if out.returncode:
            raise RuntimeError(out.stderr[-300:])
        for n in json.loads(out.stdout):
            vms.append({"name": n["name"].rsplit("/", 1)[-1], "zone": z, "labels": n.get("labels", {}), "kind": "tpu",
                        "status": "RUNNING" if n.get("state") in ("READY", "CREATING", "STARTING", "REPAIRING") else n.get("state", "?"),
                        "lastStartTimestamp": n.get("createTime")})
    return vms


def delete(name, rec):
    if rec.get("kind") == "tpu":
        cmd = [GCLOUD, "compute", "tpus", "queued-resources", "delete", name, f"--zone={rec['zone']}", "--force"]
    else:
        cmd = [GCLOUD, "compute", "instances", "delete", name, f"--zone={rec['zone']}"]
    subprocess.run(cmd + [f"--project={PROJECT}", "--quiet"], timeout=900)


def spend(state, t):
    per = {}
    for v in state["vms"].values():
        if not v.get("start"):
            continue
        end = ts(v["end"]) if v.get("end") else t
        per[v["run"]] = per.get(v["run"], 0.0) + max(0.0, (end - ts(v["start"])).total_seconds()) / 3600 * price(v["zone"])
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", action="append", default=[], help="run=usd")
    ap.add_argument("--total-cap", type=float, default=None, help="USD across all qrl-bc4a VMs")
    ap.add_argument("--max-vms", type=int, default=None)
    ap.add_argument("--allow", default=None, help="comma list; any other RUNNING qrl-bc4a VM is deleted at once")
    ap.add_argument("--state", required=True)
    ap.add_argument("--warn-frac", type=float, default=0.85)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--lock", default="/tmp/qrl-v2/watchdog.lock")
    a = ap.parse_args()
    err = load_plan(a)
    if err:
        print(f"{now():%FT%TZ} REFUSING to run: {err}", flush=True)
        sys.exit(2)
    lock = open(a.lock, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"{now():%FT%TZ} REFUSING to run: another watchdog holds {a.lock}", flush=True)
        sys.exit(3)
    caps = {k: float(v) for k, v in (c.split("=") for c in a.cap)}
    state = json.load(open(a.state)) if os.path.exists(a.state) else {"vms": {}, "stopped": [], "warned": []}
    print(f"{now():%FT%TZ} watchdog start: total cap ${a.total_cap}, max VMs {a.max_vms}, allow {a.allow}, "
          f"plan {getattr(a, 'plan', None)}, prices {PRICES or PRICE}", flush=True)
    while True:
        if getattr(a, "plan_text", None) is not None and open(a.plan).read() != a.plan_text:
            print(f"{now():%FT%TZ} plan file {a.plan} changed; exiting so the wrapper restarts with it", flush=True)
            sys.exit(4)
        t = now()
        try:
            vms = list_vms()
        except Exception as e:  # noqa: BLE001
            print(f"{t:%FT%TZ} list failed: {e}", flush=True)
            time.sleep(60)
            continue
        alive = {}
        for v in vms:
            run = v.get("labels", {}).get("run", "?")
            rec = state["vms"].setdefault(v["name"], {"run": run, "zone": v["zone"].rsplit("/", 1)[-1]})
            if v["status"] == "RUNNING" and v.get("lastStartTimestamp") and not rec.get("start"):
                rec["start"] = v["lastStartTimestamp"]
            rec["run"] = run
            rec["kind"] = v.get("kind", "gce")
            rec["status"] = v["status"]
            rec.pop("end", None)
            if v["status"] in ("RUNNING", "STOPPING"):
                alive.setdefault(run, []).append(v["name"])
        seen = {v["name"] for v in vms}
        for name, rec in state["vms"].items():
            if name not in seen and rec.get("start") and not rec.get("end"):
                rec["end"] = t.isoformat()
                rec["status"] = "GONE"
        if a.allow is not None:
            allowed = set(a.allow.split(","))
            for v in vms:
                if v["status"] == "RUNNING" and v["name"] not in allowed:
                    print(f"{t:%FT%TZ} NOT ALLOWED: deleting {v['name']}", flush=True)
                    delete(v["name"], state["vms"][v["name"]])
                    v["status"] = "DELETED"
        if a.max_vms is not None:
            live = sorted((state["vms"][v["name"]].get("start") or "", v["name"]) for v in vms
                          if v["status"] == "RUNNING")
            for _, name in live[a.max_vms:]:
                print(f"{t:%FT%TZ} EXTRA VM beyond {a.max_vms}: deleting {name}", flush=True)
                delete(name, state["vms"][name])
        per = spend(state, t)
        if a.total_cap is not None:
            caps = {"ALL": a.total_cap}
            per = {"ALL": sum(per.values())}
            alive = {"ALL": [n for r in alive.values() for n in r]}
        for run, cap in caps.items():
            s = per.get(run, 0.0)
            proj = s + sum(price(state["vms"][n]["zone"]) for n in alive.get(run, [])) / 4
            line = f"{t:%FT%TZ} {run}: spent ${s:.2f} of ${cap:.0f}, live {alive.get(run, [])}"
            print(line, flush=True)
            if proj >= a.warn_frac * cap and run not in state["warned"]:
                state["warned"].append(run)
                open(a.state + f".{run}.WARN", "w").write(line + "\n")
            if proj >= cap and alive.get(run):
                print(f"{t:%FT%TZ} {run}: CAP REACHED, deleting {alive[run]}", flush=True)
                open(a.state + f".{run}.CAP", "w").write(line + "\n")
                for name in alive[run]:
                    delete(name, state["vms"][name])
                state["stopped"].append(run)
        json.dump(state, open(a.state + ".tmp", "w"), indent=1)
        os.replace(a.state + ".tmp", a.state)
        if a.once:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
