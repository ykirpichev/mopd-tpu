"""Spend guard for the $90 plan. Runs on the agent machine AND on every qrl-bc4a TPU VM (--self NAME), because
the agent machine can be paused for long stretches (cloud-agent VM suspend), and then nothing local runs.

Every minute it lists the qrl-bc4a instances, writes one ledger record per RUNNING VM boot (name@lastStart,
last_seen) to GCS, reads all records back, and prices them (a boot that is gone is billed up to its last_seen
plus 2 minutes). It enforces, from the plan file alone:
  - total cap: when spend + 15 min of the live VMs reaches total_cap_usd, delete every qrl-bc4a VM
    (on a VM: the others first, then itself) and write a CAP marker to GCS;
  - max_vms: delete the newest RUNNING VMs beyond it, and every PENDING request once max_vms are granted;
  - allowlist: delete any granted qrl-bc4a VM whose name is not a plan slot.
Logs go straight to --log (no pipes), a heartbeat file is touched every loop, every gcloud call has a timeout,
and only one guard per machine holds --lock.

    python3 rl/tpu/guard90.py --plan /tmp/qrl-v2/plan.json --state /tmp/qrl-v2/guard.json \
        --log /tmp/qrl-v2/watchdog.log --heartbeat /tmp/qrl-v2/guard.hb
    python3 rl/tpu/guard90.py --plan plan.json --print-spend
"""
import argparse
import datetime as dt
import fcntl
import json
import re
import os
import shutil
import subprocess
import sys
import tempfile
import time

PROJECT = "tpu-builders-509006"
GCLOUD = shutil.which("gcloud") or os.path.expanduser("~/google-cloud-sdk/bin/gcloud")
TPU_ZONES = ("asia-northeast1-b", "us-east5-a", "us-east5-b", "europe-west4-a", "us-central1-a")
GRANTED = ("PROVISIONING", "STAGING", "RUNNING")
CONFIRMED = ("STAGING", "RUNNING")  # only these count toward max_vms or may cancel other requests
LOG = None
PLAN_PATH = None
NO_ENFORCE = False
# CPU dry-run VMs: priced in the ledger, never enforced on or counted toward max_vms. Outside the qrl-bc4a- prefix
# on purpose: guards from older code on live TPU VMs would delete them as non-allowlisted.
DRY_PREFIX = "qrl-dry-bc4a-"
WASTE = [-1e9, 0, 0.0]
GKE_CANCEL = [-1e9]


def now():
    return dt.datetime.now(dt.timezone.utc)


def ts(s):  # also gcloud storage's "+0000" offsets, which Python 3.10's fromisoformat rejects
    s = re.sub(r"([+-]\d\d)(\d\d)$", r"\1:\2", s.strip().replace("Z", "+00:00"))
    return dt.datetime.fromisoformat(s)


def log(msg):
    line = f"{now():%FT%TZ} {msg}"
    if LOG:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    else:
        try:
            print(line, flush=True)
        except BrokenPipeError:
            pass


def gc(args, timeout=120, stdin=None):
    try:
        return subprocess.run([GCLOUD, *args], capture_output=True, text=True, timeout=timeout, input=stdin)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", f"timeout after {timeout}s")


def load_plan(path):
    pl = json.load(open(path))
    if pl.get("plan") not in KNOWN_PLANS:
        raise SystemExit(f"REFUSING: {path} is not a known plan {KNOWN_PLANS}")
    return pl


KNOWN_PLANS = ("plan90", "plan25k1", "plan65k1long", "plan55sync60", "mopdpilot", "mopdfull", "mopdpaper")
PRICE_KEYS = {"FLEX_START": "price_usd_h", "SPOT": "price_spot_usd_h"}


def price(pl, zone, model="FLEX_START"):
    """USD/h of one ct6e-standard-4t by (zone, provisioning model). Unknown models (on-demand) get the on-demand
    price so a stray STANDARD VM is never under-counted."""
    if model == "GKE_CLUSTER":
        return float(pl["gke"]["fee_usd_h"])
    if model == "DRY_CPU":
        return float(pl.get("price_dry_usd_h", 0.1))
    p = pl.get(PRICE_KEYS.get(model, ""), None) or pl.get("price_ondemand_usd_h") or {"default": 10.8}
    return p.get(zone.rsplit("-", 1)[0], p["default"])


def slot_names(pl):
    return {s[0] for s in pl["slots"]}


def gke_prefix(pl):
    return f"gke-{pl['gke']['cluster']}-tpu-" if pl.get("gke") else None


def list_gke(pl):
    """GKE TPU nodes (they count toward max_vms) plus the cluster itself as a 'boot' priced at the fee."""
    if not pl.get("gke"):
        return []
    g = pl["gke"]
    out = gc(["compute", "instances", "list", f"--project={PROJECT}", f"--filter=name~^{gke_prefix(pl)}", "--format=json"])
    if out.returncode:
        raise RuntimeError(out.stderr[-300:])
    vms = [{"name": v["name"], "zone": v["zone"].rsplit("/", 1)[-1], "status": v["status"],
            "start": v.get("lastStartTimestamp"), "kind": "gke", "model": model_of(v)} for v in json.loads(out.stdout)]
    out = gc(["container", "clusters", "list", f"--project={PROJECT}", f"--filter=name={g['cluster']}", "--format=json"])
    if out.returncode:
        raise RuntimeError(out.stderr[-300:])
    for c in json.loads(out.stdout or "[]"):
        vms.append({"name": c["name"], "zone": g["location"], "status": "CLUSTER", "start": c["createTime"],
                    "kind": "cluster", "model": "GKE_CLUSTER", "cstatus": c["status"]})
    return vms


def gke_cmd(pl, *args):
    if NO_ENFORCE and args and args[0] != "status":
        log(f"NO-ENFORCE (dry VM guard): would run gke90 {' '.join(args)}")
        return True, "no-enforce"
    here = os.path.dirname(os.path.abspath(__file__))
    plan = PLAN_PATH or os.path.expanduser("~/plan.json")
    try:
        r = subprocess.run([sys.executable, os.path.join(here, "gke90.py"), "--plan", plan, *args],
                           capture_output=True, text=True, timeout=420)
        return r.returncode == 0, (r.stdout + r.stderr)[-300:]
    except subprocess.TimeoutExpired:
        return False, "gke90 timeout"


def model_of(v):
    return v.get("scheduling", {}).get("provisioningModel", "STANDARD")


def list_vms(tpus):
    raw = []
    for flt in ("name~^qrl-bc4a-", f"name~^{DRY_PREFIX}"):
        out = gc(["compute", "instances", "list", f"--project={PROJECT}", f"--filter={flt}", "--format=json"])
        if out.returncode:
            raise RuntimeError(out.stderr[-300:])
        raw += json.loads(out.stdout)
    vms = [{"name": v["name"], "zone": v["zone"].rsplit("/", 1)[-1], "status": v["status"],
            "start": v.get("lastStartTimestamp"), "kind": "dry" if v["name"].startswith(DRY_PREFIX) else "gce",
            "model": "DRY_CPU" if v["name"].startswith(DRY_PREFIX) else model_of(v)} for v in raw]
    for z in TPU_ZONES if tpus else ():
        out = gc(["compute", "tpus", "tpu-vm", "list", f"--zone={z}", f"--project={PROJECT}", "--format=json"])
        if out.returncode:
            continue
        for n in json.loads(out.stdout or "[]"):
            name = n["name"].rsplit("/", 1)[-1]
            if name.startswith("qrl-bc4a-"):
                vms.append({"name": name, "zone": z, "status": "RUNNING", "start": n.get("createTime"), "kind": "tpu",
                             "model": "SPOT" if n.get("schedulingConfig", {}).get("spot") else "FLEX_START"})
    return vms


def delete(v, pl=None):
    if NO_ENFORCE:
        log(f"NO-ENFORCE (dry VM guard): would delete {v['name']} ({v['zone']}, {v['status']})")
        return
    log(f"deleting {v['name']} ({v['zone']}, {v['status']})")
    if v["kind"] in ("gke", "cluster"):
        ok, msg = gke_cmd(pl, *(["kill-node", v["name"]] if v["kind"] == "gke" else ["delete-cluster"]))
        if not ok:
            log(f"delete {v['name']} failed: {msg}")
        return
    if v["kind"] == "tpu":
        cmd = ["compute", "tpus", "queued-resources", "delete", v["name"], f"--zone={v['zone']}", "--force"]
    else:
        cmd = ["compute", "instances", "delete", v["name"], f"--zone={v['zone']}"]
    r = gc(cmd + [f"--project={PROJECT}", "--quiet"], timeout=300)
    if r.returncode:
        log(f"delete {v['name']} failed: {r.stderr[-200:]}")


def delete_all(vs, pl=None):
    """Delete in parallel, so a pass never acts minutes after its snapshot."""
    import threading
    th = [threading.Thread(target=delete, args=(v, pl)) for v in vs]
    for x in th:
        x.start()
    for x in th:
        x.join(330)


def confirmed_now(pl):
    """Fresh count of STAGING/RUNNING machines (qrl-bc4a VMs + GKE TPU nodes); None when the list fails."""
    try:
        return sum(v["status"] in CONFIRMED for v in list_vms(False) + list_gke(pl) if v["kind"] in ("gce", "tpu", "gke"))
    except Exception:  # noqa: BLE001
        return None


def key(v):
    return f"{v['name']}@{v['start']}"


def gs_put(url, text):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(text)
    r = gc(["storage", "cp", f.name, url, "--quiet"], timeout=60)
    os.unlink(f.name)
    return r.returncode == 0


def ledger_sync(pl, state, live, t):
    """Write this boot's records, read everyone's; state['boots'] ends up as the union (max last_seen)."""
    base = pl["ledger_gs"].rstrip("/")
    for v in live:
        rec = state["boots"].setdefault(key(v), {"name": v["name"], "zone": v["zone"], "start": v["start"]})
        rec["model"] = v.get("model", "FLEX_START")
        rec["last_seen"] = t.isoformat()
        safe = key(v).replace(":", "").replace("@", "__").replace("+", "p")
        gs_put(f"{base}/{safe}.json", json.dumps(rec))
    s = read_records(base)
    if s is not None:
        dec = json.JSONDecoder()
        i = 0
        while i < len(s):
            while i < len(s) and s[i].isspace():
                i += 1
            if i >= len(s):
                break
            r, i = dec.raw_decode(s, i)
            k = f"{r['name']}@{r['start']}"
            mine = state["boots"].setdefault(k, r)
            if ts(r["last_seen"]) > ts(mine["last_seen"]):
                mine["last_seen"] = r["last_seen"]
            if "model" in r:
                mine["model"] = r["model"]
        return True
    return False


def read_records(base):
    """Concatenated <base>/*.json, or None. A wildcard cat 404s when another guard rewrites a record between the
    listing and the read (common with several guards writing every minute), so retry, then read object by object."""
    for _ in range(3):
        out = gc(["storage", "cat", f"{base}/*.json"])
        if out.returncode == 0:
            return out.stdout
        if "matched no objects" in out.stderr:
            return ""
        time.sleep(2)
    ls = gc(["storage", "ls", f"{base}/*.json"])
    if ls.returncode:
        return "" if "matched no objects" in ls.stderr else None
    parts = []
    for u in ls.stdout.split():
        for _ in range(3):
            r = gc(["storage", "cat", u], 60)
            if r.returncode == 0:
                parts.append(r.stdout)
                break
        else:
            return None
    return "\n".join(parts)


def spend(pl, state, live_keys, t):
    tot = 0.0
    for k, r in state["boots"].items():
        end = t if k in live_keys else min(t, ts(r["last_seen"]) + dt.timedelta(minutes=2))
        tot += max(0.0, (end - ts(r["start"])).total_seconds()) / 3600 * price(pl, r["zone"], r.get("model", "FLEX_START"))
    return tot


def waste(pl):
    """Spot preemption losses that auto90.sh records under <ledger>/waste/ when it takes over a dead boot's claim.
    Informational: those hours are already billed through the dead boot's ledger record."""
    out = gc(["storage", "cat", f"{pl['ledger_gs'].rstrip('/')}/waste/*.json"])
    if out.returncode:
        return 0, 0.0
    n, usd = 0, 0.0
    for line in out.stdout.splitlines():
        try:
            usd += float(json.loads(line).get("usd", 0)); n += 1
        except ValueError:
            pass
    return n, usd


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--state", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--heartbeat", default=None)
    ap.add_argument("--lock", default=None)
    ap.add_argument("--self", dest="me", default=None, help="this VM's name (VM mode: delete itself last)")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--print-spend", action="store_true")
    ap.add_argument("--no-enforce", action="store_true", help="dry-run VM: log deletions instead of issuing them")
    a = ap.parse_args()
    global PLAN_PATH
    global NO_ENFORCE
    LOG, PLAN_PATH, NO_ENFORCE = a.log, os.path.abspath(a.plan), a.no_enforce
    pl = load_plan(a.plan)
    plan_text = open(a.plan).read()
    if a.print_spend:
        state = {"boots": {}}
        vms = list_vms(False) + list_gke(pl)
        live = [v for v in vms if (v["status"] == "RUNNING" or v["kind"] == "cluster") and v["start"]]
        ledger_sync(pl, state, [], now())
        print(f"{spend(pl, state, {key(v) for v in live}, now()):.2f}")
        return
    if a.lock:
        lock = open(a.lock, "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"REFUSING to run: another guard holds {a.lock}")
            sys.exit(3)
    state = json.load(open(a.state)) if a.state and os.path.exists(a.state) else {"boots": {}}
    state.setdefault("boots", {})
    cap, maxv = float(pl["total_cap_usd"]), int(pl["max_vms"])
    allow = slot_names(pl)
    models = set(pl.get("allowed_models", ["FLEX_START", "SPOT"]))
    gs_cap = pl["ledger_gs"].rstrip("/").rsplit("/", 1)[0] + "/CAP_REACHED"
    log(f"guard start{' on ' + a.me if a.me else ''}: total cap ${cap:.0f}, max VMs {maxv}, allow {sorted(allow)}, "
        f"ledger {pl['ledger_gs']}, owner {pl['owner']}")
    while True:
        t0 = time.monotonic()
        if open(a.plan).read() != plan_text:
            log(f"plan file {a.plan} changed; exiting so the supervisor restarts with it")
            sys.exit(4)
        t = now()
        try:
            vms = list_vms(a.me is None) + list_gke(pl)
        except Exception as e:  # noqa: BLE001
            log(f"list failed: {e}")
            vms = None
        if vms is not None:
            for v in vms:
                if v["kind"] in ("gke", "cluster", "dry"):  # list_gke only returns our cluster and its TPU pools' nodes
                    pass
                elif v["status"] in GRANTED + ("PENDING",) and v["name"] not in allow:
                    log(f"NOT ALLOWED: {v['name']}")
                    delete(v)
                    v["status"] = "DELETED"
                elif v["status"] in GRANTED + ("PENDING",) and v.get("model", "FLEX_START") not in models:
                    log(f"NOT ALLOWED: {v['name']} has provisioning model {v.get('model')} (on-demand is never allowed)")
                    delete(v)
                    v["status"] = "DELETED"
            max_run_s = float(pl.get("gke", {}).get("pod_max_run_s", 43200))
            for v in vms:  # GKE Spot pools can't carry a max-run (MIGs refuse termination DELETE): enforce it here
                if v["kind"] == "gke" and v["status"] in GRANTED and v["start"] \
                        and (t - ts(v["start"])).total_seconds() > max_run_s + 600:
                    log(f"MAX-RUN: GKE node {v['name']} up {(t - ts(v['start'])).total_seconds() / 3600:.1f} h")
                    delete(v)
                    v["status"] = "DELETED"
            granted = sorted((v for v in vms if v["status"] in CONFIRMED and v["kind"] != "dry"), key=lambda v: (v["start"] or "9", v["name"]))
            for v in granted[maxv:]:
                if v["name"] != a.me:
                    log(f"EXTRA VM beyond {maxv}: {v['name']}")
                    delete(v)
                    v["status"] = "DELETED"
            if a.me and a.me in [v["name"] for v in granted[maxv:]]:
                log(f"EXTRA VM beyond {maxv}: myself")
                delete(next(v for v in granted if v["name"] == a.me))
            n_now = confirmed_now(pl) if len(granted) >= maxv else None
            if len(granted) >= maxv and (n_now is None or n_now < maxv):
                log(f"not cancelling other requests: re-check shows {n_now} confirmed (snapshot said {len(granted)})")
            elif len(granted) >= maxv:
                cl = [v for v in vms if v["kind"] == "cluster" and v.get("cstatus") == "RUNNING"]
                if cl and not any(v["kind"] == "gke" for v in granted[:maxv]) and time.monotonic() - GKE_CANCEL[0] > 600:
                    GKE_CANCEL[0] = time.monotonic()
                    log(f"{n_now} confirmed RUNNING/STAGING (max {maxv}): cancelling the GKE request (job + TPU pools to 0)")
                    for cmd in (["delete-job"], ["scale0"]):
                        ok, msg = gke_cmd(pl, *cmd)
                        ok or log(f"gke {cmd[0]} failed: {msg}")
                drop = [v for v in vms if v["status"] in ("PENDING", "PROVISIONING") and v["kind"] == "gce"]
                if drop:
                    log(f"{n_now} confirmed RUNNING/STAGING (max {maxv}): dropping {[v['name'] for v in drop]}")
                    delete_all(drop, pl)
                    for v in drop:
                        v["status"] = "DELETED"
            live = [v for v in vms if (v["status"] == "RUNNING" or v["kind"] == "cluster") and v["start"]]
            ok = ledger_sync(pl, state, live, t)
            live_keys = {key(v) for v in live}
            s = spend(pl, state, live_keys, t)
            proj = s + sum(price(pl, v["zone"], v.get("model", "FLEX_START")) for v in live) / 4
            if time.monotonic() - WASTE[0] > 600:
                WASTE[:] = [time.monotonic(), *waste(pl)]
            log(f"ALL: spent ${s:.2f} of ${cap:.0f}, live {[v['name'] + '/' + v.get('model', '?') for v in live]}, "
                f"pending {[v['name'] for v in vms if v['status'] == 'PENDING']}"
                f"{f', spot preemptions {WASTE[1]} (${WASTE[2]:.2f} lost, included)' if WASTE[1] else ''}"
                f"{'' if ok else ' (ledger read FAILED)'}")
            if proj >= cap:
                log(f"CAP REACHED (projected ${proj:.2f}): deleting every qrl-bc4a VM")
                gs_put(gs_cap, json.dumps({"at": t.isoformat(), "spent": s, "by": a.me or "agent"}))
                if a.state:
                    open(a.state + ".CAP", "w").write(f"{t.isoformat()} spent {s:.2f}\n")
                rest = [v for v in vms if v["status"] in GRANTED + ("PENDING",) or v["kind"] == "cluster"]
                for v in sorted(rest, key=lambda v: (v["name"] == a.me, v["kind"] == "cluster")):
                    delete(v)
            if a.state:
                json.dump(state, open(a.state + ".tmp", "w"), indent=1)
                os.replace(a.state + ".tmp", a.state)
            # Heartbeats only after a pass that could see the VMs: a guard that can't enforce goes stale.
            if a.heartbeat:
                with open(a.heartbeat, "w") as f:
                    f.write(t.isoformat() + "\n")
            if a.me:
                gs_put(f"{pl['ledger_gs'].rstrip('/').rsplit('/', 1)[0]}/hb/{a.me}",
                       json.dumps({"at": t.isoformat(), "spent": round(s, 2), "live": [v["name"] for v in live]}))
        if a.once:
            break
        time.sleep(max(5, a.interval - (time.monotonic() - t0)))


if __name__ == "__main__":
    main()
