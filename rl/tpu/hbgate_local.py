"""Agent-machine heartbeat gate for plan90. Prints the granted qrl-bc4a VMs that must be deleted because NOTHING
is guarding them: the local guard heartbeat is stale AND the VM's own guard has not checked in to GCS
(<plan90>/hb/<vm>, judged by the object's server-side update time) within STALE_S, counting a VM's boot grace
from its lastStartTimestamp. A fresh VM-side heartbeat always wins, so a paused or restarted agent machine
never deletes healthy VMs; the VM-side guard is the primary enforcer.

    python rl/tpu/hbgate_local.py --vms list.json --local-hb /tmp/qrl-v2/guard.hb \
        --hb-gs gs://.../plan90/hb [--vm-hb-age NAME=SECONDS ...]      # prints "name zone reason" lines
"""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import time

GCLOUD = shutil.which("gcloud") or os.path.expanduser("~/google-cloud-sdk/bin/gcloud")


def vm_hb_age(url, now):
    try:
        out = subprocess.run([GCLOUD, "storage", "objects", "describe", url, "--format=value(update_time)"],
                             capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None, "describe timed out"
    if out.returncode or not out.stdout.strip():
        return None, "no VM heartbeat object"
    t = dt.datetime.fromisoformat(out.stdout.strip().replace("Z", "+00:00"))
    return (now - t).total_seconds(), "ok"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vms", required=True, help="gcloud compute instances list --format=json output")
    ap.add_argument("--local-hb", required=True)
    ap.add_argument("--hb-gs", required=True)
    ap.add_argument("--stale-s", type=int, default=300)
    ap.add_argument("--boot-grace-s", type=int, default=360, help="time after lastStart before a missing VM hb counts")
    ap.add_argument("--vm-hb-age", action="append", default=[], help="NAME=SECONDS override (tests)")
    a = ap.parse_args(argv)
    now = dt.datetime.now(dt.timezone.utc)
    try:
        local_age = time.time() - os.stat(a.local_hb).st_mtime
    except FileNotFoundError:
        local_age = float("inf")
    if local_age <= a.stale_s:
        return
    over = {k: float(v) for k, v in (s.split("=") for s in a.vm_hb_age)}
    for v in json.load(open(a.vms)):
        if not v["name"].startswith("qrl-bc4a-") or v["status"] != "RUNNING" or not v.get("lastStartTimestamp"):
            continue
        since = (now - dt.datetime.fromisoformat(v["lastStartTimestamp"])).total_seconds()
        if v["name"] in over:
            age, why = over[v["name"]], "override"
        else:
            age, why = vm_hb_age(f"{a.hb_gs.rstrip('/')}/{v['name']}", now)
        if age is not None and age > since:      # heartbeat from an earlier boot of this name
            age, why = None, "only an earlier boot's heartbeat"
        if age is not None and age <= a.stale_s:
            continue
        if age is None and since < a.boot_grace_s:
            continue
        reason = (f"local guard hb {local_age:.0f}s old and VM guard hb "
                  f"{'missing (' + why + ')' if age is None else f'{age:.0f}s old'}, {since:.0f}s after grant")
        print(v["name"], v["zone"].rsplit("/", 1)[-1], reason.replace(" ", "_"))


if __name__ == "__main__":
    main()
