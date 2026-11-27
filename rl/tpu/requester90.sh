#!/bin/bash
# Server-side plan90 re-requester: one pass of guard90.py --once and driver90.sh ONCE=1, run by the Cloud Run job
# qrl-bc4a-requester (image google-cloud-cli:slim, default compute service account), which Cloud Scheduler
# qrl-bc4a-requester-tick starts every 5 minutes. GCE Flex-start requests can wait at most 2 h
# (requestValidForDuration <= 7200 s), and the agent machine is paused for hours, so this keeps the slots
# requested without it. Everything comes from GCS (code.tgz, plan.json published by the local driver); the
# caps are enforced by the same guard90/driver90 logic, and on each VM by its own guard. The job's command:
#   bash -c 'gcloud storage cat gs://qrl-bc4a-compile-cache-tpu-builders-509006/plan90/requester90.sh | bash'
set -u
B=${QRL_Q:-gs://qrl-bc4a-compile-cache-tpu-builders-509006/plan90}  # the job sets QRL_Q for plans other than plan90
W=/tmp/w; rm -rf $W; mkdir -p $W/rl $W/d; cd $W
# One pass at a time: the job's task timeout is 1200 s, so a lock older than 1500 s is from a killed pass.
LOCK=$B/requester.lock now=$(date +%s)
if ! echo $now | timeout 60 gcloud storage cp - $LOCK --if-generation-match=0 --quiet 2>/dev/null; then
  t=$(timeout 60 gcloud storage cat $LOCK 2>/dev/null || echo 0)
  if [ $(( now - ${t:-0} )) -lt 1500 ]; then echo "requester: previous pass still running (lock $((now - t))s old); skipping"; exit 0; fi
  echo "requester: taking over stale lock ($((now - t))s old)"; echo $now | timeout 60 gcloud storage cp - $LOCK --quiet
fi
trap 'timeout 60 gcloud storage rm $LOCK --quiet >/dev/null 2>&1' EXIT
for i in 1 2 3; do
  timeout 120 gcloud storage cp $B/code.tgz $B/plan.json $W/ --quiet && break; sleep 10
done
[ -s $W/plan.json ] && [ -s $W/code.tgz ] || { echo "requester: could not fetch code/plan from $B"; exit 1; }
tar xzf $W/code.tgz -C $W/rl
PY=$(command -v python3 || gcloud info --format='value(basic.python_location)')
[ -x "$PY" ] || { echo "requester: no python"; exit 1; }
echo "requester: python $PY"
# First-grant watcher: the first learner train step in any synced run (VMs rsync events/ every 5 min) is recorded
# once in $B/FIRST_TRAIN_STEP with the step's own end time, the VM and its zone.
if ! timeout 60 gcloud storage ls $B/FIRST_TRAIN_STEP >/dev/null 2>&1; then
  for f in $(timeout 60 gcloud storage ls "$B/results/*/*/events/learner.jsonl" 2>/dev/null); do
    l=$(timeout 60 gcloud storage cat "$f" 2>/dev/null | grep -m1 '"kind": "train"') || continue
    "$PY" - "$f" "$l" $W/plan.json <<'EOF' | timeout 60 gcloud storage cp - $B/FIRST_TRAIN_STEP --if-generation-match=0 --quiet
import datetime as dt, json, re, sys
f, ev, plan = sys.argv[1], json.loads(sys.argv[2]), json.load(open(sys.argv[3]))
boot, run = f.split("/results/")[1].split("/")[:2]
vm = re.sub(r"-\d{8}T\d{6}$", "", boot)
zone = next((s[1] for s in plan["slots"] if s[0] == vm), "?")
print(json.dumps({"vm": vm, "zone": zone, "boot": boot, "run": run, "step": ev.get("step"),
                  "step_end_utc": dt.datetime.fromtimestamp(ev["end"], dt.timezone.utc).isoformat(),
                  "seen_utc": dt.datetime.now(dt.timezone.utc).isoformat()}))
EOF
    echo "requester: first train step recorded from $f"; break
  done
fi
echo "requester: guard pass"
timeout 600 "$PY" $W/rl/tpu/guard90.py --plan $W/plan.json --once
echo "requester: driver pass"
PLAN_FILE=$W/plan.json PY="$PY" STATE_DIR=$W/d DRIVER_LOCK=$W/driver.lock GUARD_HB=$W/no-local-guard \
  REQUIRE_LOCAL_GUARD=0 NO_PUBLISH=1 ONCE=1 SPOT_ROTATE=1 SPOT_BUDGET_S=300 LOGF=/dev/stdout timeout 900 bash $W/rl/tpu/driver90.sh
echo "requester: done (rc $?)"
