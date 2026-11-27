#!/bin/bash
# GCE startup-script for plan90 VMs (runs as root on every boot). Pulls the code and plan from GCS, starts the
# spend guard (guard90.py --self) under a restart loop, then the autonomous work loop (remote_v2.sh AUTO=1),
# so a granted VM needs nothing from the agent machine. If the code can't be fetched within 10 minutes the VM
# deletes itself rather than run unguarded. In a GKE pod (GKE=1, started by gke90.py's Job) the metadata server is
# the node's, so NAME/ZONE are the node's instance; the pod stays in the foreground until the work loop exits.
# Every stage is timestamped in /root/stages.log and copied to $Q/stages/<name>.log (the first-grant watcher).
# DRY mode (instance metadata qrl-dry=1, or QRL_DRY=1 in a pod): a CPU-only end-to-end check of this same path.
# The guard runs with --no-enforce on a separate ledger/hb under plan90-dry/, the work loop claims from the
# plan90-dry/ queue only, runs an in-container import smoke instead of the pilot, releases the claim and exits.
MD=http://metadata.google.internal/computeMetadata/v1/instance
md() { curl -sf -H Metadata-Flavor:Google $MD/$1 2>/dev/null || python3 -c "
import urllib.request as u; print(u.urlopen(u.Request('$MD/$1', headers={'Metadata-Flavor': 'Google'}), timeout=10).read().decode())" 2>/dev/null; }
B=${QRL_PLAN_Q:-$(md attributes/qrl-q)}  # the plan's GCS prefix (instance metadata qrl-q, set by driver90)
B=${B:-gs://qrl-bc4a-compile-cache-tpu-builders-509006/plan90}
NAME=$(md name)
ZONE=$(md zone | awk -F/ '{print $NF}')
DRY=${QRL_DRY:-$(md attributes/qrl-dry)}; [ "$DRY" = 1 ] || DRY=0
Q=$B; [ "$DRY" = 1 ] && Q=${B}-dry
export HOME=/root GKE=${GKE:-0} QRL_DRY=$DRY QRL_Q=$Q QRL_NAME=$NAME
export PATH=$PATH:/snap/bin:/usr/lib/google-cloud-sdk/bin:/usr/local/bin:/opt/google-cloud-sdk/bin
cd /root
log() { echo "$(date -u +%FT%TZ) $*" >> /root/startup.log; }
stage() {  # name status detail
  echo "$(date -u +%FT%TZ) STAGE $*" >> /root/stages.log
  timeout 30 gcloud storage cp /root/stages.log "$QRL_Q/stages/$QRL_NAME.log" --quiet >/dev/null 2>&1
}
export -f stage 2>/dev/null
log "boot $NAME $ZONE (gke $GKE, dry $DRY)"
t0=$(date +%s)
ok=0
for i in $(seq 1 20); do
  timeout 120 gcloud storage cp $B/code.tgz /root/code.tgz --quiet && timeout 60 gcloud storage cp $B/plan.json /root/plan.json --quiet \
    && { rm -rf /root/rl; mkdir -p /root/rl && tar xzf /root/code.tgz -C /root/rl && ok=1 && break; }
  sleep 30
done
if [ $ok != 1 ]; then
  log "code fetch failed; deleting myself"
  stage code_fetch FAIL
  [ "$GKE" = 1 ] && exit 1  # the pod fails; the guard/requester see a node without a heartbeat
  gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet
  exit 1
fi
log "code fetched"
stage boot PASS "$NAME $ZONE gke=$GKE dry=$DRY"
stage code_fetch PASS "$(( $(date +%s) - t0 ))s, code.tgz $(stat -c %s /root/code.tgz) bytes"
GUARD_ARGS=""
if [ "$DRY" = 1 ]; then  # separate ledger + heartbeats; the dry guard never deletes anything
  python3 -c "import json; p = json.load(open('/root/plan.json')); p['ledger_gs'] = '$Q/ledger'
json.dump(p, open('/root/plan.json', 'w'), indent=1)"
  GUARD_ARGS="--no-enforce"
fi
nohup bash -c "while true; do timeout 3600 python3 /root/rl/tpu/guard90.py --plan /root/plan.json --self $NAME $GUARD_ARGS \
  --state /root/guard.json --log /root/guard.log --heartbeat /root/guard.hb --lock /root/guard.lock; sleep 10; done" \
  >/dev/null 2>&1 &
VM_NAME=$NAME VM_ZONE=$ZONE nohup bash /root/rl/tpu/vm_hbgate.sh >/dev/null 2>&1 &
log "guard started"
stage guard_started PASS "args '$GUARD_ARGS'"
if [ "$GKE" != 1 ] && ! command -v docker >/dev/null; then  # the ubuntu-accel TPU image ships without docker
  t1=$(date +%s)
  for i in 1 2 3 4 5; do
    timeout 600 bash -c "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io" >> /root/startup.log 2>&1 \
      && systemctl start docker && break
    sleep 30
  done
  if ! docker info >/dev/null 2>&1; then
    log "docker install failed; deleting myself (work items untouched)"
    stage docker_install FAIL
    gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet
    exit 1
  fi
  log "docker installed"
  stage docker_install PASS "$(( $(date +%s) - t1 ))s, $(docker --version)"
fi
if ! pgrep -f "remote_v2.sh" >/dev/null; then
  AUTO=1 VM_NAME=$NAME VM_ZONE=$ZONE nohup bash /root/rl/tpu/remote_v2.sh > /root/qrl-remote.log 2>&1 &
  RPID=$!
  log "work loop started"
  stage work_loop_started PASS
  if [ "$GKE" = 1 ]; then  # a container must keep its main process: follow the work loop, then release the node
    tail -F /root/startup.log /root/qrl-remote.log /root/guard.log 2>/dev/null &
    wait $RPID
    log "work loop exited; releasing the GKE node"
    bash /root/rl/tpu/self_kill.sh "$NAME" "$ZONE" >> /root/startup.log 2>&1
    [ "$DRY" = 1 ] && exit 0
    sleep 600
  fi
fi
