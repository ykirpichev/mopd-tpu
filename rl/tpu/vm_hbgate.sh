#!/bin/bash
# Heartbeat gate on a plan90 TPU VM. The guard (guard90.py --self) touches $HB after every pass that could list
# the VMs. If $HB is older than STALE_S: write $FLAG (the work loop refuses to start or continue work while it
# exists) and kill the guard so its restart loop brings up a fresh one. If the heartbeat is still stale GRACE_S
# later: stop the training container and delete this VM. A missing $HB counts from when the gate started.
#   bash vm_hbgate.sh            (env: HB FLAG STALE_S GRACE_S POLL_S LOGF DRY_RUN VM_NAME VM_ZONE)
HB=${HB:-/root/guard.hb}
FLAG=${FLAG:-/root/HB_STALE}
STALE_S=${STALE_S:-300}
GRACE_S=${GRACE_S:-300}
POLL_S=${POLL_S:-30}
LOGF=${LOGF:-/root/startup.log}
DRY_RUN=${DRY_RUN:-0}
log() { echo "$(date -u +%FT%TZ) hbgate: $*" >> "$LOGF"; }
t0=$(date +%s); stale_since=""
log "start (stale ${STALE_S}s, grace ${GRACE_S}s, dry-run $DRY_RUN)"
while true; do
  sleep "$POLL_S"
  now=$(date +%s); mt=$(stat -c %Y "$HB" 2>/dev/null || echo "$t0"); age=$(( now - mt ))
  if [ $age -le $STALE_S ]; then
    [ -n "$stale_since" ] && { log "guard heartbeat fresh again (${age}s); work may continue"; rm -f "$FLAG"; stale_since=""; }
    continue
  fi
  if [ -z "$stale_since" ]; then
    stale_since=$now; touch "$FLAG"
    log "guard heartbeat ${age}s old: work loop gated, restarting the guard"
    [ "$DRY_RUN" = 1 ] || pkill -f '^python3 /root/rl/tpu/guard90.py' 2>/dev/null || python3 -c "
import os, signal
for p in filter(str.isdigit, os.listdir('/proc')):
    try:
        if open(f'/proc/{p}/cmdline', 'rb').read().startswith(b'python3\0/root/rl/tpu/guard90.py'): os.kill(int(p), signal.SIGKILL)
    except OSError: pass"
  elif [ $(( now - stale_since )) -ge $GRACE_S ]; then
    log "guard heartbeat still stale (${age}s) after ${GRACE_S}s: stopping training and deleting this VM"
    if [ "$DRY_RUN" = 1 ]; then log "DRY_RUN: would run 'docker rm -f qrl' and delete ${VM_NAME:-?} in ${VM_ZONE:-?}"; exit 0; fi
    [ "${GKE:-0}" = 1 ] || docker rm -f qrl >/dev/null 2>&1
    bash /root/rl/tpu/self_kill.sh "$VM_NAME" "$VM_ZONE" >> "$LOGF" 2>&1
    sleep 600
  fi
done
