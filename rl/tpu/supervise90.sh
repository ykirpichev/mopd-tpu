#!/bin/bash
# Supervisor for the agent-machine side of plan90: keeps exactly one guard90.py and one driver90.sh running,
# logging straight to files (no tee), and restarts either one when it exits or its heartbeat is older than
# STALE_S (default 300 s). The TPU VMs carry their own guard, so a paused agent machine never leaves them unguarded.
#   tmux new-window -n watchdog50 "bash rl/tpu/supervise90.sh"
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
S=${S:-/tmp/qrl-v2}
STALE_S=${STALE_S:-300}
PY=/tmp/venv/bin/python
log() { echo "$(date -u +%FT%TZ) $*" >> $S/supervisor.log; }
exec 8>$S/supervisor.lock
flock -n 8 || { log "REFUSING: another supervisor holds $S/supervisor.lock"; exit 3; }
start_guard() {
  setsid $PY $HERE/guard90.py --plan $S/plan.json --state $S/guard.json --log $S/watchdog.log \
    --heartbeat $S/guard.hb --lock $S/watchdog.lock </dev/null >>$S/watchdog.log 2>&1 8>&- &
  GUARD=$!; log "guard started pid $GUARD"
}
start_driver() {
  PLAN_FILE=$S/plan.json LOGF=$S/driver90.log GUARD_HB=$S/guard.hb DRIVER_LOCK=$S/driver.lock STATE_DIR=$S/d90 \
    setsid bash $HERE/driver90.sh </dev/null >>$S/driver90.log 2>&1 8>&- &
  DRIVER=$!; log "driver started pid $DRIVER"
}
age() { local t; t=$(stat -c %Y "$1" 2>/dev/null) || { echo 999999; return; }; echo $(( $(date +%s) - t )); }
mkdir -p $S/d90; date -u +%FT%TZ > $S/guard.hb; date -u +%FT%TZ > $S/d90/driver.hb
start_guard; start_driver
while true; do
  sleep 30
  if ! kill -0 $GUARD 2>/dev/null; then log "guard exited; restarting"; sleep 5; start_guard
  elif [ $(age $S/guard.hb) -gt $STALE_S ]; then
    log "guard heartbeat $(age $S/guard.hb)s old; killing and restarting"; kill -9 -- -$GUARD 2>/dev/null; kill -9 $GUARD 2>/dev/null
    sleep 2; date -u +%FT%TZ > $S/guard.hb; start_guard
  fi
  if [ -f $S/d90/ALL_DONE ] || [ -f $S/d90/STOP ]; then
    kill -0 $DRIVER 2>/dev/null || { [ -z "${DLOG:-}" ] && { log "driver finished (ALL_DONE/STOP); not restarting"; DLOG=1; }; }
  elif ! kill -0 $DRIVER 2>/dev/null; then log "driver exited; restarting"; sleep 5; start_driver
  elif [ $(age $S/d90/driver.hb) -gt $STALE_S ]; then
    log "driver heartbeat $(age $S/d90/driver.hb)s old; killing and restarting"; kill -9 -- -$DRIVER 2>/dev/null; kill -9 $DRIVER 2>/dev/null
    sleep 2; date -u +%FT%TZ > $S/d90/driver.hb; start_driver
  fi
done
