#!/bin/bash
# Out-of-band repair on a running MOPD VM, as root over SSH (the agent machine is the only caller):
#   gcloud compute ssh VM --zone Z --command "sudo MODE=... SNIPPET_B64=... bash -s" < rl/tpu/vm_repair.sh
# It first installs the latest $B/code.tgz: qrl/ and plan/ are swapped by rename, tpu/*.sh and *.py are replaced by
# rename (a running bash keeps reading its old inode). Then
#   MODE=run SNIPPET_B64=$(printf %s 'mrun tif 2 3 "" grpo if:16 0 $TSTEPS $TCKPTS 8486 && ...' | base64 -w0)
#       runs the snippet in the background with the work loop's environment and mopd.sh functions; the loop keeps
#       running. Output: /root/qrl-repair.log (copied into the synced results).
#   MODE=restart-loop [MOPD_WAIT_S=5400]
#       kills the work loop (remote_v2.sh and its subshells; the container is recreated by the restart) and starts
#       it again as vm_startup.sh does. The loop re-claims its own item (the claim owner is this boot).
set -uo pipefail
export HOME=/root PATH=$PATH:/snap/bin:/usr/lib/google-cloud-sdk/bin:/usr/local/bin:/opt/google-cloud-sdk/bin
MD=http://metadata.google.internal/computeMetadata/v1/instance
md() { curl -sf -H Metadata-Flavor:Google $MD/$1; }
B=$(md attributes/qrl-q); NAME=$(md name); ZONE=$(md zone | awk -F/ '{print $NF}')
[ -n "$B" ] && [ -n "$NAME" ] || { echo "no metadata"; exit 1; }
export QRL_Q=$B QRL_NAME=$NAME GKE=0 QRL_DRY=0
echo "repair on $NAME ($ZONE), queue $B, mode ${MODE:?}"

T=/root/code.repair; rm -rf $T; mkdir -p $T
timeout 300 gcloud storage cp $B/code.tgz $T/code.tgz --quiet && tar xzf $T/code.tgz -C $T || { echo "code fetch failed"; exit 1; }
for d in qrl plan; do rm -rf /root/rl/$d.old; mv /root/rl/$d /root/rl/$d.old; mv $T/$d /root/rl/$d; done
for f in $T/tpu/*.sh $T/tpu/*.py; do cp "$f" "/root/rl/tpu/.$(basename $f).new" && mv "/root/rl/tpu/.$(basename $f).new" "/root/rl/tpu/$(basename $f)"; done
echo "code installed: $(cat /root/rl/qrl/GIT_SHA | cut -c1-12)"

stage() {
  echo "$(date -u +%FT%TZ) STAGE $*" >> /root/stages.log
  timeout 30 gcloud storage cp /root/stages.log "$QRL_Q/stages/$QRL_NAME.log" --quiet >/dev/null 2>&1
}
export -f stage

if [ "$MODE" = restart-loop ]; then
  pkill -f "bash /root/rl/tpu/remote_v2.sh"; sleep 5; pkill -9 -f "bash /root/rl/tpu/remote_v2.sh"
  stage loop_restart PASS "repair: work loop restarted with $(cat /root/rl/qrl/GIT_SHA | cut -c1-12)"
  echo "$(date -u +%FT%TZ) ===== work loop restarted by vm_repair.sh =====" >> /root/qrl-remote.log
  AUTO=1 VM_NAME=$NAME VM_ZONE=$ZONE MOPD_WAIT_S=${MOPD_WAIT_S:-5400} \
    nohup bash /root/rl/tpu/remote_v2.sh >> /root/qrl-remote.log 2>&1 < /dev/null &
  sleep 2; pgrep -f "bash /root/rl/tpu/remote_v2.sh" >/dev/null && echo "work loop restarted" || echo "restart FAILED"
  exit 0
fi

[ "$MODE" = run ] || { echo "unknown MODE $MODE"; exit 1; }
SNIPPET=$(echo "${SNIPPET_B64:?}" | base64 -d)
cat > /root/repair_run.sh <<'EOF'
set -uo pipefail
MODEL=/dev/shm/models/Qwen3.5-2B; RES=$HOME/qrl-results; CACHE_BUCKET=gs://qrl-bc4a-compile-cache-tpu-builders-509006
B=$QRL_Q; VM_NAME=$QRL_NAME
MYSTART=$(timeout 60 gcloud compute instances describe $VM_NAME --zone $VM_ZONE --format='value(lastStartTimestamp)')
RESULTS_GS=$B/results/$VM_NAME-$(date -u -d "$MYSTART" +%Y%m%dT%H%M%S)
ME="$VM_NAME $VM_ZONE $MYSTART"
log() { echo "$(date -u +%FT%TZ) [repair] $*"; }
gsx() { timeout 60 gcloud storage ls "$1" >/dev/null 2>&1; }
X() { sudo docker exec qrl bash -lc "$*"; }
chip_env() {
  echo "-e TPU_VISIBLE_CHIPS=$1 -e TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 -e TPU_PROCESS_BOUNDS=1,1,1 \
-e TPU_PROCESS_PORT=$2 -e TPU_PROCESS_ADDRESSES=localhost:$2 -e CLOUD_TPU_TASK_ID=0 -e TORCH_TPU_TOPOLOGY=1,1,1"
}
in_chip() { local chip=$1 port=$2 lg=$3; shift 3; sudo docker exec $(chip_env "$chip" "$port") qrl bash -lc "cd /rl && $* > $lg 2>&1"; }
final_sync() {
  cp /root/qrl-repair.log "$RES/" 2>/dev/null
  gcloud storage rsync -r "$RES" "$RESULTS_GS" --exclude='.*\.safetensors$' --quiet >/dev/null 2>&1
}
source /root/rl/tpu/mopd.sh
log "start: $SNIPPET"
eval "$SNIPPET"
log "end rc=$?"
final_sync
EOF
VM_ZONE=$ZONE SNIPPET="${SNIPPET:?}" nohup bash /root/repair_run.sh >> /root/qrl-repair.log 2>&1 < /dev/null &
echo "repair snippet started pid $!"
