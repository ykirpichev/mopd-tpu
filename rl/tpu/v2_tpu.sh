#!/bin/bash
# Cloud TPU API path: Flex-start queued resource for one v6e-4 TPU VM (same $1.35/chip-h as GCE Flex-start),
# then upload code + go.env and start remote_v2.sh, like v2_vm.sh.
#   bash rl/tpu/v2_tpu.sh NAME ZONE RUN_LABEL MAX_RUN GO_ENV_FILE      (queued resource and node are both NAME)
set -uo pipefail
set +x
NAME=$1 ZONE=$2 RUN=$3 MAX_RUN=$4 GO=$5
G="--project=tpu-builders-509006 --quiet"
HERE=$(cd "$(dirname "$0")/.." && pwd)
LEDGER=$HERE/tpu/ledger.tsv
PATH="$HOME/google-cloud-sdk/bin:$PATH"
[[ $NAME == qrl-bc4a-* ]] || exit 2
log() { echo "$(date -u +%FT%TZ) [$NAME] $*"; }
printf '%s\tcreate-qr\t%s\t%s\tTPU-API FLEX_START run=%s max-run=%s\n' "$(date -u +%FT%TZ)" "$NAME" "$ZONE" "$RUN" "$MAX_RUN" >> "$LEDGER"
log "queued-resource create ($ZONE, max-run $MAX_RUN)"
gcloud alpha compute tpus queued-resources create "$NAME" --node-id="$NAME" --zone="$ZONE" $G \
  --accelerator-type=v6e-4 --runtime-version=v2-alpha-tpuv6e --provisioning-model=flex-start \
  --max-run-duration="$MAX_RUN" --valid-until-duration=7200s --labels=owner=qrl-bc4a,run="$RUN" \
  || { log "create failed"; exit 4; }
while true; do
  st=$(gcloud compute tpus queued-resources describe "$NAME" --zone="$ZONE" $G --format='value(state.state)' 2>/dev/null)
  case $st in
    ACTIVE) break ;;
    ""|FAILED|SUSPENDED|DELETING) log "queued resource state '${st}'"; exit 5 ;;
  esac
  sleep 20
done
log "RUNNING"
S="gcloud compute tpus tpu-vm"
for i in $(seq 1 40); do $S ssh "$NAME" --zone="$ZONE" $G --command true >/dev/null 2>&1 && break; sleep 10; done
tar czf - -C "$HERE" qrl tpu tests $( [ -d "$HERE/pilot" ] && echo pilot ) | \
  $S ssh "$NAME" --zone="$ZONE" $G --command 'mkdir -p ~/rl && tar xzf - -C ~/rl'
$S scp "$GO" "$NAME:~/go.env" --zone="$ZONE" $G
$S ssh "$NAME" --zone="$ZONE" $G --command "tmux new-session -d -s qrl 'bash ~/rl/tpu/remote_v2.sh > ~/qrl-remote.log 2>&1'"
log "remote started"
