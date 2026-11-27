#!/bin/bash
# Create one Flex-start ct6e-standard-4t for a v2 run, upload code + go.env, start remote_v2.sh.
#   bash rl/tpu/v2_vm.sh NAME ZONE RUN_LABEL MAX_RUN GO_ENV_FILE
# The VM carries labels owner=qrl-bc4a,run=RUN_LABEL so tpu/watchdog_v2.py can price each run separately;
# --max-run-duration (GCE deletes the VM when it expires) is the backstop if the watchdog dies.
set -uo pipefail
set +x
NAME=$1 ZONE=$2 RUN=$3 MAX_RUN=$4 GO=$5
PROJECT=tpu-builders-509006
G="--project=$PROJECT --quiet"
HERE=$(cd "$(dirname "$0")/.." && pwd)
LEDGER=$HERE/tpu/ledger.tsv
PATH="$HOME/google-cloud-sdk/bin:$PATH"
[[ $NAME == qrl-bc4a-* ]] || { echo "refusing: name must start with qrl-bc4a-"; exit 2; }
log() { echo "$(date -u +%FT%TZ) [$NAME] $*"; }

printf '%s\tcreate\t%s\t%s\tFLEX_START run=%s max-run=%s\n' "$(date -u +%FT%TZ)" "$NAME" "$ZONE" "$RUN" "$MAX_RUN" >> "$LEDGER"
log "create ($ZONE, max-run $MAX_RUN)"
gcloud compute instances create "$NAME" --zone="$ZONE" $G \
  --machine-type=ct6e-standard-4t --provisioning-model=FLEX_START --max-run-duration="$MAX_RUN" \
  --request-valid-for-duration=7200s --instance-termination-action=DELETE \
  --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e --image-project=ubuntu-os-accelerator-images \
  --boot-disk-size=300GB --scopes=cloud-platform --labels=owner=qrl-bc4a,run="$RUN" \
  || { log "create failed"; printf '%s\tcreate-failed\t%s\t%s\n' "$(date -u +%FT%TZ)" "$NAME" "$ZONE" >> "$LEDGER"; exit 4; }
log "RUNNING"
for i in $(seq 1 40); do
  gcloud compute ssh "$NAME" --zone="$ZONE" $G --command true >/dev/null 2>&1 && break; sleep 10
done
tar czf - -C "$HERE" qrl tpu tests $( [ -d "$HERE/pilot" ] && echo pilot ) | \
  gcloud compute ssh "$NAME" --zone="$ZONE" $G --command 'mkdir -p ~/rl && tar xzf - -C ~/rl'
gcloud compute scp "$GO" "$NAME:~/go.env" --zone="$ZONE" $G
gcloud compute ssh "$NAME" --zone="$ZONE" $G --command \
  "tmux new-session -d -s qrl 'bash ~/rl/tpu/remote_v2.sh > ~/qrl-remote.log 2>&1'"
log "remote started"
