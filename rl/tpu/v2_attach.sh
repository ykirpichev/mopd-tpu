#!/bin/bash
# Wait for an already-requested Flex-start VM (gcloud create gives up client-side after 30 min while the
# request stays queued) to reach RUNNING, then upload code + go.env and start remote_v2.sh.
#   bash rl/tpu/v2_attach.sh NAME ZONE GO_ENV_FILE
set -uo pipefail
set +x
NAME=$1 ZONE=$2 GO=$3
G="--project=tpu-builders-509006 --quiet"
HERE=$(cd "$(dirname "$0")/.." && pwd)
PATH="$HOME/google-cloud-sdk/bin:$PATH"
[[ $NAME == qrl-bc4a-* ]] || exit 2
log() { echo "$(date -u +%FT%TZ) [$NAME] $*"; }
log "attach: waiting for RUNNING"
while true; do
  st=$(gcloud compute instances describe "$NAME" --zone="$ZONE" $G --format='value(status)' 2>/dev/null)
  case $st in
    RUNNING) break ;;
    "") log "gone"; exit 5 ;;
    TERMINATED|STOPPING|SUSPENDED) log "status $st"; exit 6 ;;
  esac
  sleep 20
done
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
