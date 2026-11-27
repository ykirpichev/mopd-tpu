#!/bin/bash
# Run the sync-vs-async experiment on one ct6e-standard-4t (4 v6e chips) as a Compute Engine VM.
#   bash rl/tpu/flex_run.sh                 # auth + preflight only (lists project TPU resources, creates nothing)
#   ALLOW_CREATE=1 bash rl/tpu/flex_run.sh  # Flex-start in europe-west4-a (default)
#   ALLOW_CREATE=1 PROVISIONING=SPOT ZONE=us-east1-d PRICE_PER_H=2.685 bash rl/tpu/flex_run.sh   # Spot fallback
# Needs GCP_SA_KEY_JSON (Cloud Agents secret) and gcloud (rl/tpu/install_gcloud.sh).
# Chips 0/1 run the sync run (vLLM on 0, learner on 1); chips 2/3 run the async run at the same time.
# The VM is deleted on exit (EXIT/INT/TERM trap) and also self-deletes at MAX_RUN.
set -uo pipefail
set +x
PROJECT=${PROJECT:-tpu-builders-509006}
ZONE=${ZONE:-europe-west4-a}
PROVISIONING=${PROVISIONING:-FLEX_START}
PREFIX=qrl-bc4a
NAME=${NAME:-$PREFIX-$(date -u +%m%d%H%M)}
MAX_RUN=${MAX_RUN:-8h}
PRICE_PER_H=${PRICE_PER_H:-5.40}       # Flex-start ct6e-standard-4t: 4 x $1.35/chip-h
BUDGET_USD=${BUDGET_USD:-60}
SPENT_BEFORE=${SPENT_BEFORE:-0}         # TPU spend from earlier sessions, USD
CACHE_BUCKET=${CACHE_BUCKET:-gs://$PREFIX-compile-cache-$PROJECT}
HERE=$(cd "$(dirname "$0")/.." && pwd)
LEDGER=${LEDGER:-$HERE/tpu/ledger.tsv}
OUT=${OUT:-/tmp/qrl-out/$NAME}
G="--project=$PROJECT --quiet"
REMOTE_ENV=${REMOTE_ENV:-}              # e.g. "STEPS=40 SEEDS='0 1'" passed to remote.sh
[ -d "$HOME/google-cloud-sdk/bin" ] && PATH="$HOME/google-cloud-sdk/bin:$PATH"
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/driver.log"; }
# shellcheck source=gcp_auth.sh
source "$HERE/tpu/gcp_auth.sh"
trap gcp_auth_cleanup EXIT INT TERM
ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1)
if [ -n "$ACCOUNT" ]; then
  gcloud config set project "$PROJECT" --quiet >/dev/null 2>&1
  log "auth: using logged-in account $ACCOUNT"
elif [ -n "${GCP_SA_KEY_JSON:-}" ]; then
  gcp_auth || { log "auth failed"; exit 2; }
else
  log "auth: no active gcloud account; run 'gcloud auth login --no-launch-browser --update-adc' or set GCP_SA_KEY_JSON"
  exit 2
fi

list_or_fail() {  # stdout of a gcloud list; nonzero exit if the API call fails
  local out
  out=$("$@" 2>>"$OUT/gcloud-stderr.log") || { log "list failed: $*"; return 1; }
  printf '%s' "$out"
}

preflight() {
  local tpus qrs="" clusters ct6e z q
  tpus=$(list_or_fail gcloud compute tpus tpu-vm list --zone=- $G --format='value(name)') || return 2
  for z in europe-west4-a europe-west4-b us-east1-d us-east5-a us-east5-b asia-northeast1-b us-south1-ai1b \
           us-central1-b asia-southeast1-b southamerica-west1-a; do
    q=$(list_or_fail gcloud compute tpus queued-resources list --zone=$z $G --format='value(name)') || return 2
    qrs="$qrs$q"
  done
  clusters=$(list_or_fail gcloud container clusters list $G --format='value(name,location)') || return 2
  ct6e=$(list_or_fail gcloud compute instances list $G --filter='machineType~ct6e' --format='value(name,zone,status)') || return 2
  log "tpu-vms: ${tpus:-none} | queued: ${qrs:-none} | clusters: ${clusters:-none} | ct6e: ${ct6e:-none}"
  [ -z "$tpus$qrs$clusters$ct6e" ]
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  gcp_auth_cleanup
  if [ -n "${CREATED:-}" ]; then
    log "cleanup: pulling results"
    timeout 600 gcloud compute ssh "$NAME" --zone=$ZONE $G \
      --command 'tar czf - --exclude="*.safetensors" -C ~ qrl-results qrl-remote.log' > "$OUT/qrl-results.tgz" 2>/dev/null
    log "cleanup: deleting $NAME"
    for i in 1 2 3 4; do
      gcloud compute instances delete "$NAME" --zone=$ZONE $G && break
      sleep $((4 * 2 ** i))
    done
    printf '%s\tdelete\t%s\t%s\n' "$(date -u +%FT%TZ)" "$NAME" "$ZONE" >> "$LEDGER"
  fi
  if [ -n "${T_RUNNING:-}" ]; then
    local h; h=$(python3 -c "print(($(date +%s)-$T_RUNNING)/3600)")
    log "billed-hours ~$h cost ~\$$(python3 -c "print(round($h*$PRICE_PER_H,2))") total ~\$$(python3 -c "print(round($SPENT_BEFORE+$h*$PRICE_PER_H,2))")"
  fi
  log "remaining $PREFIX instances: $(gcloud compute instances list $G --filter="name~^$PREFIX" --format='value(name,zone,status)' 2>/dev/null | tr '\n' ' ')"
  exit $rc
}

preflight; pf=$?
[ $pf -eq 2 ] && { log "preflight: a list call failed; refusing to create"; exit 3; }
[ $pf -ne 0 ] && { log "preflight: project still has TPU/GKE/ct6e resources; refusing to create"; exit 3; }
[ "${ALLOW_CREATE:-0}" = 1 ] || { log "preflight clean; set ALLOW_CREATE=1 to create"; exit 0; }
python3 -c "import sys; sys.exit(0 if $SPENT_BEFORE + $PRICE_PER_H/4 < $BUDGET_USD else 1)" || { log "budget exhausted"; exit 5; }

trap cleanup EXIT INT TERM
if ! gcloud storage buckets describe "$CACHE_BUCKET" $G >/dev/null 2>&1; then
  gcloud storage buckets create "$CACHE_BUCKET" $G --location="${ZONE%-*}" --uniform-bucket-level-access \
    --labels=owner=$PREFIX && \
  printf '%s\tcreate-bucket\t%s\t%s\n' "$(date -u +%FT%TZ)" "$CACHE_BUCKET" "${ZONE%-*}" >> "$LEDGER"
fi

case $PROVISIONING in
  FLEX_START) PFLAGS="--provisioning-model=FLEX_START --max-run-duration=$MAX_RUN --request-valid-for-duration=7200s" ;;
  SPOT) PFLAGS="--provisioning-model=SPOT --max-run-duration=$MAX_RUN" ;;
  *) log "unknown PROVISIONING=$PROVISIONING"; exit 2 ;;
esac
printf '%s\tcreate\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$NAME" "$ZONE" "$PROVISIONING" >> "$LEDGER"
CREATED=1
log "create $NAME ($ZONE, $PROVISIONING, max-run $MAX_RUN)"
# shellcheck disable=SC2086
gcloud compute instances create "$NAME" --zone=$ZONE $G \
  --machine-type=ct6e-standard-4t $PFLAGS --instance-termination-action=DELETE \
  --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e --image-project=ubuntu-os-accelerator-images \
  --boot-disk-size=300GB --scopes=cloud-platform --labels=owner=$PREFIX \
  || { log "create failed"; exit 4; }
T_RUNNING=$(date +%s)
log "RUNNING"

for i in $(seq 1 30); do
  gcloud compute ssh "$NAME" --zone=$ZONE $G --command true >/dev/null 2>&1 && break; sleep 10
done
tar czf - -C "$HERE" qrl tpu tests | gcloud compute ssh "$NAME" --zone=$ZONE $G --command 'mkdir -p ~/rl && tar xzf - -C ~/rl'
gcloud compute ssh "$NAME" --zone=$ZONE $G --command \
  "tmux new-session -d -s qrl 'env CACHE_BUCKET=$CACHE_BUCKET $REMOTE_ENV bash ~/rl/tpu/remote.sh > ~/qrl-remote.log 2>&1; echo REMOTE_EXIT \$? >> ~/qrl-remote.log'"

while true; do
  sleep 60
  h=$(python3 -c "print(($(date +%s)-$T_RUNNING)/3600)")
  spend=$(python3 -c "print(round($SPENT_BEFORE + $h*$PRICE_PER_H, 2))")
  st=$(timeout 60 gcloud compute ssh "$NAME" --zone=$ZONE $G --command 'tail -n 2 ~/qrl-remote.log' 2>/dev/null | tr '\n' ' ')
  log "spend~\$$spend | $st"
  # Stop with 15 minutes of headroom so pulling results and deleting stay under the cap.
  python3 -c "import sys; sys.exit(0 if $spend + $PRICE_PER_H/4 < $BUDGET_USD else 1)" || { log "budget stop"; exit 5; }
  echo "$st" | grep -q REMOTE_EXIT && break
  gcloud compute instances describe "$NAME" --zone=$ZONE $G --format='value(status)' 2>/dev/null | grep -q RUNNING || { log "VM gone"; exit 6; }
done
log "remote finished"
