#!/bin/bash
# $50-total plan driver. Owns the Flex-start ct6e-standard-4t requests in asia-northeast1-b (at most 2),
# bootstraps each VM once it is RUNNING (the first one runs the mini-pilot), hands out the three pairs one
# at a time through ~/go.assign, re-queues a pair whose VM vanished, and deletes a VM as soon as no pair
# is left for it. Spend is capped separately by watchdog_v2.py --total-cap 50 --max-vms 2.
#   bash rl/tpu/driver50.sh            (state in /tmp/qrl-v2/d50; STOP file there ends the loop)
set -uo pipefail
if [ -f /tmp/qrl-v2/plan.json ] && ! grep -q '"plan": "plan50"' /tmp/qrl-v2/plan.json; then
  echo "$(date -u +%FT%TZ) REFUSING: /tmp/qrl-v2/plan.json is not plan50 (see /tmp/qrl-v2/TAKEOVER2); run driver90.sh"; exit 2
fi
export PATH=$HOME/google-cloud-sdk/bin:$PATH
P=tpu-builders-509006 ZONE=asia-northeast1-b
G="--project=$P --zone=$ZONE --quiet"
HERE=$(cd "$(dirname "$0")/.." && pwd)
LEDGER=$HERE/tpu/ledger.tsv
D=/tmp/qrl-v2/d50; mkdir -p $D
BUCKET=gs://qrl-bc4a-compile-cache-tpu-builders-509006
CAP=${CAP:-3072}
VMS=(qrl-bc4a-r1-s0 qrl-bc4a-r2-s0)
log() { echo "$(date -u +%FT%TZ) $*"; }
if [ ! -f $D/queue ]; then
  cat > $D/queue <<'EOF'
sync-s0 sync bounded 0 use 0|async-s0 async fixed 2 use 0
sync-s1 sync bounded 0 use 1|nois-s0 async fixed 2 none 0
async-s1 async fixed 2 use 1|nois-s1 async fixed 2 none 1
EOF
fi

sshv() { local n=$1; shift; timeout 120 gcloud compute ssh "$n" $G --command "$*" 2>/dev/null; }

bootstrap() {  # name pilot(0/1)
  local n=$1 pilot=$2
  for i in $(seq 1 40); do sshv "$n" true && break; sleep 10; done
  tar czf - -C "$HERE" qrl tpu tests | sshv "$n" 'mkdir -p ~/rl && tar xzf - -C ~/rl'
  { echo "CACHE_BUCKET=$BUCKET"; echo "RESULTS_GS=$BUCKET/results/$n"; echo "CAP=$CAP"
    if [ "$pilot" = 1 ]; then echo "PILOT=1"; echo "BENCH='64 0 ;256 16384 '"; echo "BENCH_TIMEOUT=900"; echo "BASE_SAMPLES=16"; fi
  } > $D/go-$n.env
  timeout 120 gcloud compute scp $D/go-$n.env "$n:~/go.env" $G
  sshv "$n" "tmux new-session -d -s qrl 'bash ~/rl/tpu/remote_v2.sh > ~/qrl-remote.log 2>&1'"
  log "[$n] bootstrapped (pilot=$pilot)"
}

decide() {  # after the pilot: learner flags from the bench, steps from the cost model
  local b64=$D/pilot/bench_c64_t0.json b256=$D/pilot/bench_c256_t16384.json flags="" ratio=1.0
  if [ -s $b64 ] && [ -s $b256 ]; then
    ratio=$(/tmp/venv/bin/python -c "import json;a=json.load(open('$b64'))['steps'][-1]['fwd_bwd_s'];b=json.load(open('$b256'))['steps'][-1]['fwd_bwd_s'];print(round(a/b,2))")
    /tmp/venv/bin/python -c "import sys; sys.exit(0 if $ratio >= 1.15 else 1)" && flags="--gdn-chunk 256 --mb-tokens 16384"
  fi
  local spd=1.0; [ -n "$flags" ] && spd=$ratio
  # Filtered prompts if that still allows >= 50 steps, else no filter (half the learner tokens) and more steps.
  local choice=$(cd $HERE && /tmp/venv/bin/python -c "
from plan.cost_model import plan50
fits = lambda s, f: (lambda u: u[1] <= 40 and u[2] <= 47)(plan50($CAP, s, vms=2, learner_speedup=$spd, filt=f)['usd'])
ok = [s for s in (60, 50) if fits(s, True)]
if ok: print(ok[0], 1)
else:
    ok = [s for s in (70, 60, 50, 40, 30) if fits(s, False)]
    print(ok[0] if ok else 30, 0)")
  local steps=${choice% *} filt=${choice#* }
  { echo "STEPS=$steps"; echo "TRAIN_FILTER=$filt"; echo "LEARNER_FLAGS='$flags'"; echo "BENCH_RATIO=$ratio"; } > $D/plan.env
  log "decide: bench ratio c64/c256 $ratio -> flags '$flags', steps $steps, filter $filt"
}

assign() {  # name
  local n=$1 line a b
  line=$(head -1 $D/queue); [ -n "$line" ] || return 1
  a=${line%%|*} b=${line#*|}
  source $D/plan.env
  { echo "RUN_A='$a'"; echo "RUN_B='$b'"; echo "STEPS=$STEPS"; echo "CAP=$CAP"; echo "TRAIN_FILTER=$TRAIN_FILTER"
    echo "LEARNER_FLAGS='$LEARNER_FLAGS'"; echo "EVAL_SPECS='step$STEPS 16'"; } > $D/assign-$n
  sshv "$n" 'mkdir -p ~/rl/pilot' && timeout 120 gcloud compute scp $D/pilot/profile.json "$n:~/rl/pilot/profile.json" $G \
    && timeout 120 gcloud compute scp $D/assign-$n "$n:~/go.assign" $G || return 1
  echo "$line" > $D/cur-$n; sed -i 1d $D/queue
  log "[$n] assigned: $line (steps $STEPS)"
}

create() {  # name
  local n=$1
  printf '%s\tcreate\t%s\t%s\tFLEX_START driver50 max-run=7h\n' "$(date -u +%FT%TZ)" "$n" "$ZONE" >> "$LEDGER"
  gcloud compute instances create "$n" $G --async \
    --machine-type=ct6e-standard-4t --provisioning-model=FLEX_START --max-run-duration=7h \
    --request-valid-for-duration=7200s --instance-termination-action=DELETE \
    --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e --image-project=ubuntu-os-accelerator-images \
    --boot-disk-size=300GB --scopes=cloud-platform --labels=owner=qrl-bc4a,run=plan50 >/dev/null 2>&1 \
    && log "[$n] requested" || log "[$n] request failed"
}

delete_vm() {  # name reason
  log "[$1] delete ($2)"
  printf '%s\tdelete\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$ZONE" "$2" >> "$LEDGER"
  gcloud compute instances delete "$1" $G >/dev/null 2>&1 &
  rm -f $D/boot-$1
}

while [ ! -f $D/STOP ]; do
  qlen=$(grep -c . $D/queue)
  pilot_done=0; [ -s $D/pilot/profile.json ] && pilot_done=1
  # VMs wanted: those running a pair plus one per unassigned pair, at most 2 (the pilot runs on the first).
  busy=0; for o in "${VMS[@]}"; do [ -f $D/cur-$o ] && busy=$((busy + 1)); done
  free=$(( 2 - busy )); want=$(( busy + (qlen < free ? qlen : free) ))
  live=0; declare -A ST=()
  for n in "${VMS[@]}"; do
    ST[$n]=$(gcloud compute instances describe $n $G --format='value(status)' 2>/dev/null)
    [ -n "${ST[$n]}" ] && live=$((live + 1))
  done
  for n in "${VMS[@]}"; do
    st=${ST[$n]}
    if [ -z "$st" ]; then
      rm -f $D/boot-$n
      if [ -f $D/cur-$n ]; then log "[$n] gone mid-pair; re-queue"; { cat $D/cur-$n; cat $D/queue; } > $D/q.tmp; mv $D/q.tmp $D/queue; rm -f $D/cur-$n; fi
      if [ -f $D/pilot_vm ] && [ "$(cat $D/pilot_vm)" = $n ] && [ $pilot_done = 0 ]; then rm -f $D/pilot_vm; fi
      if [ $live -lt $want ] && [ ! -f $D/NO_CREATE ]; then create $n; live=$((live + 1)); fi
      continue
    fi
    if [ "$st" = PENDING ] || [ "$st" = PROVISIONING ] || [ "$st" = STAGING ]; then
      if [ $qlen -eq 0 ] && [ "$st" = PENDING ]; then delete_vm $n no-work-left; fi
      continue
    fi
    [ "$st" = RUNNING ] || continue
    if [ ! -f $D/boot-$n ]; then
      if [ $qlen -eq 0 ] && [ ! -f $D/cur-$n ]; then delete_vm $n granted-but-no-work; continue; fi
      pilot=0
      if [ $pilot_done = 0 ] && { [ ! -f $D/pilot_vm ] || [ "$(cat $D/pilot_vm)" = $n ]; }; then pilot=1; echo $n > $D/pilot_vm; fi
      touch $D/boot-$n; bootstrap $n $pilot >> $D/boot-$n.log 2>&1 &
      continue
    fi
    out=$(sshv $n 'tail -1 ~/qrl-remote.log; test -f ~/go.assign && echo ASSIGN_PENDING; grep -c "pilot done" ~/qrl-remote.log')
    if [ -z "$out" ] && [ -n "$(find $D/boot-$n -mmin +25 2>/dev/null)" ] && ! grep -q bootstrapped $D/boot-$n.log; then
      log "[$n] bootstrap stalled; retrying"; rm -f $D/boot-$n; continue
    fi
    case "$out" in *"waiting for ~/go.assign"*) ;; *) continue ;; esac
    case "$out" in *ASSIGN_PENDING*) continue ;; esac
    if [ -f $D/cur-$n ]; then log "[$n] pair done: $(cat $D/cur-$n)"; cat $D/cur-$n >> $D/done; rm -f $D/cur-$n; fi
    if [ $pilot_done = 0 ]; then
      if [ -f $D/pilot_vm ] && [ "$(cat $D/pilot_vm)" = $n ] && [ "$(echo "$out" | tail -1)" -ge 1 ] 2>/dev/null; then
        mkdir -p $D/pilot
        timeout 300 gcloud compute scp --recurse "$n:~/qrl-results/pilot/*" $D/pilot/ $G >/dev/null 2>&1
        [ -s $D/pilot/profile.json ] && { log "[$n] pilot results fetched"; pilot_done=1; } || log "[$n] pilot fetch failed"
      fi
      [ $pilot_done = 1 ] || continue
    fi
    [ -f $D/plan.env ] || decide
    if [ "$(grep -c . $D/queue)" -gt 0 ]; then assign $n || log "[$n] assign failed"
    else delete_vm $n all-pairs-assigned; fi
  done
  sleep 60
done
log "STOP"
