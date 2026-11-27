#!/bin/bash
# $90-total plan driver (agent machine). The TPU VMs run themselves (vm_boot.sh -> vm_startup.sh -> guard90.py
# --self + vm_hbgate.sh + remote_v2.sh AUTO=1 / auto90.sh: pilot, re-price, pairs, self-delete), because this
# machine can be paused for hours. This loop only:
#   - publishes code + plan + vm_startup.sh to GCS;
#   - keeps Flex-start ct6e-standard-4t requests (hard max-run = plan max_run, termination DELETE, bootstrap
#     startup script) open in the plan's slots while fewer than max_vms are granted and work is left, and refuses
#     to request while the local guard heartbeat is stale;
#   - deletes any qrl-bc4a request/VM without the bootstrap script or a max-run, and leftover PENDING requests
#     once max_vms are granted;
#   - heartbeat gate: deletes a granted VM when the local guard heartbeat is stale AND the VM's own guard has
#     not checked in to GCS (hbgate_local.py); a fresh VM-side heartbeat always keeps the VM;
#   - logs milestones and writes its own heartbeat.
# Every gcloud call has a timeout. Config comes only from the plan file; a mismatched CAP/STEPS env is refused.
#   bash rl/tpu/driver90.sh       (STOP file in the state dir ends the loop)
# Test hooks: DRY_RUN=1 (log creates/deletes, publish nothing), ONCE=1, LIST_JSON=<file> (fake instance list),
#   DRIVER_LOCK, GUARD_HB, LOGF, STATE_DIR. Cloud requester (requester90.sh): NO_PUBLISH=1, REQUIRE_LOCAL_GUARD=0, PY.
set -uo pipefail
export PATH=$HOME/google-cloud-sdk/bin:$PATH
PLAN_FILE=${PLAN_FILE:-/tmp/qrl-v2/plan.json}
PY=${PY:-/tmp/venv/bin/python}
HERE=$(cd "$(dirname "$0")/.." && pwd)
LOGF=${LOGF:-/tmp/qrl-v2/driver90.log}
DRY_RUN=${DRY_RUN:-0}
GUARD_HB=${GUARD_HB:-/tmp/qrl-v2/guard.hb}
STALE_S=${STALE_S:-300}
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOGF"; }
[ -f "$PLAN_FILE" ] || { log "REFUSING: no plan file $PLAN_FILE"; exit 2; }
pj() { $PY -c "import json,sys; v=json.load(open('$PLAN_FILE'))$1; print(v)"; }
case "$(pj "['plan']")" in plan90|plan25k1|plan65k1long|plan55sync60|mopdpilot|mopdfull|mopdpaper) ;; *) log "REFUSING: $PLAN_FILE is not a known plan"; exit 2 ;; esac
PLAN_NAME=$(pj "['plan']")
P_CAP=$(pj "['cap']") P_STEPS=$(pj "['steps']") P_TOTAL=$(pj "['total_cap_usd']") MAXV=$(pj "['max_vms']") MAXRUN=$(pj "['max_run']")
D=${STATE_DIR:-$(pj "['state_dir']")}
if [ -n "${CAP:-}" ] && [ "$CAP" != "$P_CAP" ]; then log "REFUSING: CAP=$CAP but plan says $P_CAP (owner $(pj "['owner']"))"; exit 2; fi
if [ -n "${STEPS:-}" ] && [ "$STEPS" != "$P_STEPS" ]; then log "REFUSING: STEPS=$STEPS but plan says $P_STEPS"; exit 2; fi
mkdir -p $D
DRIVER_LOCK=${DRIVER_LOCK:-/tmp/qrl-v2/driver.lock}
exec 9>$DRIVER_LOCK
flock -n 9 || { log "REFUSING: another driver holds $DRIVER_LOCK"; exit 3; }
PLAN_SUM=$(md5sum < "$PLAN_FILE")
P=tpu-builders-509006
LEDGER=$HERE/tpu/ledger.tsv
[ "$DRY_RUN" = 1 ] && LEDGER=$D/ledger-dry.tsv
B=$(pj ".get('gs_prefix', 'gs://qrl-bc4a-compile-cache-tpu-builders-509006/plan90')")
HB=$D/driver.hb
STARTUP=$HERE/tpu/vm_boot.sh
MARK='vm_startup.sh from GCS'
# Work items: plan 'pairs' (independent), or MOPD 'items' with dependencies ({"deps": [...], "h90": hours}).
mapfile -t IDS < <($PY -c "import json; p = json.load(open('$PLAN_FILE')); [print(k) for k in p.get('items') or p['pairs']]")
HAS_ITEMS=$($PY -c "import json; print(1 if json.load(open('$PLAN_FILE')).get('items') else 0)")
declare -A ZONE_OF=() MODEL_OF=()
VMS=()
while read -r n z m; do VMS+=("$n"); ZONE_OF[$n]=$z; MODEL_OF[$n]=$m; done < <($PY -c "
import json; [print(s[0], s[1], s[2] if len(s) > 2 else 'FLEX_START') for s in json.load(open('$PLAN_FILE'))['slots']]")
gsx() { timeout 60 gcloud storage ls "$1" >/dev/null 2>&1; }
gke_live() {  # confirmed (STAGING/RUNNING) GKE TPU nodes; they count against max_vms with the Compute VMs
  $PY -c "import json; print(sum(v['status'] in ('STAGING', 'RUNNING') and 'tpu' in v['name']
                                for v in json.load(open('$D/gke.json'))))" 2>/dev/null || echo 0
}
hb_age() { local t; t=$(stat -c %Y "$1" 2>/dev/null) || { echo 999999; return; }; echo $(( $(date +%s) - t )); }

publish() {
  git -C "$HERE" rev-parse HEAD > "$HERE/qrl/GIT_SHA" 2>/dev/null || true
  tar czf $D/code.tgz -C "$HERE" --exclude='.nltk_data' --exclude='__pycache__' qrl tpu tests plan || return 1
  timeout 300 gcloud storage cp $D/code.tgz "$B/code.tgz" --quiet >/dev/null 2>&1 \
    && timeout 60 gcloud storage cp "$PLAN_FILE" "$B/plan.json" --quiet >/dev/null 2>&1 \
    && timeout 60 gcloud storage cp "$HERE/tpu/vm_startup.sh" "$HERE/tpu/requester90.sh" "$B/" --quiet >/dev/null 2>&1
}
if [ "$DRY_RUN" = 1 ] || [ "${NO_PUBLISH:-0}" = 1 ]; then log "not publishing (dry-run $DRY_RUN, no-publish ${NO_PUBLISH:-0})"; else
  until publish; do log "publish of code/plan to $B failed; retrying"; sleep 60; done; fi
log "driver90 start: cap $P_CAP, steps <= $P_STEPS, max VMs $MAXV, max-run $MAXRUN, slots ${VMS[*]}, dry-run $DRY_RUN"

create() {  # name -> 0 requested (Flex) / granted (Spot), 1 failed, 2 Spot stockout (retry next pass), 3 not tried
  local n=$1 m=${MODEL_OF[$1]}
  local ga; ga=$(hb_age "$GUARD_HB")
  if [ "${REQUIRE_LOCAL_GUARD:-1}" = 1 ] && [ "$ga" -gt "$STALE_S" ]; then log "[$n] REFUSING to request: local guard heartbeat ${ga}s old"; return 3; fi
  if [ "$DRY_RUN" = 1 ]; then
    log "[$n] DRY_RUN: would request $m in ${ZONE_OF[$n]} (max-run $MAXRUN)"
    [ "$m" = SPOT ] && [[ " ${DRY_SPOT_TIMEOUT:-} " == *" $n "* ]] && { log "[$n] DRY_RUN: SPOT insert timed out; holding a place (not granted)"; return 4; }
    [ "$m" = SPOT ] && { [[ " ${DRY_SPOT_GRANT:-} " == *" $n "* ]] && return 0 || return 2; }
    return 0
  fi
  local extra=(--async --request-valid-for-duration=7200s)
  [ "$m" = SPOT ] && extra=()  # Spot is granted or refused synchronously, so the caller can count grants
  printf '%s\tcreate\t%s\t%s\t%s driver90 max-run=%s\n' "$(date -u +%FT%TZ)" "$n" "${ZONE_OF[$n]}" "$m" "$MAXRUN" >> "$LEDGER"
  timeout 240 gcloud compute instances create "$n" --project=$P --zone=${ZONE_OF[$n]} --quiet "${extra[@]}" \
    --machine-type=ct6e-standard-4t --provisioning-model=$m --max-run-duration=$MAXRUN \
    --instance-termination-action=DELETE \
    --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e --image-project=ubuntu-os-accelerator-images \
    --boot-disk-size=300GB --scopes=cloud-platform --labels=owner=qrl-bc4a,run=$PLAN_NAME --metadata=qrl-q=$B \
    --metadata-from-file=startup-script=$STARTUP > $D/create-$n.log 2>&1
  local rc=$?
  [ $rc = 0 ] && { log "[$n] $([ $m = SPOT ] && echo GRANTED || echo requested) $m in ${ZONE_OF[$n]} (max-run $MAXRUN)"; return 0; }
  # a timed-out Spot insert may still be granted server-side: hold its place against MORE Spot inserts this pass only;
  # it never counts as granted (cancelling Flex or the GKE Job needs confirmed STAGING/RUNNING machines)
  [ $rc = 124 ] && [ "$m" = SPOT ] && { log "[$n] SPOT insert timed out; holding a place against more Spot inserts (not granted)"; return 4; }
  local err; err=$(grep -v '^ *$' $D/create-$n.log | tail -2 | tr '\n' ' ' | cut -c1-300)
  log "[$n] $m request failed in ${ZONE_OF[$n]}: $err"
  case "$err" in *RESOURCE_POOL_EXHAUSTED*|*"does not have enough resources"*|*stockout*|*"currently unavailable"*) return 2 ;; esac
  return 1
}

spot_rounds() {  # hold want: ask Spot slots cheapest-first until hold (confirmed + in-flight machines) reaches want.
  # The hold only limits further Spot inserts; nothing here counts as granted or deletes anything.
  local g=$1 w=$2 i=0 c=() b n rc t now=$(date +%s)
  # One Spot asker at a time across the local driver and the Cloud Run requester; the holder refreshes the lock every
  # round (a round is at most the 240 s create timeout), so a lock older than 600 s is dead.
  if [ "$DRY_RUN" != 1 ]; then
    if ! echo $now | timeout 60 gcloud storage cp - $B/spot.lock --if-generation-match=0 --quiet >/dev/null 2>&1; then
      t=$(timeout 60 gcloud storage cat $B/spot.lock 2>/dev/null || echo 0)
      if [ $(( now - ${t:-0} )) -lt 600 ]; then log "Spot: another driver is asking (lock $(( now - t ))s old); skipping"; return 0; fi
      echo $now | timeout 60 gcloud storage cp - $B/spot.lock --quiet >/dev/null 2>&1
    fi
    SPOT_LOCK_HELD=1
    # re-count under the lock: the other driver may have been granted Spot VMs since our list
    t=$(timeout 120 gcloud compute instances list --project=$P --filter="name~qrl-bc4a-" \
          --format='value(name,status)' 2>/dev/null | awk '$2 ~ /^(PROVISIONING|STAGING|RUNNING)$/ && $1 !~ /^qrl-dry-bc4a-/ && ($1 ~ /^qrl-bc4a-/ || $1 ~ /^gke-qrl-bc4a-.*tpu/)' | wc -l) \
      || t=$w
    [ "$t" -gt "$g" ] && { log "Spot: $t machines granted or in flight now (listed $g)"; g=$t; }
  fi
  for n in "${VMS[@]}"; do
    [ "${MODEL_OF[$n]}" = SPOT ] && [ -z "${ST[$n]:-}" ] && [ $(date +%s) -ge "${RETRY_AT[$n]:-0}" ] && c+=("$n")
  done
  # A Spot insert can sit in PROVISIONING ~4 min before a stockout, so ask SPOT_PAR zones at once (cheapest first)
  # and delete grants beyond max_vms right away (most expensive first). The Cloud Run pass has a time budget and
  # starts at a zone offset that rotates every 5 min, so successive passes cover every zone.
  local par=${SPOT_PAR:-4} t0=$(date +%s) won=() k
  if [ "${SPOT_ROTATE:-0}" = 1 ] && [ ${#c[@]} -gt 0 ]; then
    k=$(( (now / 300 * par) % ${#c[@]} )); c=("${c[@]:k}" "${c[@]:0:k}")
  fi
  while [ $i -lt ${#c[@]} ] && [ $g -lt $w ] && [ ! -f $D/NO_CREATE ] && [ $(( $(date +%s) - t0 )) -lt ${SPOT_BUDGET_S:-900} ]; do
    k=$(( w - g > par ? w - g : par ))
    b=("${c[@]:i:k}"); i=$((i + ${#b[@]}))
    date -u +%FT%TZ > $HB  # a Spot batch can take ~4 min; the round as a whole is bounded by SPOT_BUDGET_S
    [ "$DRY_RUN" = 1 ] || date +%s | timeout 60 gcloud storage cp - $B/spot.lock --quiet >/dev/null 2>&1
    for n in "${b[@]}"; do
      RETRY_AT[$n]=$(( $(date +%s) + 300 ))
      { create $n; echo $? > $D/spotrc-$n; } 9>&- &
    done
    wait
    for n in "${b[@]}"; do
      rc=$(cat $D/spotrc-$n 2>/dev/null || echo 1); rm -f $D/spotrc-$n
      case $rc in
        0) g=$((g + 1)); log "[$n] Spot insert returned OK; counted once listed STAGING/RUNNING" ;;
        4) g=$((g + 1)) ;;
        1) RETRY_AT[$n]=$(( $(date +%s) + 1800 )) ;;
      esac
    done
  done
  release_spot_lock
}
release_spot_lock() {
  [ "${SPOT_LOCK_HELD:-0}" = 1 ] && [ "$DRY_RUN" != 1 ] && timeout 30 gcloud storage rm $B/spot.lock --quiet >/dev/null 2>&1
  SPOT_LOCK_HELD=0
}
trap 'release_spot_lock' EXIT
trap 'exit 143' TERM INT
# confirmed machines: qrl-bc4a instances (from the list file) and GKE TPU nodes in STAGING/RUNNING; nothing else
confirmed_from() {  # list.json gke.json
  $PY -c "import json, sys
n = sum(v['status'] in ('STAGING', 'RUNNING') for v in json.load(open(sys.argv[1])) if v['name'].startswith('qrl-bc4a-'))
n += sum(v['status'] in ('STAGING', 'RUNNING') and 'tpu' in v['name'] for v in json.load(open(sys.argv[2])))
print(n)" "$1" "$2" 2>/dev/null || echo 0
}

delete_vm() {  # name zone reason (a delete already issued in the last 15 min is not repeated)
  [ $(( $(date +%s) - ${DEL_AT[$1]:-0} )) -lt 900 ] && return 0
  DEL_AT[$1]=$(date +%s)
  if [ "$DRY_RUN" = 1 ]; then log "[$1] DRY_RUN: would delete in $2 ($3)"; return 0; fi
  log "[$1] delete in $2 ($3)"
  printf '%s\tdelete\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" "$3" >> "$LEDGER"
  if [[ $1 == gke-* ]]; then  # GKE node: drop the Job and resize its pool to 0 (its instance group would re-create it)
    timeout 450 $PY $HERE/tpu/gke90.py --plan "$PLAN_FILE" kill-node "$1" >>$D/gke.log 2>&1 9>&- &
  else
    timeout 300 gcloud compute instances delete "$1" --project=$P --zone=$2 --quiet >/dev/null 2>&1 9>&- &
  fi
}

HAS_GKE=$($PY -c "import json; print(1 if json.load(open('$PLAN_FILE')).get('gke') else 0)")
gke() { timeout 450 $PY $HERE/tpu/gke90.py --plan "$PLAN_FILE" "$@" 2>>$D/gke.log; }
gke_manage() {  # granted want: the GKE Job's pending pod is the GKE "request"; keep it iff more machines are wanted
  [ "$HAS_GKE" = 1 ] || return 0
  local g=$1 w=$2 st cl job pods
  st=$(gke status) || { log "GKE: status failed: $(tail -1 $D/gke.log)"; return 0; }
  read -r cl job pods < <(echo "$st" | $PY -c "import json, sys; d = json.load(sys.stdin)
print(d['cluster'], d.get('job', '-'), sum(1 for p in d.get('pods', []) if p['node']))")
  [ "$cl" = RUNNING ] || { [ "${GKE_LAST:-}" = "$cl" ] || log "GKE: cluster $cl"; GKE_LAST=$cl; return 0; }
  if [ $g -lt $w ] && [ ! -f $D/NO_CREATE ]; then
    if [ "$DRY_RUN" = 1 ]; then log "GKE: DRY_RUN: would ensure Job (job $job)"; return 0; fi
    r=$(gke ensure-job); [ "$r" = "${GKE_LAST:-}" ] || log "GKE: job $r ($g granted, want $w)"; GKE_LAST=$r
  elif [ "$job" != none ] && [ "$pods" = 0 ]; then
    if [ "$DRY_RUN" = 1 ]; then log "GKE: DRY_RUN: would cancel the pending Job and resize TPU pools to 0"; return 0; fi
    log "GKE: $g granted (want $w): cancelling the pending Job, TPU pools to 0"; gke delete-job >/dev/null; gke scale0 >/dev/null
    GKE_LAST=cancelled
  fi
}

stop_requester() {  # end of plan: no more requests from here or from the Cloud Run requester
  [ -n "${REQ_STOPPED:-}" ] && return 0
  REQ_STOPPED=1
  local j r
  j=$(pj "['requester']['scheduler_job']") r=$(pj "['requester']['region']")
  log "STOP CONDITION ($1): no more requests; pausing Cloud Scheduler job $j"
  touch $D/NO_CREATE
  if [ "$DRY_RUN" = 1 ]; then log "DRY_RUN: would run: gcloud scheduler jobs pause $j --location $r --project $P; delete GKE cluster"; return 0; fi
  [ "$HAS_GKE" = 1 ] && log "GKE: $(gke delete-cluster)"
  timeout 120 gcloud scheduler jobs pause "$j" --location "$r" --project $P >/dev/null 2>$D/pause.err \
    && log "paused Cloud Scheduler job $j" || log "pause of $j failed: $(tail -1 $D/pause.err)"
}

declare -A RETRY_AT=() SEEN=() DEL_AT=()
while [ ! -f $D/STOP ]; do
  date -u +%FT%TZ > $HB
  if [ "$(md5sum < "$PLAN_FILE")" != "$PLAN_SUM" ]; then log "plan file changed; exiting so the supervisor restarts with it"; exit 4; fi
  if [ -n "${LIST_JSON:-}" ]; then cp "$LIST_JSON" $D/list.json
  elif ! timeout 120 gcloud compute instances list --project=$P --filter="name~^qrl-bc4a-" --format=json > $D/list.json 2>$D/list.err; then
    log "list failed: $(tail -1 $D/list.err)"; sleep 60 9>&-; continue
  fi
  # CPU dry-run VMs (qrl-dry-bc4a-*) are not plan machines: invisible to requests, gates and the cap (guard prices them)
  $PY -c "import json; json.dump([v for v in json.load(open('$D/list.json')) if not v['name'].startswith('qrl-dry-bc4a-')],
          open('$D/list.json.tmp', 'w'))" && mv $D/list.json.tmp $D/list.json
  if [ -n "${LIST_GKE_JSON:-}" ]; then cp "$LIST_GKE_JSON" $D/gke.json
  else timeout 120 gcloud compute instances list --project=$P --filter="name~^gke-qrl-bc4a-" --format=json > $D/gke.json 2>/dev/null \
    || echo '[{"name": "gke-unknown-tpu", "status": "RUNNING"}]' > $D/gke.json; fi  # unknown counts as live: never over-request
  # 1. heartbeat gate first: nothing else matters if a granted VM is unguarded
  while read -r n z why; do
    [ -n "$n" ] && delete_vm "$n" "$z" "HEARTBEAT GATE: ${why//_/ }"
  done < <($PY -c "import json; json.dump(json.load(open('$D/list.json')) + [v for v in json.load(open('$D/gke.json'))
           if '-tpu-' in v['name'] and 'zone' in v], open('$D/hbvms.json', 'w'))" 2>/dev/null
           $PY $HERE/tpu/hbgate_local.py --vms $D/hbvms.json --local-hb "$GUARD_HB" --hb-gs "$B/hb" --stale-s $STALE_S \
           ${VM_HB_AGE_ARGS:-} 2>>$D/hbgate.err)
  declare -A ST=() ZN=()
  while IFS=, read -r n s z ok mr pm; do
    [ -n "$n" ] || continue; ST[$n]=$s; ZN[$n]=$z
    if [ "$pm" != FLEX_START ] && [ "$pm" != SPOT ]; then delete_vm $n $z "provisioning model $pm: on-demand is never allowed"
    elif [ "$ok" != 1 ]; then delete_vm $n $z "no bootstrap startup script; would run unguarded"
    elif [ -z "$mr" ]; then delete_vm $n $z "no max-run backstop"
    fi
  done < <($PY -c "
import json
for v in json.load(open('$D/list.json')):
    ok = any(i.get('key') == 'startup-script' and '$MARK' in i.get('value', '') for i in v.get('metadata', {}).get('items', []))
    sc = v.get('scheduling', {})
    mr = sc.get('maxRunDuration', {}).get('seconds', '') or sc.get('terminationTime', '')
    print(v['name'], v['status'], v['zone'].rsplit('/', 1)[-1], 1 if ok else 0, mr, sc.get('provisioningModel', 'STANDARD'), sep=',')")
  done_n=0; declare -A DONE_=(); for id in "${IDS[@]}"; do gsx "$B/done/$id" && { done_n=$((done_n + 1)); DONE_[$id]=1; }; done
  left_work=$(( ${#IDS[@]} - done_n ))
  ready_work=$left_work
  if [ "$HAS_ITEMS" = 1 ]; then  # machines are wanted only for items a VM can start now (deps done) or is working on
    ready_work=0
    for id in "${IDS[@]}"; do
      [ -n "${DONE_[$id]:-}" ] && continue
      deps_ok=1
      for dp in $($PY -c "import json; print(' '.join(json.load(open('$PLAN_FILE'))['items']['$id'].get('deps', [])))"); do
        [ -n "${DONE_[$dp]:-}" ] || deps_ok=0
      done
      if [ $deps_ok = 1 ] || gsx "$B/claims/$id"; then ready_work=$((ready_work + 1)); fi
    done
  fi
  cap_hit=0; gsx "$B/CAP_REACHED" && cap_hit=1
  spent=$(timeout 300 $PY $HERE/tpu/guard90.py --plan "$PLAN_FILE" --print-spend 2>/dev/null | tail -1)
  if [ -n "$spent" ] && $PY -c "import sys; sys.exit(0 if float('$spent') >= float('$P_TOTAL') else 1)" 2>/dev/null; then
    [ $cap_hit = 1 ] || log "spend \$$spent >= cap \$$P_TOTAL"; cap_hit=1
  fi
  for m in pilot "${IDS[@]}"; do
    if [ -z "${SEEN[done-$m]:-}" ] && gsx "$B/done/$m"; then SEEN[done-$m]=1; log "MILESTONE done $m"; touch $D/NOTIFY_DONE_$m; fi
  done
  if [ $left_work -eq 0 ] || [ $cap_hit = 1 ]; then
    stop_requester "done $done_n/${#IDS[@]}, cap_hit $cap_hit, spent \$${spent:-?}"
    for n in "${!ST[@]}"; do [ "${ST[$n]}" = PENDING ] && delete_vm $n ${ZN[$n]} "no work left (done $done_n, cap $cap_hit)"; done
    gke_left=0; [ "$HAS_GKE" = 1 ] && [ "$(gke status | $PY -c 'import json,sys; print(json.load(sys.stdin)["cluster"])' 2>/dev/null || echo unknown)" != absent ] && gke_left=1
    if [ $gke_left = 1 ] && [ $(( $(date +%s) - ${GKE_DEL_AT:-0} )) -ge 900 ] && [ "$DRY_RUN" != 1 ]; then GKE_DEL_AT=$(date +%s); log "GKE: $(gke delete-cluster)"; fi
    if [ ${#ST[@]} -eq 0 ] && [ $gke_left = 0 ]; then log "ALL DONE: $done_n/${#IDS[@]} pairs, cap_hit=$cap_hit; no qrl-bc4a VM left"; touch $D/ALL_DONE; break; fi
    [ "${ONCE:-0}" = 1 ] && break; sleep 60 9>&-; continue
  fi
  granted=0 inflight=0
  for n in "${VMS[@]}"; do
    case "${ST[$n]:-}" in
      STAGING|RUNNING) granted=$((granted + 1))
        [ -n "${SEEN[grant-$n]:-}" ] || { SEEN[grant-$n]=1; log "MILESTONE GRANTED $n in ${ZONE_OF[$n]} (${MODEL_OF[$n]}, ${ST[$n]})"; touch $D/NOTIFY_GRANT; } ;;
      PROVISIONING) inflight=$((inflight + 1)) ;;
    esac
  done
  granted=$((granted + $(gke_live)))
  want=$(( ready_work < MAXV ? ready_work : MAXV ))
  for n in "${VMS[@]}"; do
    st=${ST[$n]:-}
    if [ "$st" = PENDING ] && [ $granted -ge $want ]; then delete_vm $n ${ZONE_OF[$n]} "extra-pending: $granted granted, want $want"; continue; fi
    case "$st" in TERMINATED|SUSPENDED|STOPPING) delete_vm $n ${ZONE_OF[$n]} "status $st"; continue ;; esac
    [ -z "$st" ] && [ "${MODEL_OF[$n]}" = SPOT ] && { unset "SEEN[grant-$n]"; continue; }
    if [ -z "$st" ]; then
      unset "SEEN[grant-$n]"
      if [ $granted -lt $want ] && [ ! -f $D/NO_CREATE ] && [ $(date +%s) -ge "${RETRY_AT[$n]:-0}" ]; then
        rm -f $D/createfail-$n
        # --async: a zone that rejects the request (stockout) just never shows up; re-ask at most every 5 min
        RETRY_AT[$n]=$(( $(date +%s) + 300 ))
        while [ "$(jobs -rp | wc -l)" -ge "${CREATE_PAR:-6}" ]; do sleep 1; done  # each gcloud is ~100 MB
        { create $n || touch $D/createfail-$n; } 9>&- &
      fi
    fi
  done
  wait
  gke_manage $granted $want
  # Spot last: Flex and GKE above act on the confirmed count without waiting for ~4-min Spot inserts
  if [ $((granted + inflight)) -lt $want ]; then
    spot_rounds $((granted + inflight)) $want
  fi  # new Spot grants are counted next pass, once listed STAGING/RUNNING
  for n in "${VMS[@]}"; do
    [ -f $D/createfail-$n ] && { RETRY_AT[$n]=$(( $(date +%s) + 1800 )); rm -f $D/createfail-$n; }
  done
  [ "${ONCE:-0}" = 1 ] && break
  sleep 60 9>&-
done
log "STOP"
