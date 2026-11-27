# Sourced by remote_v2.sh when AUTO=1 (plan90). The VM schedules itself through GCS under $B:
#   claims/<id>   "name zone lastStart" of the boot working on it (created with --if-generation-match=0;
#                 taken over only when that boot is gone)
#   pilot/        mini-pilot outputs; plan.env  re-price result (STEPS, flags, filter, pair hours)
#   done/<id>     finished work items;  CAP_REACHED  written by any guard at the $90 cap
# Work items: "pilot", then the pairs p1..p3 from plan.json. A VM deletes itself when nothing is left for it,
# when no pair fits its remaining max-run, or after IDLE_MAX_S waiting for the pilot of another VM.
ME="$VM_NAME $VM_ZONE $MYSTART"
IDLE_MAX_S=${IDLE_MAX_S:-5400}
gsx() { timeout 60 gcloud storage ls "$1" >/dev/null 2>&1; }
HB=${HB:-$HOME/guard.hb}; HB_FLAG=${HB_FLAG:-$HOME/HB_STALE}

hb_ok() {  # the VM's own guard ran a full pass within the last 5 minutes, and vm_hbgate.sh hasn't gated us
  [ ! -f "$HB_FLAG" ] && [ -f "$HB" ] && [ $(( $(date +%s) - $(stat -c %Y "$HB") )) -le 300 ]
}
hb_gate() {  # block work while the guard is stale; give up (and delete the VM) after 10 minutes
  local w=0
  until hb_ok; do
    [ $w = 0 ] && log "work gated: guard heartbeat stale or missing"
    [ $w -ge ${HB_GATE_MAX_S:-600} ] && self_delete "guard heartbeat stale for ${w}s"
    sleep ${HB_POLL_S:-30}; w=$((w + ${HB_POLL_S:-30}))
  done
  [ $w -gt 0 ] && log "work ungated after ${w}s"
  return 0
}

self_delete() {
  log "self-delete: $*"
  final_sync
  bash ~/rl/tpu/self_kill.sh "$VM_NAME" "$VM_ZONE"
  sleep 600
  exit 0
}

owner_alive() {  # "name zone start" -> 0 alive, 1 gone, 2 unknown
  local n=$1 z=$2 s=$3 out
  out=$(timeout 60 gcloud compute instances describe "$n" --zone "$z" --format='value(lastStartTimestamp)' 2>&1)
  local rc=$?
  if [ $rc = 0 ]; then [ "$out" = "$s" ] && return 0 || return 1; fi
  case "$out" in *"was not found"*|*"notFound"*) return 1 ;; esac
  return 2
}

claim() {  # id -> 0 when this boot holds the claim
  local u=$B/claims/$1 tmp=$(mktemp) gen owner
  echo "$ME" > "$tmp"
  timeout 60 gcloud storage cp "$tmp" "$u" --if-generation-match=0 --quiet >/dev/null 2>&1 && return 0
  gen=$(timeout 60 gcloud storage objects describe "$u" --format='value(generation)' 2>/dev/null) || return 1
  owner=$(timeout 60 gcloud storage cat "$u" 2>/dev/null) || return 1
  [ "$owner" = "$ME" ] && return 0
  owner_alive $owner; [ $? = 1 ] || return 1
  log "claim $1: previous owner ($owner) is gone (Spot preemption, max-run or a gate); taking over"
  local at; at=$(timeout 60 gcloud storage objects describe "$u" --format='value(update_time)' 2>/dev/null)
  timeout 60 gcloud storage cp "$tmp" "$u" --if-generation-match="$gen" --quiet >/dev/null 2>&1 || return 1
  record_waste "$1" "$owner" "$at"
  return 0
}

record_waste() {  # id "name zone start" claimed_at: the lost work, priced, into <ledger>/waste/ (already in spend)
  python3 - "$@" <<'EOF' | timeout 60 gcloud storage cp - "$(python3 -c "import json; print(json.load(open('$HOME/plan.json'))['ledger_gs'].rstrip('/'))")/waste/$1-$(date +%s).json" --quiet >/dev/null 2>&1
import datetime as dt, json, os, sys
sys.path.insert(0, os.path.expanduser("~/rl/tpu"))
import guard90
wid, owner, at = sys.argv[1], sys.argv[2].split(), sys.argv[3]
pl = json.load(open(os.path.expanduser("~/plan.json")))
model = next((s[2] if len(s) > 2 else "FLEX_START" for s in pl["slots"] if s[0] == owner[0]), "FLEX_START")
now = dt.datetime.now(dt.timezone.utc)
try:
    h = (now - guard90.ts(at)).total_seconds() / 3600
except ValueError:
    h = 0.0
print(json.dumps({"id": wid, "owner": " ".join(owner), "model": model, "claimed_at": at, "released_at": now.isoformat(),
                  "hours_lost": round(h, 3), "usd": round(h * guard90.price(pl, owner[1], model), 2)}))
EOF
}

remaining_h() {
  timeout 60 gcloud compute instances describe "$VM_NAME" --zone "$VM_ZONE" \
    --format='value(lastStartTimestamp,scheduling.maxRunDuration.seconds)' | python3 -c "
import datetime as dt, sys
s, m = (sys.stdin.read().split() + ['', ''])[:2]
el = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(s)).total_seconds()
print(round(((float(m) if m else $([ "${GKE:-0}" = 1 ] && echo 43200 || echo 604800)) - el) / 3600, 2))"
}

live_rates() {  # the confirmed machines at their guard price, for plan.reprice (instead of its flat $5.40/h)
  timeout 300 python3 - <<'EOF'
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/rl/tpu"))
import guard90
pl = json.load(open(os.path.expanduser("~/plan.json")))
vms = [v for v in guard90.list_vms(False) + guard90.list_gke(pl)
       if v["kind"] in ("gce", "gke") and v["status"] in ("RUNNING", "STAGING")]
live = [{"name": v["name"], "usd_h": guard90.price(pl, v["zone"], v["model"]), "offset_h": 0.0} for v in vms]
slot = [guard90.price(pl, s[1], s[2] if len(s) > 2 else "FLEX_START") for s in pl["slots"]]
flex = [guard90.price(pl, s[1]) for s in pl["slots"] if len(s) < 3 or s[2] == "FLEX_START"] or slot
mode = sum(v["usd_h"] for v in live) / len(live) if live else max(slot)
print(json.dumps({"live": live, "replacement_usd_h": [min(min(flex), mode), mode, max(max(slot), mode)],
                  "fee_usd_h": float(pl["gke"]["fee_usd_h"]) if pl.get("gke") else 0.0}))
EOF
}

do_pilot() {
  BENCH='64 0 ;256 16384 '; BENCH_TIMEOUT=900; BASE_SAMPLES=16
  run_pilot
  timeout 600 gcloud storage rsync -r "$RES/pilot" "$B/pilot" --quiet
  local sp
  sp=$(timeout 300 python3 ~/rl/tpu/guard90.py --plan ~/plan.json --print-spend 2>/dev/null || echo 0)
  live_rates > "$RES/pilot/live_rates.json" 2>> "$RES/pilot/reprice.txt" || rm -f "$RES/pilot/live_rates.json"
  X "cd /rl && python -m plan.reprice --pilot /results/pilot --plan /rl/plan.json --spent-usd $sp --setups-left 1 \
     --out /results/pilot/reprice.json" > "$RES/pilot/reprice.txt" 2>&1
  if [ ! -s "$RES/pilot/reprice.env" ]; then
    log "reprice failed: $(tail -3 "$RES/pilot/reprice.txt")"
    printf "STEPS=%s\nTRAIN_FILTER=1\nLEARNER_FLAGS=''\nBENCH_RATIO=1.0\nPAIR_SA_H=6.0\nPAIR_AA_H=3.8\n" \
      "$(python3 -c "import json; print(json.load(open('$HOME/plan.json'))['steps'])")" > "$RES/pilot/reprice.env"
  fi
  log "reprice: $(tail -1 "$RES/pilot/reprice.txt") | $(tr '\n' ' ' < "$RES/pilot/reprice.env")"
  timeout 120 gcloud storage cp "$RES/pilot/reprice.txt" "$RES/pilot/reprice.json" "$B/pilot/" --quiet
  [ -s "$RES/pilot/profile.json" ] || { log "pilot produced no profile.json; not publishing plan.env"; return 1; }
  hb_ok && timeout 120 gcloud storage cp "$RES/pilot/reprice.env" "$B/plan.env" --quiet
}

infra_fail() {  # id what: a fast failure is infrastructure, not science: release the claim and leave the item for
  # another machine; only the 3rd such failure of an item falls through to the normal done+failed marking
  local n; n=$(timeout 60 gcloud storage ls "$B/infra_fail/$1-*" 2>/dev/null | wc -l)
  echo "$ME $2" | timeout 60 gcloud storage cp - "$B/infra_fail/$1-$(date +%s)" --quiet >/dev/null 2>&1
  [ "$n" -ge 2 ] && { log "$1: $2 (infrastructure failure #$((n + 1))); marking it failed"; return 1; }
  timeout 60 gcloud storage rm "$B/claims/$1" --quiet >/dev/null 2>&1
  self_delete "$1: $2 (infrastructure failure #$((n + 1))); claim released for another machine"
}

pair_fits() {  # id (claimed by me): plan budget gate (plan key pair_gate_h90). Fits when spend so far + the P90
  # remaining hours of every other live VM's claimed pair at its rate + this pair's P90 hours at my rate <= the cap.
  timeout 600 python3 - "$1" <<'EOF'
import datetime as dt, json, os, subprocess, sys
sys.path.insert(0, os.path.expanduser("~/rl/tpu"))
import guard90
pl = json.load(open(os.path.expanduser("~/plan.json")))
h90 = pl.get("pair_gate_h90")
if not h90:
    sys.exit(0)
me, B = os.environ["VM_NAME"], os.environ["B"]
gs = lambda *a: subprocess.run(["gcloud", "storage", *a], capture_output=True, text=True, timeout=120)
spent = float(subprocess.run([sys.executable, os.path.expanduser("~/rl/tpu/guard90.py"), "--plan",
                              os.path.expanduser("~/plan.json"), "--print-spend"],
                             capture_output=True, text=True, timeout=300).stdout.split()[-1])
live = {v["name"]: v for v in guard90.list_vms(False) if v["kind"] == "gce" and v["status"] in guard90.CONFIRMED}
rate = lambda v: guard90.price(pl, v["zone"], v["model"])
now = dt.datetime.now(dt.timezone.utc)
others = 0.0
for u in gs("ls", f"{B}/claims/").stdout.split():
    pid = u.rsplit("/", 1)[-1]
    if pid == sys.argv[1] or gs("ls", f"{B}/done/{pid}").returncode == 0:
        continue
    owner = gs("cat", u).stdout.split()
    if not owner or owner[0] == me or owner[0] not in live:
        continue
    at = gs("objects", "describe", u, "--format=value(update_time)").stdout.strip()
    el = (now - dt.datetime.fromisoformat(at.replace("Z", "+00:00"))).total_seconds() / 3600 if at else 0.0
    others += max(0.3, h90 - el) * rate(live[owner[0]])
worst = max(guard90.price(pl, s[1], s[2] if len(s) > 2 else "FLEX_START") for s in pl["slots"])
mine = h90 * (rate(live[me]) if me in live else worst)
tot, cap = spent + others + mine, float(pl["total_cap_usd"])
print(f"gate {sys.argv[1]}: spent ${spent:.2f} + others' P90 remaining ${others:.2f} + mine ${mine:.2f} "
      f"({h90} h) = ${tot:.2f} vs cap ${cap:.0f}")
sys.exit(0 if tot <= cap else 3)
EOF
}

auto_loop() {
  local idle0=$(date +%s) id line need left pick
  mapfile -t IDS < <(python3 -c "import json; [print(k) for k in json.load(open('$HOME/plan.json'))['pairs']]")
  while true; do
    hb_gate
    gsx "$B/CAP_REACHED" && self_delete "cap reached"
    if ! gsx "$B/plan.env"; then
      if claim pilot; then
        log "claimed pilot"
        if ! do_pilot && ! infra_fail pilot "no pilot profile"; then  # 3rd pilot failure: deterministic, end the plan
          for id in pilot "${IDS[@]}"; do echo "$ME pilot failed 3x" | timeout 60 gcloud storage cp - "$B/failed/$id" --quiet
            echo "$ME pilot failed 3x" | timeout 60 gcloud storage cp - "$B/done/$id" --quiet; done
          self_delete "pilot failed 3 times; all items marked failed (stop condition ends the plan)"
        fi
        hb_ok || self_delete "guard went stale during the pilot; pilot left unmarked for another VM"
        echo "$ME" | timeout 60 gcloud storage cp - "$B/done/pilot" --quiet
        idle0=$(date +%s); continue
      fi
      [ $(( $(date +%s) - idle0 )) -gt $IDLE_MAX_S ] && self_delete "idle ${IDLE_MAX_S}s waiting for the pilot"
      sleep 60; continue
    fi
    timeout 60 gcloud storage cp "$B/plan.env" ~/plan.env --quiet && source ~/plan.env
    mkdir -p "$RES/pilot" ~/rl/pilot
    [ -s "$RES/pilot/profile.json" ] || timeout 300 gcloud storage cp "$B/pilot/profile.json" "$RES/pilot/profile.json" --quiet
    cp "$RES/pilot/profile.json" ~/rl/pilot/profile.json
    left=$(remaining_h); pick=""; local nofit=0 open_=0
    for id in "${IDS[@]}"; do
      gsx "$B/done/$id" && continue
      open_=$((open_ + 1))
      line=$(python3 -c "import json; print(json.load(open('$HOME/plan.json'))['pairs']['$id'])")
      case "$line" in *" sync "*) need=$PAIR_SA_H ;; *) need=$PAIR_AA_H ;; esac
      if ! python3 -c "import sys; sys.exit(0 if $left >= $need else 1)"; then nofit=$((nofit + 1)); continue; fi
      if claim "$id"; then
        local gl grc; gl=$(B=$B VM_NAME=$VM_NAME pair_fits "$id" 2>&1); grc=$?
        log "$(echo "$gl" | tail -1)"
        [ $grc = 0 ] || [ $grc = 3 ] || log "gate error rc $grc; proceeding (the guards' total cap still applies)"
        if [ $grc = 3 ]; then  # over budget at the granted rates: skip it for good (done+skipped), never half-run it
          echo "$ME $gl" | timeout 60 gcloud storage cp - "$B/skipped/$id" --quiet
          echo "$ME skipped (budget)" | timeout 60 gcloud storage cp - "$B/done/$id" --quiet
          timeout 60 gcloud storage rm "$B/claims/$id" --quiet >/dev/null 2>&1
          log "SKIPPED $id: does not fit the budget"; open_=$((open_ - 1)); continue
        fi
        pick=$id; break
      fi
    done
    if [ -z "$pick" ]; then
      [ $open_ = 0 ] && self_delete "all work items done"
      [ $nofit -gt 0 ] && self_delete "$nofit open pair(s) need more than the ${left}h max-run left"
      self_delete "every open pair is claimed by a live VM"
    fi
    RUN_A=${line%%|*}; RUN_B=${line#*|}
    EVAL_SPECS=${EVAL_SPECS:-"step$STEPS 16"}  # plan.env may list intermediate evals (with CKPT_STEPS)
    log "claimed $pick: $line (steps $STEPS, ${left}h max-run left, need ${need}h)"
    hb_gate
    local t_pair=$(date +%s) na=${RUN_A%% *} nb=${RUN_B%% *}  # before run_pair, which clears RUN_A/RUN_B
    run_pair
    hb_ok || self_delete "guard went stale during $pick; pair left unmarked for another VM"
    if [ -s "$RES/$na/eval.step$STEPS.json" ] && [ -s "$RES/$nb/eval.step$STEPS.json" ]; then
      echo "$ME" | timeout 60 gcloud storage cp - "$B/done/$pick" --quiet
      log "done $pick"
    elif [ $(( $(date +%s) - t_pair )) -lt ${INFRA_FAIL_S:-1800} ] && infra_fail "$pick" "no final evals after $(( $(date +%s) - t_pair ))s"; then
      :
    else  # not retried: a deterministic failure would just burn budget again
      echo "$ME" | timeout 60 gcloud storage cp - "$B/failed/$pick" --quiet
      echo "$ME failed" | timeout 60 gcloud storage cp - "$B/done/$pick" --quiet
      log "FAILED $pick (missing final evals); marked done+failed"
    fi
    idle0=$(date +%s)
  done
}

dry_loop() {  # DRY mode (B is plan90-dry/): exercise the boot path to the point of training, then release and exit
  local i ok=0
  for i in $(seq 1 20); do hb_ok && { ok=1; break; }; sleep 15; done
  [ $ok = 1 ] && stage guard_heartbeat PASS "local hb $(( $(date +%s) - $(stat -c %Y "$HB") ))s old" || stage guard_heartbeat FAIL
  pgrep -f "guard90.py" >/dev/null && stage guard_running PASS || stage guard_running FAIL
  gsx "$B/hb/$VM_NAME" && stage hb_in_gcs PASS "$B/hb/$VM_NAME" || stage hb_in_gcs FAIL "$B/hb/$VM_NAME"
  pgrep -f "vm_hbgate.sh" >/dev/null && stage hbgate_running PASS || stage hbgate_running FAIL
  if claim pilot && [ "$(timeout 60 gcloud storage cat "$B/claims/pilot")" = "$ME" ]; then
    stage claim PASS "$B/claims/pilot = $ME"
  else
    stage claim FAIL "$B/claims/pilot"
  fi
  local env="-e JAX_PLATFORMS=cpu -e PJRT_DEVICE=CPU" m
  for m in torch torch_tpu vllm vllm_torchtpu transformers datasets safetensors \
           qrl.data qrl.grpo qrl.channel qrl.gdn_patch qrl.rollout qrl.learner qrl.evaluate qrl.profile_prompts qrl.run; do
    timed "import_$m" sudo docker exec $env qrl bash -lc "set -o pipefail; cd /rl && timeout 600 python -c 'import $m; print(\"$m\", getattr($m, \"__version__\", \"\"))' 2>&1 | tail -3"
  done
  timed import_bench_learner sudo docker exec $env qrl bash -lc "cd /rl && python -m py_compile tpu/bench_learner.py && echo compiled"
  timed tokenizer sudo docker exec $env qrl bash -lc "cd /rl && python -c \"from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('$MODEL'); print('vocab', len(t))\""
  timeout 60 gcloud storage rm "$B/claims/pilot" --quiet >/dev/null 2>&1 && ! gsx "$B/claims/pilot" \
    && stage release PASS || stage release FAIL
  stage self_delete START
  self_delete "dry run complete"
}
