#!/bin/bash
# Change the training length of the live plan90 run and raise its total cap together, from the agent machine.
#   bash rl/tpu/set_steps90.sh STEPS NEW_CAP_USD            dry run: prices it, probes the VMs read-only, prints the plan
#   bash rl/tpu/set_steps90.sh STEPS NEW_CAP_USD --apply    does it (only with the user's explicit approval)
# Refuses unless the re-price at the current spend and live rates (plan.reprice.live_total over plan50) gives
# median <= NEW_CAP - 10 and P90 <= NEW_CAP. --apply, in this order:
#   1. total_cap_usd = NEW_CAP, reprice_median_max_usd = NEW_CAP - 10 in every plan.json a guard reads: the repo copy
#      (rl/tpu/plan90.json), the local supervisor's /tmp/qrl-v2/plan.json (its guard90 and driver90 exit on a plan
#      change and are restarted with it; driver90 republishes it to GCS), $B/plan.json (Cloud Run requester, new
#      VMs), and /root/plan.json + /root/rl/plan.json on each live VM over IAP ssh (the VM guard exits on the change
#      and its restart loop reloads it). Waits for each VM guard to log the new cap.
#   2. published.env on every live VM that has one (an unfinished do_pilot would re-publish it as plan.env).
#   3. $B/plan.env with STEPS, PAIR_SA_H, PAIR_AA_H (other keys kept).
#   4. every pair that is claimed or done at another step count: stop the owner VM's work loop and container,
#      archive the pair's local results, release claims/<id> and remove done/ failed/ infra_fail/ markers, then
#      restart the work loop (same boot, so the same results prefix); it re-claims with the new STEPS.
#   5. appends to $B/OPERATOR_NOTES.txt.
# Max-run: a VM whose remaining max-run cannot fit a pair (PAIR_*_H) self-deletes through auto90's nofit check,
# and driver90/requester90 grant a replacement (new VMs get the plan max_run, 12 h, which fits any single pair);
# the replacement setup is added to the price. GKE TPU pods can't be patched here: --apply refuses while one is live.
set -uo pipefail
export PATH=$HOME/google-cloud-sdk/bin:$PATH
HERE=$(cd "$(dirname "$0")/.." && pwd)
B=gs://qrl-bc4a-compile-cache-tpu-builders-509006/plan90
P=tpu-builders-509006
PY=${PY:-$( [ -x /tmp/venv/bin/python ] && echo /tmp/venv/bin/python || echo python3)}
LOCAL_PLAN=${LOCAL_PLAN:-/tmp/qrl-v2/plan.json}
RESTART_H=${RESTART_H:-0.15}       # stop + work-loop setup on the same VM before a restarted pair trains
PILOT_LEFT_H=${PILOT_LEFT_H:-0.25} # a VM still in the pilot (base eval) is busy this long before its first pair
STEPS=${1:?usage: set_steps90.sh STEPS NEW_CAP_USD [--apply]} NEW_CAP=${2:?usage: set_steps90.sh STEPS NEW_CAP_USD [--apply]}
APPLY=0; [ "${3:-}" = --apply ] && APPLY=1
W=$(mktemp -d /tmp/set_steps90.XXXX)
log() { echo "$(date -u +%FT%TZ) $*"; }
die() { log "REFUSING: $*"; exit 2; }
gsx() { timeout 60 gcloud storage ls "$1" >/dev/null 2>&1; }
ssh_vm() {  # name zone script: runs script as root on the VM over IAP
  local b64; b64=$(printf '%s' "$3" | base64 -w0)
  timeout 300 gcloud compute ssh "$1" --zone "$2" --project $P --tunnel-through-iap --quiet \
    --command "echo $b64 | base64 -d | sudo bash" 2>&1 \
    | grep -v -e '^Warning' -e '^WARNING' -e 'Existing host keys' -e 'consider installing NumPy' -e 'increasing_the_tcp' -e '^$' \
    | sed 's/, allow \[.*\], ledger/, ledger/'
}

timeout 60 gcloud storage cp $B/plan.json $B/plan.env $W/ --quiet >/dev/null 2>&1 || die "cannot read $B/plan.json and plan.env"
mkdir -p $W/pilot
timeout 120 gcloud storage cp "$B/pilot/profile.json" $W/pilot/ --quiet >/dev/null 2>&1 || die "no $B/pilot/profile.json"
for f in timing.json bench_c64_t0.json bench_c256_t16384.json; do  # newest boot's pilot output that has it
  u=$(timeout 60 gcloud storage ls "$B/results/*/pilot/$f" 2>/dev/null | sort | tail -1)
  [ -n "$u" ] && timeout 60 gcloud storage cp "$u" $W/pilot/ --quiet >/dev/null 2>&1
done
SPENT=$(timeout 300 $PY $HERE/tpu/guard90.py --plan $W/plan.json --print-spend 2>/dev/null) || die "guard90 --print-spend failed"
log "spend now \$$SPENT (guard90 --print-spend); current $B/plan.env: $(tr '\n' ' ' < $W/plan.env)"

# Work state: claims, done markers, and which boot owns each claim.
gsx "$B/done/pilot" && PILOT_DONE=1 || PILOT_DONE=0
PILOT_OWNER=$(timeout 60 gcloud storage cat $B/claims/pilot 2>/dev/null | awk '{print $1}')
mapfile -t IDS < <($PY -c "import json; [print(k) for k in json.load(open('$W/plan.json'))['pairs']]")
OLD_STEPS=$(sed -n 's/^STEPS=//p' $W/plan.env)
declare -A OWNER=() ISDONE=()
for id in "${IDS[@]}"; do
  OWNER[$id]=$(timeout 60 gcloud storage cat $B/claims/$id 2>/dev/null || true)
  gsx "$B/done/$id" && ISDONE[$id]=1 || ISDONE[$id]=0
done
RESTART=()  # pairs claimed or done at the old step count
if [ "$OLD_STEPS" != "$STEPS" ]; then
  for id in "${IDS[@]}"; do [ -n "${OWNER[$id]}" ] || [ "${ISDONE[$id]}" = 1 ] && RESTART+=("$id"); done
fi
RESTART_VMS=" "; for id in "${RESTART[@]}"; do [ "${ISDONE[$id]}" = 1 ] || RESTART_VMS+="${OWNER[$id]%% *} "; done

$PY - "$W" "$SPENT" "$STEPS" "$NEW_CAP" "$RESTART_VMS" "$PILOT_DONE" "${PILOT_OWNER:-}" "$RESTART_H" "$PILOT_LEFT_H" \
  "$HERE" > $W/price.json <<'EOF' || die "re-price failed"
import datetime as dt, json, os, subprocess, sys
W, spent, steps, new_cap, restart_vms, pilot_done, pilot_owner, restart_h, pilot_left_h, here = sys.argv[1:]
spent, steps, new_cap = float(spent), int(steps), float(new_cap)
sys.path.insert(0, here); sys.path.insert(0, os.path.join(here, "tpu"))
import numpy as np
import guard90
from plan.cost_model import plan50
from plan.reprice import REBOOT_H, REGRANT_WAIT_H, live_total, measured, r_h
guard90.PLAN_PATH = os.path.join(W, "plan.json")
pl = json.load(open(guard90.PLAN_PATH))
m = measured(os.path.join(W, "pilot"), pl["cap"])
vms = [v for v in guard90.list_vms(False) + guard90.list_gke(pl)
       if v["kind"] in ("gce", "gke") and v["status"] in ("RUNNING", "STAGING")]
now = dt.datetime.now(dt.timezone.utc)
live = []
for v in vms:
    rem = 12.0
    if v["kind"] == "gce":
        out = subprocess.run(["gcloud", "compute", "instances", "describe", v["name"], "--zone", v["zone"],
                              "--format=value(lastStartTimestamp,scheduling.maxRunDuration.seconds)"],
                             capture_output=True, text=True, timeout=60).stdout.split()
        if out:
            mr = float(out[1]) if len(out) > 1 else 604800
            rem = (mr - (now - dt.datetime.fromisoformat(out[0])).total_seconds()) / 3600
    off = float(restart_h) if v["name"] in restart_vms.split() else 0.0
    if v["name"] == pilot_owner and pilot_done != "1":
        off = max(off, float(pilot_left_h))
    live.append({"name": v["name"], "zone": v["zone"], "kind": v["kind"],
                 "usd_h": guard90.price(pl, v["zone"], v["model"]), "offset_h": off, "max_run_left_h": round(rem, 2)})
slot = [guard90.price(pl, s[1], s[2] if len(s) > 2 else "FLEX_START") for s in pl["slots"]]
flex = [guard90.price(pl, s[1]) for s in pl["slots"] if len(s) < 3 or s[2] == "FLEX_START"] or slot
mode = sum(v["usd_h"] for v in live) / len(live) if live else max(slot)
rates = {"live": live, "replacement_usd_h": [min(min(flex), mode), mode, max(max(slot), mode)],
         "fee_usd_h": float(pl["gke"]["fee_usd_h"]) if pl.get("gke") else 0.0}
kw = dict(vms=2, learner_speedup=m.get("learner_speedup"), trunc=m["trunc"], finished_len=m["finished_len"],
          eval_tok_s=m.get("eval_tok_s"), t128=m.get("t128"), informative=m["informative"])
r = plan50(pl["cap"], steps, spent_usd=spent, setups_left=0, samples=True, **kw)
lt = live_total(r.pop("_samples"), spent, rates)
sa_h, aa_h = r_h(r["pair_sync_async_h"]), r_h(r["pair_async_async_h"])
# Max-run: p1/p2 start on the two live machines, p3 on the first one free (live_total's schedule). A machine that
# can't fit its next pair self-deletes (auto90 nofit) and a replacement does it: one more setup + regrant wait.
top = sorted(live, key=lambda v: -v["usd_h"])[:2]
fit_sa = [v["max_run_left_h"] - v["offset_h"] >= sa_h for v in top]
repl = (2 - len(top)) + sum(not f for f in fit_sa)
fit_p3 = any(f and v["max_run_left_h"] - v["offset_h"] - sa_h >= aa_h for v, f in zip(top, fit_sa))
repl += 0 if fit_p3 else 1
extra = max(0, repl - (2 - len(top))) * (REBOOT_H * mode + rates["fee_usd_h"] * float(np.mean(REGRANT_WAIT_H)))
usd = [x + extra for x in lt["usd"]]
ok = usd[1] <= new_cap - 10 and usd[2] <= new_cap
print(json.dumps({"steps": steps, "new_cap": new_cap, "spent": spent, "usd": usd, "wall_h_left": lt["wall_h_left"], "vm_h_left": lt["vm_h_left"], "PAIR_SA_H": sa_h, "PAIR_AA_H": aa_h,
                  "pair_sa_q": r["pair_sync_async_h"], "pair_aa_q": r["pair_async_async_h"], "rates": rates,
                  "max_run": {"fit_sa": fit_sa, "fit_p3_on_live": fit_p3, "replacements": repl,
                              "replacement_extra_usd": round(extra, 2)},
                  "gke_tpu_live": [v["name"] for v in live if v["kind"] == "gke"], "ok": ok}))
EOF
pj() { $PY -c "import json; d = json.load(open('$W/price.json')); print($1)"; }
log "re-price at $STEPS steps, cap \$$NEW_CAP: median \$$(pj "round(d['usd'][1], 1)"), P90 \$$(pj "round(d['usd'][2], 1)") (P10 \$$(pj "round(d['usd'][0], 1)")); rule: median <= \$$((NEW_CAP - 10)), P90 <= \$$NEW_CAP"
log "  pair hours: sync+async P50 $(pj "round(d['pair_sa_q'][1], 2)") h -> PAIR_SA_H=$(pj "d['PAIR_SA_H']"), async+async P50 $(pj "round(d['pair_aa_q'][1], 2)") h -> PAIR_AA_H=$(pj "d['PAIR_AA_H']"); wall left P50 $(pj "round(d['wall_h_left'][1], 1)") h, P90 $(pj "round(d['wall_h_left'][2], 1)") h"
log "  live: $(pj "', '.join(f\"{v['name']} \${v['usd_h']}/h offset {v['offset_h']} h, max-run left {v['max_run_left_h']} h\" for v in d['rates']['live'])")"
log "  max-run: $(pj "d['max_run']")"
[ "$(pj "d['max_run']['fit_p3_on_live']")" = True ] \
  || log "  note: p3 does not fit the remaining max-run of either live VM after its first pair; that VM self-deletes (auto90 nofit) and a replacement VM (12 h max-run) runs p3 (priced above)"
log "  smallest cap that passes at $STEPS steps: \$$(pj "int(5 * -(-max(d['usd'][1] + 10, d['usd'][2]) // 5))")"
[ "$(pj "d['ok']")" = True ] || die "projected cost fails the rule for $STEPS steps at a \$$NEW_CAP cap"
P_STEPS=$($PY -c "import json; print(json.load(open('$W/plan.json'))['steps'])")
[ "$STEPS" -le "$P_STEPS" ] || die "STEPS $STEPS > plan.json steps $P_STEPS (driver90 refuses that)"
OLD_CAP=$($PY -c "import json; print(json.load(open('$W/plan.json'))['total_cap_usd'])")
[ "$NEW_CAP" -ge "${OLD_CAP%.*}" ] || die "NEW_CAP \$$NEW_CAP is below the current cap \$$OLD_CAP"

SA=$(pj "d['PAIR_SA_H']") AA=$(pj "d['PAIR_AA_H']")
sed -e "s/^STEPS=.*/STEPS=$STEPS/" -e "s/^PAIR_SA_H=.*/PAIR_SA_H=$SA/" -e "s/^PAIR_AA_H=.*/PAIR_AA_H=$AA/" $W/plan.env > $W/plan.new.env
mapfile -t LIVE < <(pj "'\n'.join(v['name'] + ' ' + v['zone'] + ' ' + v['kind'] for v in d['rates']['live'])")
log "plan.env would become: $(tr '\n' ' ' < $W/plan.new.env)"
log "caps would become: total_cap_usd $OLD_CAP -> $NEW_CAP, reprice_median_max_usd -> $((NEW_CAP - 10)) in $HERE/tpu/plan90.json, $LOCAL_PLAN, $B/plan.json, /root/plan.json on: ${LIVE[*]:-none}"
for id in "${IDS[@]}"; do
  log "  $id: claim '${OWNER[$id]:-none}', done ${ISDONE[$id]}$([[ " ${RESTART[*]} " == *" $id "* ]] && echo " -> RESTART at $STEPS steps" || echo " -> picks up plan.env when claimed")"
done

PROBE='for f in /root/qrl-results/pilot/published.env; do [ -f $f ] && echo "has $f: $(tr "\n" " " < $f)"; done
grep "guard start" /root/guard.log | tail -1; pgrep -f "^bash /root/rl/tpu/remote_v2.sh" | wc -l | sed "s/^/work-loop processes: /"
sudo docker ps --format "container {{.Names}} {{.Status}}" 2>/dev/null'
declare -A HASPUB=()
for l in "${LIVE[@]}"; do
  set -- $l; [ "$3" = gce ] || continue
  out=$(ssh_vm "$1" "$2" "$PROBE"); echo "$out" | sed "s/^/  [$1] /"
  echo "$out" | grep -q "^has /root/qrl-results/pilot/published.env" && HASPUB[$1]=1
done
PUBS="${!HASPUB[*]}"
if [ $APPLY = 0 ]; then
  log "DRY RUN: nothing changed. --apply would: raise the caps (above) and wait for each VM guard to log \$$NEW_CAP;" \
      "write published.env on: ${PUBS:-none}; write $B/plan.env; restart pairs: ${RESTART[*]:-none}; append $B/OPERATOR_NOTES.txt"
  exit 0
fi

### --apply
[ -z "$(pj "' '.join(d['gke_tpu_live'])")" ] || die "GKE TPU pod(s) live ($(pj "d['gke_tpu_live']")): their guard's plan can't be patched from here"
note() {
  timeout 60 gcloud storage cat $B/OPERATOR_NOTES.txt > $W/notes 2>/dev/null || : > $W/notes
  echo "$(date -u +%FT%TZ) set_steps90.sh ($(whoami)@$(hostname)): $*" >> $W/notes
  timeout 60 gcloud storage cp $W/notes $B/OPERATOR_NOTES.txt --quiet >/dev/null 2>&1
}
note "APPLY START: STEPS $OLD_STEPS -> $STEPS, total cap \$$OLD_CAP -> \$$NEW_CAP (median cap \$$((NEW_CAP - 10))); re-price median \$$(pj "round(d['usd'][1], 1)") P90 \$$(pj "round(d['usd'][2], 1)") at spend \$$SPENT; restarting: ${RESTART[*]:-none}"

PATCH="import json, sys
for f in sys.argv[1:]:
    try:
        p = json.load(open(f))
    except (OSError, ValueError):
        continue
    if p.get('plan') != 'plan90':
        continue
    p['total_cap_usd'], p['reprice_median_max_usd'] = $NEW_CAP, $((NEW_CAP - 10))
    json.dump(p, open(f + '.tmp', 'w'), indent=1); open(f + '.tmp', 'a').write('\n')
    import os; os.replace(f + '.tmp', f); print('patched', f)"
# 1. caps: GCS first (requester, new VMs), then the local supervisor's copy (its driver republishes it), the repo,
#    and every live VM. The cap only rises, so the order can't trip a guard.
$PY -c "$PATCH" $W/plan.json && timeout 60 gcloud storage cp $W/plan.json $B/plan.json --quiet || die "could not write $B/plan.json"
$PY -c "$PATCH" "$LOCAL_PLAN" "$HERE/tpu/plan90.json"
for l in "${LIVE[@]}"; do
  set -- $l
  ssh_vm "$1" "$2" "python3 -c \"$PATCH\" /root/plan.json /root/rl/plan.json
for i in \$(seq 1 24); do grep 'guard start' /root/guard.log | tail -1 | grep -q 'total cap \\\$$NEW_CAP' && break; sleep 5; done
grep 'guard start' /root/guard.log | tail -1" | sed "s/^/  [$1] /"
done
if [ -f "$LOCAL_PLAN" ]; then
  for i in $(seq 1 24); do grep "guard start" /tmp/qrl-v2/watchdog.log 2>/dev/null | tail -1 | grep -q "total cap \$$NEW_CAP" && break; sleep 5; done
  log "local guard: $(grep "guard start" /tmp/qrl-v2/watchdog.log 2>/dev/null | tail -1)"
fi
# 2. published.env, 3. plan.env
for v in "${!HASPUB[@]}"; do
  z=$(printf '%s\n' "${LIVE[@]}" | awk -v n=$v '$1 == n {print $2}')
  ssh_vm "$v" "$z" "echo $(base64 -w0 < $W/plan.new.env) | base64 -d > /root/qrl-results/pilot/published.env \
&& echo published.env: \$(tr '\n' ' ' < /root/qrl-results/pilot/published.env)" | sed "s/^/  [$v] /"
done
timeout 60 gcloud storage cp $W/plan.new.env $B/plan.env --quiet || die "could not write $B/plan.env (caps already raised)"
log "wrote $B/plan.env: $(tr '\n' ' ' < $W/plan.new.env)"
# 4. restart pairs at the old step count
for id in "${RESTART[@]}"; do
  line=$($PY -c "import json; print(json.load(open('$W/plan.json'))['pairs']['$id'])")
  na=$(echo "${line%%|*}" | awk '{print $1}') nb=$(echo "${line#*|}" | awk '{print $1}')
  set -- ${OWNER[$id]:-}
  vm=${1:-} zone=${2:-}
  if [ "${ISDONE[$id]}" != 1 ] && [ -n "$vm" ] && [[ " ${LIVE[*]} " == *" $vm $zone gce"* ]]; then
    log "$id: stopping the work loop and container on $vm ($na, $nb)"
    ssh_vm "$vm" "$zone" "pkill -f '^bash /root/rl/tpu/remote_v2.sh'; sleep 3; docker rm -f qrl >/dev/null 2>&1
d=/root/qrl-old-steps$OLD_STEPS-\$(date -u +%H%M%S); mkdir -p \$d
for r in $na $nb; do [ -d /root/qrl-results/\$r ] && mv /root/qrl-results/\$r \$d/; rm -rf /dev/shm/qrl/\$r; done
rm -f /root/qrl-results/DONE_B; echo stopped, old results in \$d; docker ps -q | wc -l | sed 's/^/containers left: /'" | sed "s/^/  [$vm] /"
  fi
  for u in claims/$id done/$id failed/$id; do timeout 60 gcloud storage rm "$B/$u" --quiet >/dev/null 2>&1; done
  timeout 60 gcloud storage rm "$B/infra_fail/$id-*" --quiet >/dev/null 2>&1
  log "$id: claim and done/failed/infra_fail markers removed"
  if [ "${ISDONE[$id]}" != 1 ] && [ -n "$vm" ] && [[ " ${LIVE[*]} " == *" $vm $zone gce"* ]]; then
    ssh_vm "$vm" "$zone" "cd /root; setsid env HOME=/root AUTO=1 GKE=0 QRL_DRY=0 QRL_Q=$B QRL_NAME=$vm VM_NAME=$vm VM_ZONE=$zone \
PATH=\$PATH:/snap/bin:/usr/lib/google-cloud-sdk/bin:/usr/local/bin:/opt/google-cloud-sdk/bin \
nohup bash /root/rl/tpu/remote_v2.sh >> /root/qrl-remote.log 2>&1 < /dev/null & sleep 5
echo work loop restarted: \$(pgrep -f '^bash /root/rl/tpu/remote_v2.sh' | wc -l) processes" | sed "s/^/  [$vm] /"
  fi
done
note "APPLY DONE: plan.env STEPS=$STEPS PAIR_SA_H=$SA PAIR_AA_H=$AA; caps \$$NEW_CAP/\$$((NEW_CAP - 10)) in GCS, local supervisor, repo and VMs (${LIVE[*]:-none}); restarted: ${RESTART[*]:-none}. Commit rl/tpu/plan90.json."
log "done. Commit $HERE/tpu/plan90.json. Each restarted VM logs 'claimed pN: ... (steps $STEPS' in its qrl-remote.log within ~5 min."
