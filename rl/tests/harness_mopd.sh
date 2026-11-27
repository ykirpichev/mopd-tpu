#!/bin/bash
# CPU harness for rl/tpu/mopd.sh: runs mopd_loop (the VM work loop of the MOPD plans) through every item of a plan,
# with a fake GCS (a local dir), a fake `docker exec` (runs the command here, like the GKE shim in remote_v2.sh),
# and stub training / eval / teacher / probe processes. qrl.export_teacher, `qrl.evaluate --merge` and the
# teacher-metadata script run for real (tiny model). Needs sudo for the /rl and /results symlinks.
#   bash rl/tests/harness_mopd.sh [plan.json]        -> prints PASS/FAIL lines, exit 0 when all pass
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
PLAN=${1:-$HERE/tpu/mopdpilot.json}
VPY=${VPY:-/tmp/venv/bin/python}
W=$(mktemp -d /tmp/mopd-harness.XXXX)
export HOME=$W/home FG=$W/gcs REAL_PY=$VPY
mkdir -p $HOME $FG
cp -r $HERE $HOME/rl
cp "$PLAN" $HOME/plan.json
echo "harness-sha-0000000" > $HOME/rl/qrl/GIT_SHA
export RES=$HOME/qrl-results; mkdir -p $RES
sudo ln -sfn $HOME/rl /rl; sudo ln -sfn $RES /results
export MODEL=/dev/shm/models/Qwen3.5-2B
sudo rm -rf /dev/shm/qrl /dev/shm/teachers /dev/shm/models; sudo mkdir -p /dev/shm/models /dev/shm/qrl; sudo chown -R $(id -u) /dev/shm/qrl /dev/shm/models
# tiny model in the real checkpoint layout + a learner checkpoint for it
(cd $HERE && $VPY - <<'EOF'
import os, sys, torch
sys.path.insert(0, ".")
from tests.test_cpu import BASE
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained(BASE); t = cfg.text_config
t.num_hidden_layers = 2; t.layer_types = ["linear_attention", "full_attention"]; t.hidden_size, t.intermediate_size = 64, 128
t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 32
t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 2, 16, 16
t.mtp_num_hidden_layers = 0
v = cfg.vision_config; v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 1, 32, 64, 2, 64
m = Qwen3_5ForConditionalGeneration(cfg); m.save_pretrained("/dev/shm/models/Qwen3.5-2B")
AutoTokenizer.from_pretrained(BASE).save_pretrained("/dev/shm/models/Qwen3.5-2B")
from qrl.learner import checkpoint_keys, export_name_map
lm = AutoModelForCausalLM.from_pretrained("/dev/shm/models/Qwen3.5-2B")
named = list(lm.named_parameters()); nm = export_name_map([n for n, _ in named], checkpoint_keys("/dev/shm/models/Qwen3.5-2B"))
from safetensors.torch import save_file
save_file({nm[n]: p.detach().contiguous() for n, p in named if n in nm}, os.environ["HOME"] + "/ckpt.safetensors")
EOF
) >/dev/null 2>&1 || { echo "FAIL: tiny model"; exit 1; }
mkdir -p $W/bin
cat > $W/bin/python <<'EOF'
#!/bin/bash
# stub python: fakes the TPU processes, runs everything else for real
mod=""; [ "${1:-}" = -m ] && mod=$2
arg() { local k=$1; shift; while [ $# -gt 0 ]; do [ "$1" = "$k" ] && { echo "$2"; return; }; shift; done; }
case "$mod" in
  qrl.run)
    rd=$(arg --run-dir "$@"); steps=$(arg --steps "$@"); la=$(arg --learner-args "$@")
    ckd=$(echo "$la" | sed -n 's/.*--ckpt-dir \([^ ]*\).*/\1/p'); cks=$(echo "$la" | sed -n 's/.*--ckpt-steps \([^ ]*\).*/\1/p')
    obj=$(echo "$la" | sed -n 's/.*--objective \([^ ]*\).*/\1/p')
    mkdir -p $rd/events $ckd; echo "{\"argv\": \"$*\", \"rollout_args\": \"$(arg --rollout-args "$@")\", \"learner_args\": \"$la\", \"seed\": 0, \"staleness\": 1, \"staleness_mode\": \"fixed\"}" > $rd/config.json
    echo "$*" >> $HOME/calls.log
    for s in $(seq 0 $((steps - 1))); do echo "{\"proc\": \"learner\", \"kind\": \"train\", \"step\": $s, \"start\": $((1000 + 10 * s)), \"end\": $((1009 + 10 * s)), \"objective\": \"$obj\", \"ce\": 1.5}" >> $rd/events/learner.jsonl; done
    echo '{"proc": "rollout", "kind": "done", "t": 0}' >> $rd/events/rollout.jsonl
    for s in ${cks//,/ }; do cp $HOME/ckpt.safetensors $ckd/$(printf 'step%05d.safetensors' $s); done
    echo log > $rd/learner.log; exit 0 ;;
  qrl.evaluate)
    if echo "$*" | grep -q -- "--merge"; then exec $REAL_PY "$@"; fi
    out=$(arg --out "$@"); ck=$(arg --ckpts "$@"); label=${ck%%=*}; tx=$(arg --texts "$@"); set_=$(arg --eval-set "$@")
    echo "eval $set_ $ck $(arg --model "$@")" >> $HOME/calls.log
    echo "{\"$label\": {\"pass@1\": 0.25, \"samples\": 2, \"problems\": 2, \"eval_s\": 1, \"truncated_frac\": 0, \"per_problem\": [{\"id\": \"a\", \"correct\": 1, \"truncated\": 0, \"lens\": [1, 2]}, {\"id\": \"b\", \"correct\": 0, \"truncated\": 0, \"lens\": [1, 2]}]}}" > $out
    echo "{\"label\": \"$label\", \"id\": \"a\", \"text\": \"t\"}" > $tx; exit 0 ;;
  qrl.teacher|qrl.rollout|qrl.learner)
    rd=$(arg --run-dir "$@"); mkdir -p $rd/events $rd/rollouts
    echo "$mod $*" >> $HOME/calls.log
    case $mod in
      qrl.teacher) echo '{"proc": "teacher", "kind": "score", "tok_s": 5000, "tokens": 10, "tokens_all": 10, "kl_behav_teacher_all": 0.05}' >> $rd/events/teacher.jsonl ;;
      qrl.learner) echo "{\"proc\": \"learner\", \"kind\": \"train\", \"objective\": \"$(arg --objective "$@")\", \"start\": 0, \"end\": 5, \"kl_exact\": 0.01, \"hid_check_max\": 0.0}" >> $rd/events/learner.jsonl ;;
      qrl.rollout) echo b > $rd/rollouts/b000000.pt ;;
    esac
    exit 0 ;;
esac
case "$*" in *snapshot_download*) mkdir -p /dev/shm/models/Qwen3.5-4B /dev/shm/models/Qwen3.5-9B /dev/shm/models/single; exit 0 ;; esac
exec $REAL_PY "$@"
EOF
chmod +x $W/bin/python; ln -sf $W/bin/python $W/bin/python3
cat > $W/bin/gcloud <<'EOF'
#!/bin/bash
# fake gcloud: storage on $FG, instances describe for this VM, everything else succeeds quietly
p() { echo "$FG/${1#gs://}"; }
echo "gcloud $*" >> $HOME/gcloud.log
if [ "$1" = storage ]; then
  sub=$2; shift 2
  q=(); igm=""; rec=0
  for a in "$@"; do case "$a" in --quiet|-q) ;; --if-generation-match=*) igm=${a#*=} ;; -r) rec=1 ;; --format=*) fmt=${a#*=} ;; --temporary-hold) ;; *) q+=("$a") ;; esac; done
  case "$sub" in
    cp) src=${q[0]}; dst=${q[1]}
      if [[ $dst == gs://* ]]; then d=$(p "$dst"); [ "$igm" = 0 ] && [ -e "$d" ] && { echo "precondition failed" >&2; exit 1; }
        [ -n "$(ls $d.HOLD 2>/dev/null)" ] && { echo "held" >&2; exit 1; }
        mkdir -p "$(dirname "$d")"; if [ "$src" = - ]; then cat > "$d"; else cp "$src" "$d"; fi
      else s=$(p "$src"); [ -e "$s" ] || { echo "not found" >&2; exit 1; }
        if [ $rec = 1 ] || [ -d "$s" ]; then mkdir -p "$dst"; cp -r "$s" "$dst/"; else [ -d "$dst" ] && cp "$s" "$dst/" || cp "$s" "$dst"; fi; fi ;;
    cat) for u in "${q[@]}"; do f=$(p "$u"); compgen -G "$f" >/dev/null || { echo "matched no objects" >&2; exit 1; }; cat $f; done ;;
    ls) u=${q[0]}; f=$(p "$u"); f=${f%/}; if [[ $u == *'**' ]]; then find "${f%/\*\*}" -type f ! -name '*.HOLD' | sed "s|^$FG/|gs://|"; exit 0; fi
      compgen -G "$f" >/dev/null || compgen -G "$f/*" >/dev/null || { echo "matched no objects" >&2; exit 1; }
      if [ -d "$f" ]; then ls "$f" | sed "s|^|${u%/}/|"; else echo "$u"; fi ;;
    rm) for u in "${q[@]}"; do f=$(p "$u"); [ -n "$(find ${f%/} -name '*.HOLD' 2>/dev/null)" ] && exit 1; rm -rf "$f"; done ;;
    rsync) exit 0 ;;
    objects) op=${q[0]}; u=${q[1]}; f=$(p "$u")
      case "$op" in
        update) for x in $(find "${f%/\*\*}" -type f ! -name '*.HOLD'); do touch $x.HOLD; done ;;
        list) for x in $(find "${f%/\*\*}" -type f ! -name '*.HOLD'); do [ -e $x.HOLD ] && echo True || echo False; done ;;
        describe) [ -e "$f" ] || exit 1; case "${fmt:-}" in *generation*) echo 1 ;; *update_time*) date -u +%FT%TZ ;; esac ;;
      esac ;;
  esac
  exit 0
fi
if [ "$1 $2 $3" = "compute instances describe" ]; then
  case "$*" in *lastStartTimestamp,scheduling*) echo "$(date -u -d '-1 hour' +%FT%T+00:00) 43200" ;; *) echo "2026-11-24T10:00:00+00:00" ;; esac; exit 0; fi
if [ "$1 $2 $3" = "compute instances delete" ]; then echo "$4" > $HOME/DELETED; exit 0; fi
if [ "$1 $2 $3" = "compute instances list" ]; then echo "[]"; exit 0; fi
exit 0
EOF
chmod +x $W/bin/gcloud
export PATH=$W/bin:$PATH
B=gs://qrl-bc4a-compile-cache-tpu-builders-509006/$(python3 -c "import json; print(json.load(open('$HOME/plan.json'))['plan'])")
gcloud storage cp $HERE/results/plan65k1long/pilot/profile.json $B/pilot/profile.json
PLAN_NAME=$(python3 -c "import json; print(json.load(open('$HOME/plan.json'))['plan'])")
if [ "$PLAN_NAME" = mopdpaper ]; then  # stored, held teachers from an earlier campaign (real export of the tiny model)
  TB=$FG/qrl-bc4a-compile-cache-tpu-builders-509006/teachers/qwen3.5-2b
  (cd $HERE && $VPY - "$TB" <<'EOF'
import json, os, sys, torch
sys.path.insert(0, ".")
from safetensors.torch import load_file, save_file
from qrl import export_teacher
tb, home = sys.argv[1], os.environ["HOME"]
ck = load_file(home + "/ckpt.safetensors")
ev = home + "/tev.json"; json.dump({"final": {"pass@1": 0.5, "samples": 2, "problems": 2, "per_problem": [{"id": "a", "correct": 1}, {"id": "b", "correct": 1}]}}, open(ev, "w"))
for dom, d in (("math", 0.01), ("if", -0.02)):
    p = f"{home}/t{dom}.safetensors"; save_file({k: (v + d * torch.randn_like(v)).contiguous() for k, v in ck.items()}, p)
    m = f"{home}/meta_{dom}.json"; json.dump({"domain": dom, "seed": 0}, open(m, "w"))
    out = f"{tb}/{dom}-grpo-k1is-60steps-s0-20261124-9dfe9c2"
    export_teacher.main(["--base", "/dev/shm/models/Qwen3.5-2B", "--ckpt", p, "--meta", m, "--evals", f"aime={ev}", "--out", out])
EOF
  ) >/dev/null 2>&1 || { echo "FAIL: stored teachers"; exit 1; }
  for f in $(find $TB -type f); do touch $f.HOLD; done
  echo "{\"math\": \"gs://qrl-bc4a-compile-cache-tpu-builders-509006/teachers/qwen3.5-2b/math-grpo-k1is-60steps-s0-20261124-9dfe9c2/\", \"if\": \"gs://qrl-bc4a-compile-cache-tpu-builders-509006/teachers/qwen3.5-2b/if-grpo-k1is-60steps-s0-20261124-9dfe9c2/\"}" \
    | gcloud storage cp - $B/teachers.json
fi
# guard spend: no VMs, no ledger
cat > $HOME/rl/tpu/guard90.py.stub <<'EOF'
EOF
export B VM_NAME=qrl-bc4a-mp1-ew4a VM_ZONE=europe-west4-a GKE=1 HB=$HOME/guard.hb HB_FLAG=$HOME/HB_STALE MOPD_WAIT_S=60 INFRA_FAIL_S=0
touch $HB
( while true; do touch $HB; sleep 20; done ) & HBP=$!
LOGF=$HOME/loop.log
(
  cd $HOME
  MYSTART=2026-11-24T10:00:00+00:00
  log() { echo "$(date -u +%FT%TZ) $*"; }
  stage() { echo "$(date -u +%FT%TZ) STAGE $*" >> $HOME/stages.log; }
  sudo() { "$@"; }
  docker() { [ "$1" = exec ] || return 0; shift; local envs=(); while [ "$1" = -e ]; do envs+=("$2"); shift 2; done; shift
             (cd /rl && env PYTHONPATH=/rl "${envs[@]}" "$@"); }
  X() { docker exec qrl bash -lc "$*"; }
  in_chip() { local chip=$1 port=$2 lg=$3; shift 3; docker exec -e TPU_VISIBLE_CHIPS=$chip qrl bash -lc "cd /rl && $* > $lg 2>&1"; }
  timed() { local s=$1; shift; "$@" >/dev/null 2>&1; stage "$s" $([ $? = 0 ] && echo PASS || echo FAIL); }
  final_sync() { :; }
  waitjobs() { wait; }
  export -f docker sudo
  source $HOME/rl/tpu/auto90.sh
  # guard spend (no ledger in the fake bucket): 0
  sed -i 's|spent = float(subprocess.run(\[sys.executable, os.path.expanduser("~/rl/tpu/guard90.py"), "--plan",|spent = 0.0 if True else float(subprocess.run([sys.executable, os.path.expanduser("~/rl/tpu/guard90.py"), "--plan",|' $HOME/rl/tpu/mopd.sh
  self_delete() { log "self-delete: $*"; echo "$*" > $HOME/SELF_DELETE; exit 0; }
  source $HOME/rl/tpu/mopd.sh
  refresh_code() { log "refresh skipped (harness)"; }
  item_fits() { echo "gate $1: harness"; return 0; }
  mopd_loop
) > $LOGF 2>&1
kill $HBP 2>/dev/null
ok=1
chk() { if eval "$2"; then echo "PASS: $1"; else echo "FAIL: $1"; ok=0; fi; }
for id in $(python3 -c "import json; print(' '.join(json.load(open('$HOME/plan.json'))['items']))"); do
  chk "done/$id" "[ -e $FG/${B#gs://}/done/$id ] && [ ! -e $FG/${B#gs://}/failed/$id ]"
done
TF=$FG/qrl-bc4a-compile-cache-tpu-builders-509006/teachers/qwen3.5-2b
if [ "$PLAN_NAME" = mopdpaper ]; then
chk "teachers untouched and held" "[ \$(ls $TF | wc -l) = 2 ] && [ \$(find $TF -type f ! -name '*.HOLD' | wc -l) = \$(find $TF -name '*.HOLD' | wc -l) ]"
chk "lag 1 fixed everywhere" "[ \$(grep -c 'qrl.run' $HOME/calls.log) = \$(grep 'qrl.run' $HOME/calls.log | grep -c -- '--staleness 1 --staleness-mode fixed') ]"
chk "sft smoke passed once" "[ \$(grep -c 'STAGE sft_smoke PASS' $HOME/stages.log) = 1 ] && [ -e $FG/${B#gs://}/done/sft_smoke ]"
chk "sft runs: teacher-gen, no teacher service, seeds and chips" "for x in 's0 .*--seed 0 .*--rollout-chip 0 --learner-chip 1' 's1 .*--seed 1 .*--rollout-chip 0 --learner-chip 1' 's2 .*--seed 2 .*--rollout-chip 2 --learner-chip 3'; do grep \"run-dir /dev/shm/qrl/sft-\$x\" $HOME/calls.log | grep -- '--teacher-gen math=/dev/shm/teachers/math,if=/dev/shm/teachers/if' | grep -- '--objective sft' | grep -q -- '--steps 120' || exit 1; done; ! grep 'qrl/sft-' $HOME/calls.log | grep -q -- --teacher-chips"
chk "sft evals 20/40/60 on the generator chip, 120 on 4 chips" "for r in sft-s0 sft-s1 sft-s2; do for s in 20 40 60 120; do for k in eval eval_ifeval eval_ifbench; do [ -s $RES/\$r/\$k.step\$s.json ] || { echo missing \$r \$k \$s; exit 1; }; done; done; done"
chk "four merges built for real and evaluated" "for m in m1-linear m2-ta m3-ties20 m4-ties50; do python3 -c \"import json; j = json.load(open('$RES/merge-\$m/merge.json')); assert j['tensors_merged'] > 0\" && [ -s $RES/merge-\$m/eval.final.json ] && [ -s $RES/merge-\$m/eval_ifbench.final.json ] || exit 1; done; grep -q 'eval v2 final=base /dev/shm/merges/m3-ties20' $HOME/calls.log"
chk "reference re-evals with new seeds" "for sd in 2234 3234; do [ -s $RES/refs-tmath/eval.seed\$sd.json ] && [ -s $RES/refs-tif/eval_ifeval.seed\$sd.json ] && [ -s $RES/refs-tif/eval_ifbench.seed\$sd.json ] && [ -s $RES/refs-base/eval.seed\$sd.json ] || exit 1; done"
chk "new seeds for the main arms" "for r in mopd-s2 mopd-s3 mopd-dist-s2 mopd-dist-s3 mixrl-s2 mixrl-s3 opd-tmath-s2; do [ -s $RES/\$r/eval.step60.json ] || { echo missing \$r; exit 1; }; done; grep -q 'mixrl-s3 .*--seed 3 .*--rollout-chip 2' $HOME/calls.log && grep -q 'mopd-dist-s3 .*--seed 3 .*--ship-hidden all' $HOME/calls.log"
else
chk "two teachers with COMPLETE" "[ \$(ls -d $TF/*/COMPLETE 2>/dev/null | wc -l) = 2 ]"
chk "every teacher object held" "[ \$(find $TF -type f ! -name '*.HOLD' | wc -l) = \$(find $TF -name '*.HOLD' | wc -l) ]"
if [ "$(python3 -c "import json; print(json.load(open('$HOME/plan.json'))['plan'])")" = mopdfull ]; then
chk "teacher names" "ls $TF | grep -q '^math-grpo-k1is-60steps-s0-[0-9]\{8\}-harness$' && ls $TF | grep -q '^if-grpo-k1is-60steps-s0-[0-9]\{8\}-harness$'"
chk "teachers.json" "python3 -c \"import json; d = json.load(open('$FG/${B#gs://}/teachers.json')); assert d['math'].startswith('gs://') and d['if'].startswith('gs://')\""
chk "lag 1 fixed everywhere" "[ \$(grep -c 'qrl.run' $HOME/calls.log) = \$(grep 'qrl.run' $HOME/calls.log | grep -c -- '--staleness 1 --staleness-mode fixed') ]"
chk "no smoke in the full plan" "! grep -q 'STAGE smoke' $HOME/stages.log"
chk "mixrl pair seeds" "grep -q 'mixrl-s0 .*--seed 0 .*--rollout-chip 0 --learner-chip 1' $HOME/calls.log && grep -q 'mixrl-s1 .*--seed 1 .*--rollout-chip 2 --learner-chip 3' $HOME/calls.log"
if [ "$(python3 -c "import json; print(json.load(open('$HOME/plan.json')).get('single_teacher'))")" = tmath ]; then
  O1=opd-tmath
  chk "single T-math teacher on chip 2" "[ \$(grep -c 'opd-tmath-s[01] .*--teacher-chips 2 --teacher-args .*--teachers all=/dev/shm/teachers/math' $HOME/calls.log) = 2 ]"
else
  O1=opd9b
  chk "opd9b on two teacher chips" "[ \$(grep -c 'opd9b-s[01] .*--teacher-chips 2,3 --teacher-args .*--teachers all=/dev/shm/models/single.*--mb-tokens 4096' $HOME/calls.log) = 2 ]"
fi
chk "four MOPD runs" "for r in mopd-s0 mopd-s1 mopd-dist-s0 mopd-dist-s1; do grep -q \"run-dir /dev/shm/qrl/\$r \" $HOME/calls.log || exit 1; done"
chk "evals at 20/40/60 for every arm" "for r in mixrl-s0 mixrl-s1 mopd-s0 mopd-s1 mopd-dist-s0 mopd-dist-s1 $O1-s0 $O1-s1; do for s in 20 40 60; do for k in eval eval_ifeval eval_ifbench; do [ -s $RES/\$r/\$k.step\$s.json ] || { echo missing \$r \$k \$s; exit 1; }; done; done; done"
chk "step-20 MOPD evals on chip 3" "grep -q 'eval v2 step20=/results/mopd-s0/ckpt/step00020.safetensors' $HOME/calls.log"
else
chk "teacher names" "ls $TF | grep -q '^math-grpo-k1is-30steps-s0-[0-9]\{8\}-harness$' && ls $TF | grep -q '^if-grpo-k1is-30steps-s0-[0-9]\{8\}-harness$'"
chk "teacher.json has evals and provenance" "python3 -c \"import json, glob; t = json.load(open(glob.glob('$TF/math-*/teacher.json')[0])); assert t['eval_scores']['aime']['final']['pass@1'] == 0.25 and t['provenance']['git_sha'] and t['steps'] == 30\""
chk "teachers.json" "python3 -c \"import json; d = json.load(open('$FG/${B#gs://}/teachers.json')); assert d['math'].startswith('gs://') and d['if'].startswith('gs://')\""
chk "teachers fetched + verified" "[ -f /dev/shm/teachers/math/COMPLETE ] && [ -f /dev/shm/teachers/if/COMPLETE ]"
chk "mopd run args" "grep -q 'qrl.run --run-dir /dev/shm/qrl/mopd-s0 .*--teacher-chips 2 --teacher-args .*--teachers math=/dev/shm/teachers/math,if=/dev/shm/teachers/if.*--ship-hidden diag' $HOME/calls.log"
chk "mopd-dist run args" "grep -q 'mopd-dist-s0.*--objective opd_dist.*--ship-hidden all' $HOME/calls.log"
chk "lag 1 fixed everywhere" "[ \$(grep -c 'qrl.run' $HOME/calls.log) = \$(grep 'qrl.run' $HOME/calls.log | grep -c -- '--staleness 1 --staleness-mode fixed') ]"
chk "teachers domains" "grep -q 'tmath.*--domains math:16' $HOME/calls.log && grep -q 'tif.*--domains if:16' $HOME/calls.log"
chk "mixrl domains" "grep -q 'mixrl-s0.*--domains math:8,if:8.*--objective grpo' $HOME/calls.log"
chk "evals merged" "for r in tmath tif mixrl-s0 mopd-s0 mopd-dist-s0; do ls $RES/\$r/eval*.json | grep -v '\\.s[0-9]' | grep -q . || exit 1; done"
chk "final evals per arm" "[ -s $RES/mixrl-s0/eval.step30.json ] && [ -s $RES/mopd-s0/eval_ifbench.step30.json ] && [ -s $RES/mopd-dist-s0/eval_ifeval.step30.json ]"
chk "roundtrip evals" "[ -s $RES/roundtrip/eval.tmath.json ] && [ -s $RES/roundtrip/eval_ifeval.tif.json ]"
chk "probe ran 2B/4B/9B" "[ \$(grep -c 'qrl.teacher.*--teachers all=' $HOME/calls.log) = 3 ]"
chk "smoke stage" "grep -q 'STAGE smoke PASS' $HOME/stages.log"
fi
fi
chk "self-delete at the end" "grep -q 'all work items done' $HOME/SELF_DELETE"
[ $ok = 1 ] || { echo "--- loop log ---"; tail -40 $LOGF; }
echo "harness dir $W"
sudo rm -f /rl /results
[ $ok = 1 ]
