# Sourced by remote_v2.sh (AUTO=1) when plan.json has "items": the MOPD plans (docs/mopd-tpu-plan.md).
# Reuses auto90.sh (claims with takeover, heartbeat gate, self-delete, infra_fail, remaining_h, record_waste).
# Work items come from plan.json "items" ({"deps": [...], "h90": P90 hours}); a VM claims the first unclaimed item
# whose deps are all done/ in plan order, waits up to MOPD_WAIT_S when only items with unfinished deps are left
# (another VM is producing them), and self-deletes when nothing is left for it. Before every claim it refreshes
# qrl/ and plan/ from $B/code.tgz, so Python fixes reach a running VM between items. $B/HOLD pauses new claims.
# Chips: teachers 0/1 T-math, 2/3 T-IF; mixrl 0/1; distillation 0 student vLLM, 1 learner, 2 teacher service,
# 3 evals (OPD-9B: 2/3 both score). Teachers are exported to $TB/<name>/ (HF dir, teacher.json, SHA256SUMS,
# COMPLETE last), uploaded with --if-generation-match=0, then held (gcs_protect.py hold).
MOPD_WAIT_S=${MOPD_WAIT_S:-1800}; HOLD_MAX_S=${HOLD_MAX_S:-2700}
TB=gs://qrl-bc4a-compile-cache-tpu-builders-509006/teachers/qwen3.5-2b
pj() { python3 -c "import json; p = json.load(open('$HOME/plan.json')); print($1)"; }
STEPS=$(pj "p['steps']"); TSTEPS=$(pj "p['teacher_steps']"); CAP=$(pj "p['cap']"); LEN=$((CAP + 1536))
CKPTS=$(pj "p.get('ckpt_steps', p['steps'])"); TCKPTS=$(pj "p.get('teacher_ckpt_steps', p['teacher_steps'])")
LFLAGS=$(pj "p.get('learner_flags', '--gdn-chunk 256 --mb-tokens 8192')")
TFLAGS=$(pj "p.get('teacher_flags', '--gdn-chunk 256 --mb-tokens 8192')")
SEVAL=$(pj "p.get('eval_samples', {'aime': 16, 'if': 4})['aime']"); SEVAL_IF=$(pj "p.get('eval_samples', {'aime': 16, 'if': 4})['if']")
DOMS=$(pj "p.get('domains', 'math:8,if:8')")
FILTER="--train-filter /results/pilot/profile.json --keep-zero-frac 0.2"
TDIR=/dev/shm/teachers
SINGLE=$(pj "p.get('single_teacher', 'Qwen/Qwen3.5-9B')")
OPD1=$([ "$SINGLE" = tmath ] && echo opd-tmath || echo opd9b)
SMOKE=$(pj "int(p.get('smoke', True))")

refresh_code() {  # python code only: the bash scripts of this running loop stay as they are
  local t=$HOME/code.new
  rm -rf $t; mkdir -p $t
  timeout 120 gcloud storage cp "$B/code.tgz" $t/code.tgz --quiet >/dev/null 2>&1 && tar xzf $t/code.tgz -C $t 2>/dev/null \
    && [ -d $t/qrl ] && [ -d $t/plan ] || { log "code refresh failed; keeping the current code"; return 0; }
  rm -rf ~/rl/qrl.old ~/rl/plan.old; mv ~/rl/qrl ~/rl/qrl.old; mv $t/qrl ~/rl/qrl; mv ~/rl/plan ~/rl/plan.old; mv $t/plan ~/rl/plan
  log "code refreshed: $(cat ~/rl/qrl/GIT_SHA 2>/dev/null | cut -c1-12)"
}

mopd_setup() {
  timed if_deps X "pip install -q langdetect nltk immutabledict emoji syllapy absl-py 2>&1 | tail -1; python -c 'import langdetect, nltk, emoji, syllapy, immutabledict; print(\"if deps ok\")'"
  timed if_datasets X "cd /rl && python -c 'from qrl import ifeval; ifeval.setup_nltk(); print(len(ifeval.load_train()), len(ifeval.load_eval(\"ifeval\")), len(ifeval.load_eval(\"ifbench\")), \"if datasets ok\")'"
  mkdir -p "$RES/pilot"
  [ -s "$RES/pilot/profile.json" ] || timeout 300 gcloud storage cp "$B/pilot/profile.json" "$RES/pilot/profile.json" --quiet
  [ -s "$RES/pilot/profile.json" ] || { log "no math prompt profile at $B/pilot/profile.json"; return 1; }
}

hf_get() {  # repo local_dir
  X "python -c \"from huggingface_hub import snapshot_download as s; s('$1', local_dir='$2', max_workers=16)\" >/dev/null && du -sh $2"
}

# mrun NAME ROLLOUT_CHIP LEARNER_CHIP TEACHER_CHIPS OBJECTIVE DOMAINS SEED STEPS CKPT_STEPS PORT [TEACHERS] [TEACHER_EXTRA]
#      [ROLLOUT_EXTRA]
mrun() {
  local name=$1 rc=$2 lc=$3 tch=$4 obj=$5 doms=$6 seed=$7 steps=$8 ck=$9 port=${10} teachers=${11:-} textra=${12:-}
  local rextra=${13:-}
  mkdir -p "$RES/$name"; rm -f "$RES/$name/.train_done"
  stage train_start PASS "$name $obj $doms seed $seed steps $steps"
  local roll="--domains $doms --group-size 8 --max-tokens $CAP --max-model-len $LEN $FILTER --temperature 1.0 $rextra"
  local larg="--max-model-len $LEN --grad-ckpt --lr 1e-6 --log-entropy --behavior-logprobs use --objective $obj \
--ckpt-dir /results/$name/ckpt --ckpt-steps $ck $LFLAGS"
  local tpart=""
  [ -n "$tch" ] && tpart="--teacher-chips $tch --teacher-args '--teachers $teachers --max-model-len $LEN $TFLAGS $textra'"
  log "train $name: $obj $doms seed $seed steps $steps chips r$rc l$lc t$tch"
  sudo docker exec qrl bash -lc "rm -rf /dev/shm/qrl/$name; cd /rl && python -m qrl.run --run-dir /dev/shm/qrl/$name \
    --model $MODEL --mode async --staleness 1 --staleness-mode fixed --steps $steps --seed $seed \
    --rollout-chip $rc --learner-chip $lc --port-base $port --rollout-args '$roll' --learner-args '$larg' $tpart \
    > /results/$name/run.out 2>&1"
  local rc_=$?
  touch "$RES/$name/.train_done"
  sudo cp -r /dev/shm/qrl/$name/events /dev/shm/qrl/$name/config.json "$RES/$name/" 2>/dev/null
  sudo cp /dev/shm/qrl/$name/*.log "$RES/$name/" 2>/dev/null
  sudo rm -rf /dev/shm/qrl/$name/weights /dev/shm/qrl/$name/teacher_heads /dev/shm/qrl/$name/scored
  stage train_end $([ $rc_ = 0 ] && echo PASS || echo "FAIL rc=$rc_") "$name"
  return $rc_
}

# meval RUN LABEL MODEL_DIR CKPT(base|path) CHIPS(csv) SET(v2|ifeval|ifbench) SAMPLES; MEVAL_SEED=N reseeds sampling
meval() {
  local run=$1 label=$2 model=$3 ck=$4 chips=(${5//,/ }) set=$6 smp=$7 i pids=() parts=""
  local f; f=$([ "$set" = v2 ] && echo eval || echo "eval_$set")
  local o=/results/$run/$f.$label n=${#chips[@]}
  mkdir -p "$RES/$run"
  [ -s "$RES/$run/$f.$label.json" ] && return 0
  for i in $(seq 0 $((n - 1))); do
    in_chip ${chips[$i]} $((8700 + ${chips[$i]})) $o.s$i.log ${MEVAL_TIMEOUT:+timeout $MEVAL_TIMEOUT} python -m qrl.evaluate --model $model --eval-set $set \
      --samples $smp --max-tokens $CAP --max-model-len $LEN --ckpts "$label=$ck" --shard-index $i --num-shards $n ${MEVAL_SEED:+--seed $MEVAL_SEED} \
      --out $o.s$i.json --texts $o.s$i.texts.jsonl &
    pids+=($!); parts="$parts $o.s$i.json"
  done
  wait "${pids[@]}"
  X "cd /rl && python -m qrl.evaluate --merge $parts --out $o.json && cat $(for i in $(seq 0 $((n - 1))); do printf '%s.s%d.texts.jsonl ' $o $i; done) > $o.texts.jsonl" \
    > "$RES/$run/$f.$label.merge.log" 2>&1
  [ -s "$RES/$run/$f.$label.json" ] && stage "eval_${run}_${set}_$label" PASS "$(python3 -c "import json; r = json.load(open('$RES/$run/$f.$label.json'))['$label']; print(round(r['pass@1'], 4), r['problems'], r['samples'])")" \
    || stage "eval_${run}_${set}_$label" FAIL
}

eval_all() {  # RUN LABEL MODEL CKPT CHIPS: AIME v2, IFEval, IFBench
  meval "$1" "$2" "$3" "$4" "$5" v2 $SEVAL
  meval "$1" "$2" "$3" "$4" "$5" ifeval $SEVAL_IF
  meval "$1" "$2" "$3" "$4" "$5" ifbench $SEVAL_IF
}

ck() { printf '/results/%s/ckpt/step%05d.safetensors' "$1" "$2"; }

smoke() {  # bring-up on 2 free chips with the base model as both teachers: rollout -> teacher (hidden) -> learner
  local d=/dev/shm/qrl/smoke c1=$1 c2=$2
  mkdir -p "$RES/smoke"; log "smoke: rollout chip $c1, teacher chip $c2"
  sudo docker exec qrl bash -lc "rm -rf $d; mkdir -p $d"
  in_chip $c1 8751 /results/smoke/rollout.log python -m qrl.rollout --run-dir $d --model $MODEL --steps 2 --staleness 99 \
    --domains math:2,if:2 --group-size 4 --max-tokens $CAP --max-model-len $LEN $FILTER --seed 5 &
  local rp=$!
  in_chip $c2 8752 /results/smoke/teacher.log python -m qrl.teacher --run-dir $d --model $MODEL \
    --teachers math=$MODEL,if=$MODEL --steps 2 --max-model-len $LEN $TFLAGS --ship-hidden all &
  local tp=$!
  wait $rp
  in_chip $c1 8753 /results/smoke/learner_dist.log python -m qrl.learner --run-dir $d --model $MODEL --steps 1 \
    --objective opd_dist --grad-ckpt --max-model-len $LEN $LFLAGS --log-entropy --keep-weights 3
  local r1=$?
  wait $tp
  in_chip $c1 8754 /results/smoke/learner_opd.log python -m qrl.learner --run-dir $d --model $MODEL --first-step 1 \
    --steps 2 --objective opd --grad-ckpt --max-model-len $LEN $LFLAGS --log-entropy --keep-weights 3
  local r2=$?
  sudo cp -r $d/events "$RES/smoke/" 2>/dev/null; sudo rm -rf $d
  local s; s=$(python3 - "$RES/smoke/events" <<'EOF'
import json, os, sys
ev = []
for f in os.listdir(sys.argv[1]):
    ev += [json.loads(x) for x in open(os.path.join(sys.argv[1], f))]
tr = {e["objective"]: e for e in ev if e["kind"] == "train"}
sc = [e for e in ev if e["kind"] == "score"]
d, o = tr.get("opd_dist", {}), tr.get("opd", {})
print(f"dist train {d.get('end', 0) - d.get('start', 0):.0f}s kl_exact {d.get('kl_exact', -1):.2e} "
      f"hid_check max {d.get('hid_check_max', -1):.2e} mean {d.get('hid_check_abs', -1):.2e} | "
      f"opd train {o.get('end', 0) - o.get('start', 0):.0f}s kl {o.get('kl_teacher', -1):.2e} diag {o.get('diag_kl_exact', -1):.2e} | "
      f"teacher {sum(e['tok_s'] for e in sc) / max(1, len(sc)):.0f} tok/s")
EOF
)
  stage smoke $([ $r1 = 0 ] && [ $r2 = 0 ] && echo PASS || echo "FAIL rc=$r1/$r2") "$s"
  [ $r1 = 0 ] && [ $r2 = 0 ]
}

teacher_meta() {  # DOMAIN RUN OUT_JSON T_START T_END
  X "cd /rl && python - $1 $2 $3 $4 $5 '$VM_NAME' '$VM_ZONE' '$TSTEPS' <<'EOF'
import datetime as dt, json, sys
import torch, transformers
dom, run, out, t0, t1, vm, zone, steps = sys.argv[1:9]
def ver(m):
    try:
        return __import__(m).__version__
    except Exception:
        return None
def rev(kind, rid):
    try:
        from huggingface_hub import dataset_info, model_info
        return (model_info if kind == 'model' else dataset_info)(rid).sha
    except Exception as e:
        return f'unknown ({type(e).__name__})'
cfg = json.load(open(f'/results/{run}/config.json'))
sha = open('/rl/qrl/GIT_SHA').read().strip() if __import__('os').path.exists('/rl/qrl/GIT_SHA') else None
ds = {'math': ['gneubig/aime-1983-2024'], 'if': ['allenai/RLVR-IFeval']}[dom]
meta = {'domain': dom, 'student_init': 'qwen3.5-2b', 'base_model': 'Qwen/Qwen3.5-2B', 'base_revision': rev('model', 'Qwen/Qwen3.5-2B'),
        'recipe': 'grpo-k1is: async lag 1 (fixed), truncated IS cap 2, GRPO group advantage, lr 1e-6, 16 prompts x 8',
        'recipe_flags': {'rollout_args': cfg.get('rollout_args'), 'learner_args': cfg.get('learner_args'),
                         'staleness': cfg.get('staleness'), 'staleness_mode': cfg.get('staleness_mode')},
        'chat_template': 'non-thinking (enable_thinking=False)', 'seed': cfg.get('seed'), 'steps': int(steps),
        'data': {'datasets': {d: rev('dataset', d) for d in ds},
                 'prompt_filter': 'math: base-model profile (qrl.profile_prompts), keep 0<c<n plus 20% of 0/n' if dom == 'math' else 'none',
                 'prompts_seen': int(steps) * 16},
        'provenance': {'git_sha': sha, 'image': 'vllm/vllm-torchtpu:v0.32.0', 'torch': torch.__version__,
                       'torch_tpu': ver('torch_tpu'), 'vllm': ver('vllm'), 'transformers': transformers.__version__,
                       'vm': vm, 'zone': zone,
                       'train_start_utc': dt.datetime.fromtimestamp(float(t0), dt.timezone.utc).isoformat(),
                       'train_end_utc': dt.datetime.fromtimestamp(float(t1), dt.timezone.utc).isoformat()}}
json.dump(meta, open(out, 'w'), indent=1)
EOF"
}

export_upload() {  # DOMAIN RUN -> prints the teacher's gs:// url
  local dom=$1 run=$2 sha name d url f rel
  sha=$(cut -c1-7 ~/rl/qrl/GIT_SHA 2>/dev/null || echo nosha00)
  name=$dom-grpo-k1is-${TSTEPS}steps-s0-$(date -u +%Y%m%d)-$sha
  d=/results/teachers/$name; url=$TB/$name
  local t0 t1; read -r t0 t1 < <(python3 -c "
import json; ev = [json.loads(x) for x in open('$RES/$run/events/learner.jsonl')]
tr = [e for e in ev if e['kind'] == 'train']; print(min(e['start'] for e in tr), max(e['end'] for e in tr))")
  teacher_meta $dom $run /results/$run/teacher_meta.json "$t0" "$t1" >> "$RES/$run/export.log" 2>&1
  X "cd /rl && rm -rf $d && python -m qrl.export_teacher --base $MODEL --ckpt $(ck $run $TSTEPS) --run-dir /results/$run \
    --meta /results/$run/teacher_meta.json --evals aime=/results/$run/eval.final.json ifeval=/results/$run/eval_ifeval.final.json \
    ifbench=/results/$run/eval_ifbench.final.json --out $d && python -m qrl.export_teacher --verify $d" >> "$RES/$run/export.log" 2>&1 \
    || { log >&2 "export $dom failed: $(tail -2 $RES/$run/export.log)"; return 1; }
  sudo chown -R "$(id -u)" "$RES/teachers" 2>/dev/null
  for f in $(cd "$RES/teachers/$name" && find . -type f ! -name COMPLETE | sed 's|^\./||' | sort); do
    for i in 1 2 3; do
      timeout 600 gcloud storage cp "$RES/teachers/$name/$f" "$url/$f" --if-generation-match=0 --quiet >> "$RES/$run/export.log" 2>&1 && break
      gsx "$url/$f" && break
      sleep 10
    done
    gsx "$url/$f" || { log >&2 "upload of $url/$f failed"; return 1; }
  done
  timeout 120 gcloud storage cp "$RES/teachers/$name/COMPLETE" "$url/COMPLETE" --if-generation-match=0 --quiet >> "$RES/$run/export.log" 2>&1
  gsx "$url/COMPLETE" || { log >&2 "COMPLETE upload failed for $url"; return 1; }
  timeout 600 python3 ~/rl/tpu/gcs_protect.py hold "$url/" >> "$RES/$run/export.log" 2>&1 || { log >&2 "hold failed for $url"; return 1; }
  local nh; nh=$(timeout 300 gcloud storage objects list "$url/**" --format='value(temporary_hold)' 2>/dev/null | grep -ci true)
  local nf; nf=$(timeout 300 gcloud storage ls "$url/**" 2>/dev/null | wc -l)
  stage teacher_export PASS "$url files $nf held $nh"
  [ "$nh" = "$nf" ] || log >&2 "WARNING: $nh of $nf objects held under $url"
  echo "$url"
}

fetch_teachers() {  # -> /dev/shm/teachers/{math,if} from $B/teachers.json, checksums verified
  local js; js=$(timeout 60 gcloud storage cat "$B/teachers.json" 2>/dev/null) || { log "no $B/teachers.json"; return 1; }
  local dom url
  for dom in math if; do
    url=$(echo "$js" | python3 -c "import json, sys; print(json.load(sys.stdin)['$dom'].rstrip('/'))")
    [ -f $TDIR/$dom/COMPLETE ] && continue
    gsx "$url/COMPLETE" || { log "$url has no COMPLETE"; return 1; }
    sudo rm -rf $TDIR/$dom.tmp; mkdir -p $TDIR
    timeout 1200 gcloud storage cp -r "$url" $TDIR/ --quiet >/dev/null 2>&1 || { log "download $url failed"; return 1; }
    sudo mv $TDIR/$(basename $url) $TDIR/$dom
    X "cd /rl && python -m qrl.export_teacher --verify $TDIR/$dom" || { log "checksum failure in $url"; sudo rm -rf $TDIR/$dom; return 1; }
  done
  stage teachers_fetched PASS "$(echo "$js" | tr -d '\n' | cut -c1-300)"
}

job_teachers() {
  local pids=()
  ( [ "$SMOKE" = 0 ] || smoke 2 3 || { echo "$ME smoke failed (see $RES/smoke)" | timeout 60 gcloud storage cp - "$B/HOLD" --quiet
                   log "smoke FAILED: set $B/HOLD (no distillation item starts until it is removed)"; }
    mrun tif 2 3 "" grpo if:16 0 $TSTEPS $TCKPTS 8486 || log "T-IF training failed"
    meval tif final $MODEL "$(ck tif $TSTEPS)" 2,3 ifeval $SEVAL_IF
    meval tif final $MODEL "$(ck tif $TSTEPS)" 2,3 ifbench $SEVAL_IF
    meval tif final $MODEL "$(ck tif $TSTEPS)" 2,3 v2 $SEVAL ) &
  pids+=($!)
  mrun tmath 0 1 "" grpo math:16 0 $TSTEPS $TCKPTS 8476 || log "T-math training failed"
  meval tmath final $MODEL "$(ck tmath $TSTEPS)" 0,1 v2 $SEVAL
  meval tmath final $MODEL "$(ck tmath $TSTEPS)" 0,1 ifeval $SEVAL_IF
  meval tmath final $MODEL "$(ck tmath $TSTEPS)" 0,1 ifbench $SEVAL_IF
  wait "${pids[@]}"
  final_sync
  local um ui
  [ -s "$RES/tmath/eval.final.json" ] && [ -s "$RES/tif/eval_ifeval.final.json" ] || { log "teacher evals missing"; return 1; }
  um=$(export_upload math tmath | tail -1); ui=$(export_upload if tif | tail -1)
  [[ $um == gs://* ]] && [[ $ui == gs://* ]] || { log "teacher export failed (math '$um', if '$ui')"; return 1; }
  echo "{\"math\": \"$um/\", \"if\": \"$ui/\"}" | timeout 60 gcloud storage cp - "$B/teachers.json" --quiet
  final_sync
}

probe() {  # chips c_gen c_score: base student samples, scored by 2B (self), 4B and 9B; prefill tok/s and starting KL
  local cg=$1 cs=$2 d=/dev/shm/qrl/probe m
  mkdir -p "$RES/probe"
  hf_get Qwen/Qwen3.5-4B /dev/shm/models/Qwen3.5-4B > "$RES/probe/dl4b.log" 2>&1 &
  local p4=$!
  hf_get Qwen/Qwen3.5-9B /dev/shm/models/Qwen3.5-9B > "$RES/probe/dl9b.log" 2>&1 &
  local p9=$!
  sudo docker exec qrl bash -lc "rm -rf $d; mkdir -p $d"
  in_chip $cg 8761 /results/probe/rollout.log python -m qrl.rollout --run-dir $d --model $MODEL --steps 2 --staleness 99 \
    --domains $DOMS --group-size 8 --max-tokens $CAP --max-model-len $LEN $FILTER --seed 0
  wait $p4 $p9
  for m in 2B:$MODEL 4B:/dev/shm/models/Qwen3.5-4B 9B:/dev/shm/models/Qwen3.5-9B; do
    local tag=${m%%:*} path=${m#*:}
    sudo docker exec qrl bash -lc "rm -rf $d-$tag; mkdir -p $d-$tag && cp -r $d/rollouts $d-$tag/"
    in_chip $cs 8762 /results/probe/teacher_$tag.log python -m qrl.teacher --run-dir $d-$tag --model $MODEL \
      --teachers all=$path --steps 2 --max-model-len $LEN $([ $tag = 9B ] && echo "--gdn-chunk 256 --mb-tokens 4096" || echo "$TFLAGS")
    sudo cp $d-$tag/events/teacher.jsonl "$RES/probe/score_$tag.jsonl" 2>/dev/null; sudo rm -rf $d-$tag
  done
  sudo rm -rf $d
  stage probe_kl PASS "$(python3 - "$RES/probe" <<'EOF'
import json, os, sys
out = []
for tag in ("2B", "4B", "9B"):
    p = os.path.join(sys.argv[1], f"score_{tag}.jsonl")
    if not os.path.exists(p):
        out.append(f"{tag}: missing"); continue
    sc = [json.loads(x) for x in open(p) if '"score"' in x]
    kl = sum(e["kl_behav_teacher_all"] * e["tokens_all"] for e in sc) / max(1, sum(e["tokens_all"] for e in sc))
    out.append(f"{tag}: KL {kl:.4f} {sum(e['tok_s'] for e in sc) / max(1, len(sc)):.0f} tok/s")
print("; ".join(out))
EOF
)"
  MEVAL_TIMEOUT=3600 meval base9b final /dev/shm/models/Qwen3.5-9B base $cs v2 4
  MEVAL_TIMEOUT=1200 meval base9b final /dev/shm/models/Qwen3.5-9B base $cs ifeval 1
}

job_probe_mixrl() {
  local pids=()
  ( mrun mixrl-s0 0 1 "" grpo $DOMS 0 $STEPS $CKPTS 8476 || log "Mix-RL training failed" ) &
  pids+=($!)
  ( probe 2 3; meval base final $MODEL base 2,3 ifeval $SEVAL_IF; meval base final $MODEL base 2,3 ifbench $SEVAL_IF ) &
  pids+=($!)
  wait "${pids[0]}"
  wait "${pids[@]:1}"
  local s
  for s in ${CKPTS//,/ }; do eval_all mixrl-s0 step$s $MODEL "$(ck mixrl-s0 $s)" 0,1,2,3; done
  final_sync
  [ -s "$RES/mixrl-s0/eval.step$STEPS.json" ]
}

roundtrip() {  # gate 6 on CHIP: the downloaded teacher dirs load in vLLM and reproduce their logged evals
  local c=$1
  meval roundtrip tif $TDIR/if base $c ifeval $SEVAL_IF
  meval roundtrip tmath $TDIR/math base $c v2 $SEVAL
}

ckpt_wait() {  # RUN STEP -> 0 once the checkpoint exists, 1 when the training ended without it
  local f=$RES/$1/ckpt/$(printf 'step%05d' $2).safetensors
  until [ -s "$f" ]; do [ -e "$RES/$1/.train_done" ] && return 1; sleep 30; done
}

evals_during() {  # RUN CHIP: every checkpoint before the last, evaluated on CHIP while RUN trains
  local s
  for s in ${CKPTS//,/ }; do
    [ "$s" = "$STEPS" ] && continue
    ckpt_wait $1 $s || return 0
    eval_all $1 step$s $MODEL "$(ck $1 $s)" $2
  done
}

job_distill() {  # NAME OBJECTIVE SEED SHIP_HIDDEN
  local name=$1 obj=$2 seed=$3 ship=$4 pids=()
  fetch_teachers || return 1
  ( if ! gsx "$B/done/roundtrip"; then
      roundtrip 3; [ -s "$RES/roundtrip/eval.tmath.json" ] && echo "$ME" | timeout 60 gcloud storage cp - "$B/done/roundtrip" --quiet
    fi
    evals_during $name 3 ) &
  pids+=($!)
  mrun $name 0 1 2 $obj $DOMS $seed $STEPS $CKPTS 8476 "math=$TDIR/math,if=$TDIR/if" "--ship-hidden $ship" \
    || log "$name training failed"
  wait "${pids[@]}"
  local s
  for s in ${CKPTS//,/ }; do eval_all $name step$s $MODEL "$(ck $name $s)" 0,1,2,3; done
  final_sync
  [ -s "$RES/$name/eval.step$STEPS.json" ]
}

job_mixrl_pair() {  # [SEED_A SEED_B]: Mix-RL seed A (chips 0/1) and B (chips 2/3), then every checkpoint on 4 chips
  local a=${1:-0} b=${2:-1} p s r
  ( mrun mixrl-s$b 2 3 "" grpo $DOMS $b $STEPS $CKPTS 8486 || log "Mix-RL s$b training failed" ) &
  p=$!
  mrun mixrl-s$a 0 1 "" grpo $DOMS $a $STEPS $CKPTS 8476 || log "Mix-RL s$a training failed"
  wait $p
  for s in ${CKPTS//,/ }; do for r in s$a s$b; do eval_all mixrl-$r step$s $MODEL "$(ck mixrl-$r $s)" 0,1,2,3; done; done
  meval base final $MODEL base 0,1,2,3 ifeval $SEVAL_IF
  meval base final $MODEL base 0,1,2,3 ifbench $SEVAL_IF
  final_sync
  [ -s "$RES/mixrl-s$a/eval.step$STEPS.json" ] && [ -s "$RES/mixrl-s$b/eval.step$STEPS.json" ]
}

# --- paper baselines (docs/mopd-paper-plan.md sections 2, 3): offline distillation (SFT on routed-teacher samples),
# teacher weight merges and reference re-evals. SFT: chip A generates with the routed teacher's weights (swapped in
# place per domain block, no student sync), chip B trains; once A has produced every batch it evaluates the
# intermediate checkpoints while B keeps training.
SFT_STEPS=$(pj "p.get('sft_steps', 120)"); SFT_CKPTS=$(pj "p.get('sft_ckpt_steps', '20,40,60,120')")
SFT_LR=$(pj "p.get('sft_lr', '1e-6')")
TGEN="--teacher-gen math=$TDIR/math,if=$TDIR/if --teacher-gen-block 4"

gen_done_wait() {  # NAME: 0 once the run's rollout process finished generating and released its chip
  local f=/dev/shm/qrl/$1/events/rollout.jsonl
  until sudo grep -q '"kind": "done"' $f 2>/dev/null; do [ -e "$RES/$1/.train_done" ] && return 1; sleep 60; done
  while sudo docker exec qrl pgrep -f "qrl.rollout --run-dir /dev/shm/qrl/$1 " >/dev/null 2>&1; do sleep 10; done
  sleep 20
}

msft() {  # NAME GEN_CHIP LEARNER_CHIP SEED STEPS CKPT_STEPS PORT
  local LFLAGS="$LFLAGS"
  [ "$SFT_LR" = 1e-6 ] || LFLAGS="$LFLAGS --lr $SFT_LR"
  mrun $1 $2 $3 "" sft $DOMS $4 $5 $6 $7 "" "" "$TGEN"
}

sft_run() {  # NAME GEN_CHIP LEARNER_CHIP SEED PORT: SFT run, intermediate checkpoints evaluated on GEN_CHIP
  local name=$1 gc=$2 lc=$3 seed=$4 port=$5 ep s
  ( gen_done_wait $name || exit 0
    stage sft_gen_done PASS "$name"
    for s in ${SFT_CKPTS//,/ }; do
      [ "$s" = "$SFT_STEPS" ] && continue
      ckpt_wait $name $s || exit 0
      eval_all $name step$s $MODEL "$(ck $name $s)" $gc
    done ) &
  ep=$!
  msft $name $gc $lc $seed $SFT_STEPS $SFT_CKPTS $port || log "$name training failed"
  wait $ep
}

sft_smoke() {  # GEN_CHIP LEARNER_CHIP: 2 SFT steps through the real path (teacher swaps, sft learner)
  gsx "$B/done/sft_smoke" && return 0
  local r pr s
  msft sft-smoke $1 $2 0 2 2 8496; r=$?
  s=$(python3 - "$RES/sft-smoke/events" <<'EOF'
import json, os, sys
ev = []
for f in os.listdir(sys.argv[1]):
    ev += [json.loads(x) for x in open(os.path.join(sys.argv[1], f))]
tr = [e for e in ev if e["kind"] == "train"]
g = [e for e in ev if e["kind"] == "generate"]
lw = [e for e in ev if e["kind"] == "load_weights"]
print(f"train {len(tr)} ce {[round(e.get('ce', -1), 4) for e in tr]} fkl_math {[round(e.get('fkl_teacher_math', -1), 4) for e in tr]} "
      f"fkl_if {[round(e.get('fkl_teacher_if', -1), 4) for e in tr]} gen {[round(e['end'] - e['start']) for e in g]}s "
      f"swaps {[(e.get('teacher'), round(e['end'] - e['start'], 1)) for e in lw]}")
sys.exit(0 if len(tr) == 2 and all(e.get("ce", 0) > 0 for e in tr) else 4)
EOF
); pr=$?
  [ $r = 0 ] && r=$pr
  stage sft_smoke $([ $r = 0 ] && echo PASS || echo "FAIL rc=$r") "$s"
  [ $r = 0 ] && echo "$ME $s" | timeout 60 gcloud storage cp - "$B/done/sft_smoke" --quiet
  sudo rm -rf /dev/shm/qrl/sft-smoke "$RES/sft-smoke/ckpt"
  [ $r = 0 ]
}

MERGES="m1-linear:linear:1:1 m2-ta:ta:1:1 m3-ties20:ties:1:0.2 m4-ties50:ties:1:0.5"

job_merges() {  # CHIPS: build M1-M4 on the host CPU (scratch dirs in /dev/shm), one standard eval set each
  local chips=$1 m name meth lam dens d
  fetch_teachers || return 1
  for m in $MERGES; do
    IFS=: read -r name meth lam dens <<< "$m"
    d=/dev/shm/merges/$name
    if [ ! -s $d/merge.json ]; then
      X "cd /rl && rm -rf $d && mkdir -p /dev/shm/merges && python -m qrl.merge_teachers --base $MODEL \
        --teachers math=$TDIR/math if=$TDIR/if --method $meth --lam $lam --density $dens --out $d" \
        > "$RES/merge-$name.log" 2>&1 || { stage merge_$name "FAIL" "$(tail -2 $RES/merge-$name.log)"; continue; }
      mkdir -p "$RES/merge-$name"; sudo cp $d/merge.json "$RES/merge-$name/"
      stage merge_$name PASS "$(tail -1 $RES/merge-$name.log | cut -c1-300)"
    fi
    eval_all merge-$name final $d base $chips
    final_sync
  done
  sudo rm -rf /dev/shm/merges
  local n=0
  for m in $MERGES; do [ -s "$RES/merge-${m%%:*}/eval.final.json" ] && n=$((n + 1)); done
  [ $n = 4 ]
}

REF_SEEDS=$(pj "p.get('ref_seeds', '2234 3234')")

job_refs() {  # CHIPS: extra eval passes with new sampling seeds (same sample counts, no new compile shapes):
  # T-math AIME, T-IF IFEval/IFBench first (the H1 teacher rows), then the base on AIME
  local chips=$1 sd
  fetch_teachers || return 1
  for sd in $REF_SEEDS; do
    MEVAL_SEED=$sd meval refs-tmath seed$sd $TDIR/math base $chips v2 $SEVAL
    MEVAL_SEED=$sd meval refs-tif seed$sd $TDIR/if base $chips ifeval $SEVAL_IF
    MEVAL_SEED=$sd meval refs-tif seed$sd $TDIR/if base $chips ifbench $SEVAL_IF
    final_sync
  done
  for sd in $REF_SEEDS; do MEVAL_SEED=$sd meval refs-base seed$sd $MODEL base $chips v2 $SEVAL; done
  final_sync
}

job_sft_single() {  # SEED: SFT on chips 0/1 (smoke first); merges, then reference re-evals, on chips 2/3
  local seed=$1 name=sft-s$1 p rc=0
  fetch_teachers || return 1
  ( job_merges 2,3; job_refs 2,3 ) &
  p=$!
  if sft_smoke 0 1; then
    sft_run $name 0 1 $seed 8476
  else
    echo "$ME sft smoke failed (see $RES/sft-smoke)" | timeout 60 gcloud storage cp - "$B/HOLD" --quiet
    log "SFT smoke FAILED: set $B/HOLD; merges and reference evals continue on chips 2/3"
    rc=1
  fi
  wait $p
  [ $rc = 0 ] || { final_sync; return 1; }
  eval_all $name step$SFT_STEPS $MODEL "$(ck $name $SFT_STEPS)" 0,1,2,3
  final_sync
  [ -s "$RES/$name/eval.step$SFT_STEPS.json" ] && [ -s "$RES/merge-m3-ties20/eval.final.json" ]
}

job_sft_pair() {  # SEED_A SEED_B: SFT seed A (chips 0/1) and B (chips 2/3), step SFT_STEPS of both on 4 chips
  local a=$1 b=$2 p
  fetch_teachers || return 1
  sft_smoke 0 1 || return 1
  ( sft_run sft-s$b 2 3 $b 8486 ) &
  p=$!
  sft_run sft-s$a 0 1 $a 8476
  wait $p
  local r
  for r in s$a s$b; do eval_all sft-$r step$SFT_STEPS $MODEL "$(ck sft-$r $SFT_STEPS)" 0,1,2,3; done
  final_sync
  [ -s "$RES/sft-s$a/eval.step$SFT_STEPS.json" ] && [ -s "$RES/sft-s$b/eval.step$SFT_STEPS.json" ]
}

job_opd9b() {  # NAME SEED: OPD from the single teacher (plan single_teacher) on every prompt
  local name=$1 seed=$2 s d=/dev/shm/models/single
  if [ "$SINGLE" = tmath ]; then  # same-origin T-math on every prompt (gate 5 switch): one teacher chip is enough
    fetch_teachers || return 1
    evals_during $name 3 &
    local ep=$!
    mrun $name 0 1 2 opd $DOMS $seed $STEPS $CKPTS 8476 "all=$TDIR/math" "--ship-hidden none" \
      || log "$name training failed"
    wait $ep
  else
    [ -f $d/config.json ] || hf_get $SINGLE $d > "$RES/single_download.log" 2>&1 || { log "download of $SINGLE failed"; return 1; }
    mrun $name 0 1 2,3 opd $DOMS $seed $STEPS $CKPTS 8476 "all=$d" "--ship-hidden none --gdn-chunk 256 --mb-tokens 4096" \
      || log "$name training failed"
  fi
  for s in ${CKPTS//,/ }; do eval_all $name step$s $MODEL "$(ck $name $s)" 0,1,2,3; done
  final_sync
  [ -s "$RES/$name/eval.step$STEPS.json" ]
}

run_item() {
  case "$1" in
    teachers) job_teachers ;;
    probe_mixrl) job_probe_mixrl ;;
    mixrl_pair) job_mixrl_pair ;;
    opd9b_s0) job_opd9b $OPD1-s0 0 ;;
    opd9b_s1) job_opd9b $OPD1-s1 1 ;;
    mopd_s0) job_distill mopd-s0 opd 0 diag ;;
    mopd_s1) job_distill mopd-s1 opd 1 diag ;;
    mopd_dist_s0) job_distill mopd-dist-s0 opd_dist 0 all ;;
    mopd_dist_s1) job_distill mopd-dist-s1 opd_dist 1 all ;;
    mopd_s[0-9]) job_distill mopd-s${1#mopd_s} opd ${1#mopd_s} diag ;;
    mopd_dist_s[0-9]) job_distill mopd-dist-s${1#mopd_dist_s} opd_dist ${1#mopd_dist_s} all ;;
    opd9b_s[0-9]) job_opd9b $OPD1-s${1#opd9b_s} ${1#opd9b_s} ;;
    mixrl_pair_s[0-9]s[0-9]) local x=${1#mixrl_pair_s}; job_mixrl_pair ${x%%s*} ${x#*s} ;;
    sft_pair_s[0-9]s[0-9]) local x=${1#sft_pair_s}; job_sft_pair ${x%%s*} ${x#*s} ;;
    sft_single_s[0-9]) job_sft_single ${1#sft_single_s} ;;
    *) log "unknown item $1"; return 1 ;;
  esac
}

deps_done() {
  local d
  for d in $(pj "' '.join(p['items']['$1'].get('deps', []))"); do gsx "$B/done/$d" || return 1; done
  return 0
}

item_fits() {  # ID: spend so far + other live VMs' claimed items (P90 hours left) + this item's P90 hours <= cap
  timeout 600 python3 - "$1" <<'EOF'
import datetime as dt, json, os, subprocess, sys
sys.path.insert(0, os.path.expanduser("~/rl/tpu"))
import guard90
pl = json.load(open(os.path.expanduser("~/plan.json")))
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
    if pid == sys.argv[1] or pid not in pl["items"] or gs("ls", f"{B}/done/{pid}").returncode == 0:
        continue
    owner = gs("cat", u).stdout.split()
    if not owner or owner[0] == me or owner[0] not in live:
        continue
    at = gs("objects", "describe", u, "--format=value(update_time)").stdout.strip()
    el = (now - guard90.ts(at)).total_seconds() / 3600 if at else 0.0
    others += max(0.3, pl["items"][pid]["h90"] - el) * rate(live[owner[0]])
worst = max(guard90.price(pl, s[1], s[2] if len(s) > 2 else "FLEX_START") for s in pl["slots"])
h90 = pl["items"][sys.argv[1]]["h90"]
mine = h90 * (rate(live[me]) if me in live else worst)
tot, cap = spent + others + mine, float(pl["total_cap_usd"])
print(f"gate {sys.argv[1]}: spent ${spent:.2f} + others' P90 remaining ${others:.2f} + mine ${mine:.2f} "
      f"({h90} h) = ${tot:.2f} vs cap ${cap:.0f}")
sys.exit(0 if tot <= cap else 3)
EOF
}

mopd_loop() {
  mopd_setup || infra_fail setup "MOPD setup failed"
  local id pick waited=0 held=0 left need open_ blocked
  mapfile -t IDS < <(pj "'\n'.join(p['items'])")
  while true; do
    hb_gate
    gsx "$B/CAP_REACHED" && self_delete "cap reached"
    if gsx "$B/HOLD"; then
      [ $held = 0 ] && log "HOLD set; not claiming (self-delete after ${HOLD_MAX_S}s)"
      [ $held -ge $HOLD_MAX_S ] && self_delete "HOLD for ${held}s"
      sleep 60; held=$((held + 60)); continue
    fi
    held=0
    refresh_code
    left=$(remaining_h); pick=""; open_=0; blocked=0
    for id in "${IDS[@]}"; do
      gsx "$B/done/$id" && continue
      open_=$((open_ + 1))
      deps_done "$id" || { blocked=$((blocked + 1)); continue; }
      need=$(pj "p['items']['$id']['h90']")
      python3 -c "import sys; sys.exit(0 if $left >= $need else 1)" || { log "$id needs ${need}h, ${left}h max-run left"; continue; }
      if claim "$id"; then
        local gl grc; gl=$(B=$B VM_NAME=$VM_NAME item_fits "$id" 2>&1); grc=$?
        log "$(echo "$gl" | tail -1)"
        if [ $grc = 3 ]; then
          echo "$ME $gl" | timeout 60 gcloud storage cp - "$B/skipped/$id" --quiet
          echo "$ME skipped (budget)" | timeout 60 gcloud storage cp - "$B/done/$id" --quiet
          timeout 60 gcloud storage rm "$B/claims/$id" --quiet >/dev/null 2>&1
          log "SKIPPED $id: does not fit the budget"; continue
        fi
        pick=$id; break
      fi
    done
    if [ -z "$pick" ]; then
      [ $open_ = 0 ] && self_delete "all work items done"
      if [ $blocked -gt 0 ] && [ $waited -lt $MOPD_WAIT_S ]; then
        [ $waited = 0 ] && log "waiting up to ${MOPD_WAIT_S}s for dependencies ($blocked blocked items)"
        sleep 60; waited=$((waited + 60)); continue
      fi
      self_delete "nothing claimable ($open_ open, $blocked waiting on dependencies, waited ${waited}s)"
    fi
    waited=0
    log "claimed $pick (steps $STEPS, teacher steps $TSTEPS, ${left}h max-run left)"
    stage claimed PASS "$pick"
    hb_gate
    local t_item=$(date +%s)
    if run_item "$pick"; then
      hb_ok || self_delete "guard went stale during $pick; item left unmarked for another VM"
      echo "$ME" | timeout 60 gcloud storage cp - "$B/done/$pick" --quiet
      log "done $pick"; stage item_done PASS "$pick $(( ($(date +%s) - t_item) / 60 )) min"
    elif [ $(( $(date +%s) - t_item )) -lt ${INFRA_FAIL_S:-1800} ] && infra_fail "$pick" "failed after $(( $(date +%s) - t_item ))s"; then
      :
    else
      echo "$ME" | timeout 60 gcloud storage cp - "$B/failed/$pick" --quiet
      echo "$ME failed" | timeout 60 gcloud storage cp - "$B/done/$pick" --quiet
      log "FAILED $pick; marked done+failed"; stage item_done "FAIL" "$pick"
    fi
    final_sync
  done
}
