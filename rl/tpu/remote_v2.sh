#!/bin/bash
# v2 runs on one ct6e-standard-4t. Env (from ~/go.env, written by the driver):
#   PILOT=1            run the phase-0 pilot first, upload it, then wait for ~/go.env to define the pair
#   RUN_A / RUN_B      "name mode staleness_mode k behavior seed" for chips 0/1 and 2/3 (empty = none)
#   STEPS CAP LEARNER_FLAGS RESULTS_GS CACHE_BUCKET
# Results (events, logs, evals; no checkpoints) are rsynced to $RESULTS_GS every 5 minutes so a cap kill
# loses at most a few minutes of logs.
set -uo pipefail
IMAGE=${IMAGE:-vllm/vllm-torchtpu:v0.32.0}
MODEL_REPO=Qwen/Qwen3.5-2B
MODEL=/dev/shm/models/Qwen3.5-2B
RES=$HOME/qrl-results
mkdir -p "$RES" "$HOME/cache/vllm" "$HOME/cache/torch_tpu"
log() { echo "$(date -u +%FT%TZ) $*"; }
type stage >/dev/null 2>&1 || stage() { log "STAGE $*"; }  # vm_startup.sh exports the GCS-backed one
[ -f ~/go.env ] && source ~/go.env
if [ "${GKE:-0}" = 1 ]; then
  # GKE pod: this IS the vLLM TorchTPU container (privileged, 4 chips), so `docker exec qrl` runs commands in place
  # with the same mounts (as symlinks) and environment; docker pull/run/rm are no-ops.
  ln -sfn "$HOME/rl" /rl; ln -sfn "$RES" /results; ln -sfn "$HOME/cache" /cache
  DOCKER_ENV=(VLLM_XLA_CACHE_PATH=/cache/vllm TORCH_TPU_COMPILATION_CACHE_DIR=/cache/torch_tpu
              VLLM_CACHE_ROOT=/cache/vllm_root HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONPATH=/rl)
  sudo() { "$@"; }  # always: a real sudo would bypass the docker() shim below
  docker() {
    [ "$1" = exec ] || return 0
    shift; local envs=()
    while [ "$1" = -e ]; do envs+=("$2"); shift 2; done
    shift  # container name
    (cd /rl && env "${DOCKER_ENV[@]}" "${envs[@]}" "$@")
  }
fi
if [ "${AUTO:-0}" = 1 ]; then  # plan90: started by vm_startup.sh, no agent needed
  CACHE_BUCKET=gs://qrl-bc4a-compile-cache-tpu-builders-509006
  B=${QRL_Q:-$CACHE_BUCKET/plan90}  # plan90-dry/ in DRY mode: never the real queue
  CAP=$(python3 -c "import json; print(json.load(open('$HOME/plan.json'))['cap'])")
  MYSTART=$(timeout 60 gcloud compute instances describe $VM_NAME --zone $VM_ZONE --format='value(lastStartTimestamp)')
  RESULTS_GS=$B/results/$VM_NAME-$(date -u -d "$MYSTART" +%Y%m%dT%H%M%S)
  cp ~/plan.json ~/rl/plan.json
fi
: "${RESULTS_GS:?}" "${CACHE_BUCKET:?}"

timed() {  # stage cmd...: run, record PASS/FAIL with the duration and the last output line
  local s=$1 t=$(date +%s) out rc; shift
  out=$("$@" 2>&1); rc=$?
  echo "$out" | tail -5
  stage "$s" $([ $rc = 0 ] && echo PASS || echo "FAIL rc=$rc") "$(( $(date +%s) - t ))s $(echo "$out" | tail -1 | cut -c1-200)"
  return $rc
}
cache_rw() {  # read the compile cache, prove write access with a probe object under the queue prefix
  gcloud storage rsync -r "$CACHE_BUCKET/cache" "$HOME/cache" --quiet > /tmp/cache_rsync.log 2>&1 \
    || { echo "rsync failed: $(grep -v '^\s*$' /tmp/cache_rsync.log | tail -1)"; return 1; }
  echo probe | gcloud storage cp - "$B/probe/$VM_NAME" --quiet && gcloud storage rm "$B/probe/$VM_NAME" --quiet \
    && echo "cache $(du -sh $HOME/cache | cut -f1), write ok"
}
timed compile_cache cache_rw
pull() { sudo docker pull -q "$IMAGE" && echo "$IMAGE $(sudo docker image inspect -f '{{.Size}}' "$IMAGE" | awk '{printf "%.1f GB", $1/1e9}')"; }
timed image_pull pull
sudo docker rm -f qrl >/dev/null 2>&1 || true
timed container_start sudo docker run -d --name qrl --privileged --net host --shm-size 64g \
  -v /dev/shm:/dev/shm -v "$HOME/rl:/rl" -v "$RES:/results" -v "$HOME/cache:/cache" \
  -e VLLM_XLA_CACHE_PATH=/cache/vllm -e TORCH_TPU_COMPILATION_CACHE_DIR=/cache/torch_tpu \
  -e VLLM_CACHE_ROOT=/cache/vllm_root -e HF_HUB_ENABLE_HF_TRANSFER=1 -e PYTHONPATH=/rl -w /rl \
  --entrypoint sleep "$IMAGE" infinity
X() { sudo docker exec qrl bash -lc "$*"; }
timed pip_install X "pip install -q datasets 'huggingface_hub[hf_transfer]' 2>&1 | tail -1; python -c 'import datasets, huggingface_hub as h; print(\"datasets\", datasets.__version__, \"hub\", h.__version__)'"
timed hf_model X "python -c \"from huggingface_hub import snapshot_download as s; s('$MODEL_REPO', local_dir='$MODEL', max_workers=16)\" >/dev/null && du -sh $MODEL"
timed hf_datasets X "python -c 'from qrl import data; print(len(data.load_train()), len(data.load_eval(\"v2\")), \"datasets ok\")'"
log "setup done"

(
  while true; do
    for d in /dev/shm/qrl/*/; do
      [ -d "$d" ] || continue
      n=$(basename "$d"); mkdir -p "$RES/$n"
      sudo cp -r "$d/events" "$d/config.json" "$RES/$n/" 2>/dev/null; sudo cp "$d"/*.log "$RES/$n/" 2>/dev/null
    done
    cp ~/qrl-remote.log "$RES/" 2>/dev/null
    gcloud storage rsync -r "$RES" "$RESULTS_GS" --exclude='.*\.safetensors$' --quiet >/dev/null 2>&1
    sleep 300
  done
) &
SYNC_PID=$!
waitjobs() {  # wait for this shell's jobs except the rsync loop (a bare `wait` would block on it forever)
  local p; for p in $(jobs -p); do [ "$p" = "$SYNC_PID" ] || wait "$p"; done
}
final_sync() {
  sleep 2; for d in /dev/shm/qrl/*/; do [ -d "$d" ] || continue; n=$(basename "$d"); mkdir -p "$RES/$n"
    sudo cp -r "$d/events" "$d/config.json" "$RES/$n/" 2>/dev/null; sudo cp "$d"/*.log "$RES/$n/" 2>/dev/null; done
  cp ~/qrl-remote.log "$RES/" 2>/dev/null
  gcloud storage rsync -r "$RES" "$RESULTS_GS" --exclude='.*\.safetensors$' --quiet >/dev/null 2>&1
  gcloud storage rsync -r "$HOME/cache" "$CACHE_BUCKET/cache" --quiet >/dev/null 2>&1 || true
}

chip_env() {  # chip port
  echo "-e TPU_VISIBLE_CHIPS=$1 -e TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 -e TPU_PROCESS_BOUNDS=1,1,1 \
-e TPU_PROCESS_PORT=$2 -e TPU_PROCESS_ADDRESSES=localhost:$2 -e CLOUD_TPU_TASK_ID=0 -e TORCH_TPU_TOPOLOGY=1,1,1"
}
in_chip() {  # chip port log cmd...
  local chip=$1 port=$2 lg=$3; shift 3
  sudo docker exec $(chip_env "$chip" "$port") qrl bash -lc "cd /rl && $* > $lg 2>&1"
}

run_pilot() {
  CAP=${CAP:-4096}; LEN=$((CAP + 1536)); OUT=/results/pilot; mkdir -p "$RES/pilot"
  log "pilot: profile (chips 0-2) + timing probe and learner bench (chip 3), cap $CAP"
  for s in 0 1 2; do
    in_chip $s $((8600 + s)) $OUT/profile_s$s.log python -m qrl.profile_prompts --model $MODEL --samples 8 \
      --max-tokens $CAP --max-model-len $LEN --shard-index $s --num-shards 3 --out $OUT/profile_s$s.json &
  done
  (
    in_chip 3 8603 $OUT/timing.log python -m qrl.profile_prompts --model $MODEL --timing-probe --max-model-len $LEN \
      --out $OUT/timing.json
    IFS=';' read -ra BENCHES <<< "${BENCH:-64 0 ;256 16384 ;256 16384 --gdn-contig-grad --compile}"
    for c in "${BENCHES[@]}"; do
      set -- $c; tag=c$1_t$2$([ -n "${3:-}" ] && echo _compile)
      in_chip 3 8603 $OUT/bench_$tag.log timeout ${BENCH_TIMEOUT:-1500} python tpu/bench_learner.py --model $MODEL --grad-ckpt \
        --len $((CAP - 500)) --seqs 16 --gdn-chunk $1 --mb-tokens $2 ${3:-} ${4:-} --json-out $OUT/bench_$tag.json
    done
  ) &
  waitjobs
  X "cd /rl && python -m qrl.profile_prompts --merge $OUT/profile_s{0,1,2}.json --out $OUT/profile.json" \
    > "$RES/pilot/profile_summary.txt" 2>&1
  log "pilot: base eval v2, ${BASE_SAMPLES:-32} samples"
  for s in 0 1 2 3; do
    in_chip $s $((8610 + s)) $OUT/eval_base_s$s.log python -m qrl.evaluate --model $MODEL --eval-set v2 --samples ${BASE_SAMPLES:-32} \
      --max-tokens $CAP --max-model-len $LEN --ckpts base=base --shard-index $s --num-shards 4 \
      --out $OUT/eval_base_s$s.json &
  done
  waitjobs
  X "cd /rl && python -m qrl.evaluate --merge $OUT/eval_base_s{0,1,2,3}.json --out $OUT/eval_base.json"
  log "pilot done"
  final_sync
}

evaluate_run() {  # name chips(csv) port_base
  local name=$1 chips=(${2//,/ }) pb=$3 n i spec samples
  n=${#chips[@]}
  local specs
  IFS=';' read -ra specs <<< "${EVAL_SPECS:-step$MID 16;step$STEPS 32}"
  for spec in "${specs[@]}"; do
    set -- $spec
    for i in $(seq 0 $((n - 1))); do
      in_chip ${chips[$i]} $((pb + i)) /results/$name/eval.$1.s$i.log python -m qrl.evaluate --model $MODEL --eval-set v2 \
        --samples $2 --max-tokens $CAP --max-model-len $LEN \
        --ckpts "$1=/results/$name/ckpt/$(printf 'step%05d' ${1#step}).safetensors" \
        --shard-index $i --num-shards $n --out /results/$name/eval.$1.s$i.json &
    done
    waitjobs
    X "cd /rl && python -m qrl.evaluate --merge $(for i in $(seq 0 $((n - 1))); do printf '/results/%s/eval.%s.s%d.json ' $name $1 $i; done) --out /results/$name/eval.$1.json"
  done
}

train() {  # "name mode smode k behavior seed" rollout_chip learner_chip port
  set -- $1 "$2" "$3" "$4"
  local name=$1 mode=$2 smode=$3 k=$4 beh=$5 seed=$6 rc=$7 lc=$8 port=$9
  mkdir -p "$RES/$name"
  stage train_start PASS "$name"
  ( for i in $(seq 1 720); do  # first-grant watcher: first learner train step, or give up after 6 h
      grep -q '"kind": "train"' /dev/shm/qrl/$name/events/learner.jsonl 2>/dev/null \
        && { stage first_train_step PASS "$name"; exit 0; }
      sleep 30
    done ) &
  sudo docker exec qrl bash -lc "rm -rf /dev/shm/qrl/$name; cd /rl && python -m qrl.run --run-dir /dev/shm/qrl/$name \
    --model $MODEL --mode $mode --staleness $k --staleness-mode $smode --steps $STEPS --seed $seed \
    --rollout-chip $rc --learner-chip $lc --port-base $port --rollout-args '$ROLL' \
    --learner-args '--max-model-len $LEN --grad-ckpt --lr 1e-6 --log-entropy --behavior-logprobs $beh \
      --ckpt-dir /results/$name/ckpt --ckpt-steps ${CKPT_STEPS:-$MID,$STEPS} $LEARNER_FLAGS' > /results/$name/run.out 2>&1"
  local rc_=$?
  sudo cp -r /dev/shm/qrl/$name/events /dev/shm/qrl/$name/config.json "$RES/$name/" 2>/dev/null
  sudo cp /dev/shm/qrl/$name/*.log "$RES/$name/" 2>/dev/null
  return $rc_
}

run_pair() {
LEN=$((CAP + 1536)); MID=$((STEPS / 2))
[ -f "$RES/pilot/profile.json" ] || cp ~/rl/pilot/profile.json "$RES/pilot/profile.json" 2>/dev/null || { mkdir -p "$RES/pilot"; cp ~/rl/pilot/profile.json "$RES/pilot/"; }
ROLL="--prompts-per-batch 16 --group-size 8 --max-tokens $CAP --max-model-len $LEN $( [ "${TRAIN_FILTER:-1}" = 1 ] && echo "--train-filter /results/pilot/profile.json --keep-zero-frac 0.2") --temperature 1.0"
rm -f "$RES/DONE_B"
if [ -n "${RUN_B:-}" ]; then
  ( log "train B: $RUN_B"; train "$RUN_B" 2 3 8486 || log "B failed"
    nb=${RUN_B%% *}; log "eval B on chips 2,3"; evaluate_run "$nb" 2,3 8620; touch "$RES/DONE_B"; log "B done" ) &
else
  touch "$RES/DONE_B"
fi
if [ -n "${RUN_A:-}" ]; then
  log "train A: $RUN_A"; train "$RUN_A" 0 1 8476 || log "A failed"
  na=${RUN_A%% *}
  if [ "${EVAL_SPLIT:-0}" = 1 ]; then  # A evaluates on its own chips while B trains or evaluates on 2,3
    log "eval A on chips 0,1"; evaluate_run "$na" 0,1 8630
  else
    while [ ! -f "$RES/DONE_B" ]; do sleep 20; done
    log "eval A on chips 0-3"; evaluate_run "$na" 0,1,2,3 8630
  fi
fi
waitjobs
log "pair done: ${RUN_A:-} | ${RUN_B:-}"
final_sync
log "PAIR_DONE"
RUN_A=; RUN_B=
}

if [ "${AUTO:-0}" = 1 ]; then
  source ~/rl/tpu/auto90.sh
  if [ "${QRL_DRY:-0}" = 1 ]; then dry_loop
  elif python3 -c "import json, sys; sys.exit(0 if json.load(open('$HOME/plan.json')).get('items') else 1)"; then
    source ~/rl/tpu/mopd.sh; mopd_loop
  else auto_loop; fi
else
  [ "${PILOT:-0}" = 1 ] && { run_pilot; PILOT=0; }
  while true; do
    while [ -z "${RUN_A:-}${RUN_B:-}" ]; do
      log "waiting for ~/go.assign (and ~/rl/pilot/profile.json)"
      while [ ! -f ~/go.assign ]; do sleep 15; done
      sleep 5; source ~/go.assign; rm -f ~/go.assign
      [ "${PILOT:-0}" = 1 ] && { run_pilot; PILOT=0; }
    done
    run_pair
  done
fi
