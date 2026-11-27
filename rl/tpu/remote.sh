#!/bin/bash
# Runs on the ct6e-standard-4t VM. Pulls vllm-torchtpu, restores the compile cache, smoke-tests
# the loop, then for each seed: sync (chips 0/1) and async (chips 2/3) side by side, then evaluation
# of that seed's checkpoints (one vLLM engine per chip). Seed 0 finishes completely before seed 1.
set -euo pipefail
IMAGE=${IMAGE:-vllm/vllm-torchtpu:v0.32.0}
MODEL_REPO=${MODEL_REPO:-Qwen/Qwen3.5-2B}
MODEL=/dev/shm/models/$(basename "$MODEL_REPO")
STEPS=${STEPS:-40}
SEEDS=${SEEDS:-"0 1"}
K=${K:-2}
P=${P:-16}
GS=${GS:-8}
MAX_TOKENS=${MAX_TOKENS:-3072}
MAX_LEN=${MAX_LEN:-4608}  # longest AIME prompt is 1329 tokens + 3072 new
CKPT_STEPS=${CKPT_STEPS:-0,20,40}
EVAL_SAMPLES=${EVAL_SAMPLES:-8}
RES=$HOME/qrl-results
mkdir -p "$RES" "$HOME/cache/vllm" "$HOME/cache/torch_tpu"
log() { echo "$(date -u +%FT%TZ) $*"; }
sync_cache() {  # $1 src $2 dst
  [ -n "${CACHE_BUCKET:-}" ] || return 0
  gcloud storage rsync -r "$1" "$2" --quiet >/dev/null 2>&1 || gsutil -m -q rsync -r "$1" "$2" || true
}

sync_cache "$CACHE_BUCKET" "$HOME/cache"
sudo docker pull -q "$IMAGE"
sudo docker rm -f qrl >/dev/null 2>&1 || true
sudo docker run -d --name qrl --privileged --net host --shm-size 64g \
  -v /dev/shm:/dev/shm -v "$HOME/rl:/rl" -v "$RES:/results" -v "$HOME/cache:/cache" \
  -e VLLM_XLA_CACHE_PATH=/cache/vllm -e TORCH_TPU_COMPILATION_CACHE_DIR=/cache/torch_tpu \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 -e PYTHONPATH=/rl -w /rl --entrypoint sleep "$IMAGE" infinity
X() { sudo docker exec qrl bash -lc "$*"; }
X "pip install -q datasets 'huggingface_hub[hf_transfer]' 2>&1 | tail -1; python -c 'import vllm, torch, torch_tpu, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)'"
X "python -c \"from huggingface_hub import snapshot_download as s; s('$MODEL_REPO', local_dir='$MODEL', max_workers=16)\""
X "python -c 'from qrl import data; print(len(data.load_train()), len(data.load_eval()), \"datasets ok\")'"

ROLL="--prompts-per-batch $P --group-size $GS --max-tokens $MAX_TOKENS --max-model-len $MAX_LEN"
LEARN="--micro-batch 2 --max-model-len $MAX_LEN --grad-ckpt --lr 1e-6"
chip_env() { echo "TPU_VISIBLE_CHIPS=$1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TORCH_TPU_TOPOLOGY=1,1,1 TPU_PROCESS_PORT=$((8500+$1)) TPU_PROCESS_ADDRESSES=localhost:$((8500+$1))"; }

log "smoke: 2 sync steps, small batch"
X "python -m qrl.run --run-dir /results/smoke --model $MODEL --mode sync --steps 2 --seed 99 \
   --rollout-chip 0 --learner-chip 1 --rollout-args '--prompts-per-batch 2 --group-size 2 --max-tokens 256 --max-model-len $MAX_LEN' \
   --learner-args '$LEARN'" || { log "smoke failed"; X "tail -50 /results/smoke/rollout.log; tail -50 /results/smoke/learner.log"; exit 2; }
X "python -m qrl.analyze /results/smoke"
sync_cache "$HOME/cache" "$CACHE_BUCKET"

first=1
for S in $SEEDS; do
  log "seed $S: sync (chips 0/1) and async k=$K (chips 2/3) in parallel"
  X "python -m qrl.run --run-dir /results/sync-s$S --model $MODEL --mode sync --steps $STEPS --seed $S \
     --rollout-chip 0 --learner-chip 1 --port-base 8476 --rollout-args '$ROLL' \
     --learner-args '$LEARN --ckpt-dir /results/sync-s$S/ckpt --ckpt-steps $CKPT_STEPS'" &
  A=$!
  X "python -m qrl.run --run-dir /results/async-s$S --model $MODEL --mode async --staleness $K --steps $STEPS --seed $S \
     --rollout-chip 2 --learner-chip 3 --port-base 8486 --rollout-args '$ROLL' \
     --learner-args '$LEARN --ckpt-dir /results/async-s$S/ckpt --ckpt-steps $CKPT_STEPS'" &
  B=$!
  wait $A || log "sync-s$S failed"; wait $B || log "async-s$S failed"
  X "python -m qrl.analyze /results/sync-s$S /results/async-s$S --out /results/summary-s$S.json" || true
  sync_cache "$HOME/cache" "$CACHE_BUCKET"

  log "seed $S: eval"
  # Step 0 is the base model, identical for both modes and seeds: evaluate it once.
  jobs_=()
  [ $first = 1 ] && jobs_+=("/results/base base=base")
  for M in sync async; do
    d=/results/$M-s$S
    for st in ${CKPT_STEPS//,/ }; do
      [ "$st" = 0 ] && continue
      jobs_+=("$d step$st=$(printf '%s/ckpt/step%05d.safetensors' $d $st)")
    done
  done
  chip=0; pids=()
  for j in "${jobs_[@]}"; do
    d=${j%% *}; spec=${j#* }; mkdir -p "$RES/${d#/results/}"
    X "$(chip_env $chip) python -m qrl.evaluate --model $MODEL --ckpts $spec --samples $EVAL_SAMPLES \
       --max-tokens $MAX_TOKENS --max-model-len $MAX_LEN --out $d/eval.${spec%%=*}.json > $d/eval.${spec%%=*}.log 2>&1" &
    pids+=($!); chip=$(( (chip + 1) % 4 ))
    if [ $chip -eq 0 ]; then wait "${pids[@]}" || true; pids=(); fi
  done
  [ ${#pids[@]} -gt 0 ] && { wait "${pids[@]}" || true; }
  first=0
  sync_cache "$HOME/cache" "$CACHE_BUCKET"
done
log "done"
