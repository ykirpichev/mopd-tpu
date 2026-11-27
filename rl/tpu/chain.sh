#!/usr/bin/env bash
# Per chip-pair pipeline: wait for <mode>-s0, launch <mode>-s1, then evaluate both runs' checkpoints.
# Usage: chain.sh sync|async [steps]
set -uo pipefail
MODE=${1:?mode}
STEPS=${2:-40}
MID=$((STEPS/2))
MODEL=/dev/shm/models/Qwen3.5-2B
if [[ $MODE == sync ]]; then CHIPS=(0 1); PORTS=(8496 8497); else CHIPS=(2 3); PORTS=(8506 8507); fi

wait_run() { while pgrep -f "qrl.run --run-dir /dev/shm/qrl/$1 " >/dev/null; do sleep 30; done; }
save_run() { sudo docker exec qrl bash -lc "cp -r /dev/shm/qrl/$1/events /dev/shm/qrl/$1/config.json /dev/shm/qrl/$1/*.log /results/$1/"; }
evaluate() {  # chip port out ckpts...
  local chip=$1 port=$2 out=$3; shift 3
  sudo docker exec -e VLLM_CACHE_ROOT=/cache/vllm_root -e TPU_VISIBLE_CHIPS=$chip \
    -e TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 -e TPU_PROCESS_BOUNDS=1,1,1 -e TPU_PROCESS_PORT=$port \
    -e TPU_PROCESS_ADDRESSES=localhost:$port -e CLOUD_TPU_TASK_ID=0 -e TORCH_TPU_TOPOLOGY=1,1,1 qrl \
    bash -lc "cd /rl && python -m qrl.evaluate --model $MODEL --max-model-len 4608 --max-tokens 3072 --out $out --ckpts $* > ${out%.json}.log 2>&1"
}

echo "$(date -u +%FT%TZ) waiting for $MODE-s0"
wait_run "$MODE-s0"; save_run "$MODE-s0"
echo "$(date -u +%FT%TZ) launching $MODE-s1"
bash ~/launch_pair.sh 1 "$STEPS" "$MODE"
sleep 60; wait_run "$MODE-s1"; save_run "$MODE-s1"
echo "$(date -u +%FT%TZ) evaluating"
c0="s0_$MID=/results/$MODE-s0/ckpt/step$(printf %05d $MID).safetensors s0_$STEPS=/results/$MODE-s0/ckpt/step$(printf %05d $STEPS).safetensors"
c1="s1_$MID=/results/$MODE-s1/ckpt/step$(printf %05d $MID).safetensors s1_$STEPS=/results/$MODE-s1/ckpt/step$(printf %05d $STEPS).safetensors"
[[ $MODE == async ]] && c0="base=base $c0"
evaluate "${CHIPS[0]}" "${PORTS[0]}" "/results/eval-$MODE-s0.json" $c0 &
evaluate "${CHIPS[1]}" "${PORTS[1]}" "/results/eval-$MODE-s1.json" $c1 &
wait
echo "$(date -u +%FT%TZ) done $MODE"
