#!/usr/bin/env bash
# Launch sync (chips 0/1) and async k=2 (chips 2/3) for one seed inside the `qrl` container.
# Usage: launch_pair.sh <seed> [steps] [sync|async|both]
set -euo pipefail
SEED=${1:?seed}
STEPS=${2:-40}
ONLY=${3:-both}
MODEL=/dev/shm/models/Qwen3.5-2B
RA="--prompts-per-batch 16 --group-size 8 --max-tokens 3072 --max-model-len 4608"
for spec in "sync 0 1 8476" "async 2 3 8486"; do
  set -- $spec
  MODE=$1 RC=$2 LC=$3 PORT=$4
  [[ $ONLY == both || $ONLY == "$MODE" ]] || continue
  RUN=${MODE}-s${SEED}
  LA="--micro-batch 2 --max-model-len 4608 --grad-ckpt --lr 1e-6 --ckpt-dir /results/${RUN}/ckpt --ckpt-steps 0,$((STEPS/2)),${STEPS}"
  sudo docker exec -d -e VLLM_CACHE_ROOT=/cache/vllm_root qrl bash -lc \
    "rm -rf /dev/shm/qrl/${RUN}; mkdir -p /results/${RUN}; python -m qrl.run --run-dir /dev/shm/qrl/${RUN} --model ${MODEL} --mode ${MODE} --staleness 2 --steps ${STEPS} --seed ${SEED} --rollout-chip ${RC} --learner-chip ${LC} --port-base ${PORT} --rollout-args '${RA}' --learner-args '${LA}' > /results/${RUN}/run.out 2>&1"
  echo "launched ${RUN}"
done
