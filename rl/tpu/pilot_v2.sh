#!/usr/bin/env bash
# Phase-0 pilot on one v6e-4 inside the `qrl` container (same mounts as the first run).
# Chips 0-2: base-model profile of all train prompts at an 8k cap. Chip 3: decode timing probe, then the
# learner bench sweep. Afterwards all 4 chips: base-model eval on the v2 set (32 samples).
set -uo pipefail
MODEL=/dev/shm/models/Qwen3.5-2B
OUT=/results/pilot
CAP=8192
LEN=9728
mkdir -p ~/qrl-results/pilot

chip_env() {  # chip port
  echo "-e TPU_VISIBLE_CHIPS=$1 -e TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 -e TPU_PROCESS_BOUNDS=1,1,1 \
-e TPU_PROCESS_PORT=$2 -e TPU_PROCESS_ADDRESSES=localhost:$2 -e CLOUD_TPU_TASK_ID=0 -e TORCH_TPU_TOPOLOGY=1,1,1"
}
in_chip() {  # chip port log cmd...
  local chip=$1 port=$2 log=$3; shift 3
  sudo docker exec -e VLLM_CACHE_ROOT=/cache/vllm_root $(chip_env "$chip" "$port") qrl \
    bash -lc "cd /rl && $* > $log 2>&1"
}

echo "$(date -u +%FT%TZ) profile + probes"
for s in 0 1 2; do
  in_chip $s $((8600 + s)) $OUT/profile_s$s.log python -m qrl.profile_prompts --model $MODEL --samples 8 \
    --max-tokens $CAP --max-model-len $LEN --shard-index $s --num-shards 3 --out $OUT/profile_s$s.json &
done
(
  in_chip 3 8603 $OUT/timing.log python -m qrl.profile_prompts --model $MODEL --timing-probe --max-model-len $LEN \
    --out $OUT/timing.json
  for c in "64 0" "256 0" "256 16384" "256 32768"; do
    set -- $c
    in_chip 3 8603 $OUT/bench_c$1_t$2.log python tpu/bench_learner.py --model $MODEL --grad-ckpt --len 6000 \
      --seqs 16 --gdn-chunk $1 --mb-tokens $2 --json-out $OUT/bench_c$1_t$2.json
  done
  in_chip 3 8603 $OUT/bench_compile.log python tpu/bench_learner.py --model $MODEL --grad-ckpt --len 6000 \
    --seqs 16 --gdn-chunk 256 --mb-tokens 16384 --gdn-contig-grad --compile --json-out $OUT/bench_compile.json
) &
wait
sudo docker exec qrl bash -lc "cd /rl && python -m qrl.profile_prompts --merge $OUT/profile_s{0,1,2}.json --out $OUT/profile.json" \
  > ~/qrl-results/pilot/profile_summary.txt 2>&1
echo "$(date -u +%FT%TZ) base eval"
for s in 0 1 2 3; do
  in_chip $s $((8610 + s)) $OUT/eval_base_s$s.log python -m qrl.evaluate --model $MODEL --eval-set v2 --samples 32 \
    --max-tokens $CAP --max-model-len $LEN --ckpts base=base --shard-index $s --num-shards 4 \
    --out $OUT/eval_base_s$s.json &
done
wait
sudo docker exec qrl bash -lc "cd /rl && python -m qrl.evaluate --merge $OUT/eval_base_s{0,1,2,3}.json --out $OUT/eval_base.json"
echo "$(date -u +%FT%TZ) pilot done"
