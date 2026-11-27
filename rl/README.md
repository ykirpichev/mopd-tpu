# Sync vs. async GRPO rollouts on TPU v6e (TorchTPU + vLLM TorchTPU)

A minimal two-process GRPO loop used to test whether bounded-staleness asynchronous rollouts
match synchronous rollouts in quality while removing idle time.

- `qrl/rollout.py`: rollout engine (vLLM TorchTPU on one chip, or HF `generate` for CPU tests).
  Samples at temperature 1, top-p 1, and records the log-prob of every sampled token.
- `qrl/learner.py`: TorchTPU learner on one chip. bf16 weights and gradients on the chip; fp32
  master weights and AdamW on the host. One optimizer step per rollout batch.
- `qrl/grpo.py`: group-normalized advantages and the truncated-importance-sampling policy loss.
- `qrl/channel.py`: weights and rollout batches move through files on `/dev/shm`.
- `qrl/run.py`: starts both processes pinned to their chips. `--mode sync` is `--staleness 0`.
- `qrl/evaluate.py`, `qrl/analyze.py`: AIME 2024+2025 pass@1 and timing/idle summaries.
- `tpu/flex_run.sh`, `tpu/remote.sh`: Flex-start `ct6e-standard-4t` driver with a delete-on-exit trap.

## Algorithm

Batch `b` is trained by optimizer step `b`, which runs on weights version `b`. The rollout engine
may start batch `b` once version `b - k` is published, so a batch is at most `k` versions stale.
With `k = 0` the engine waits for every weight push (synchronous baseline: the sampler idles while
the learner trains and the learner idles while the sampler generates). With `k > 0` the engine keeps
sampling with the newest weights it has and loads newer ones at batch boundaries; the learner never
waits for the engine to take them.

Neither mode stores old-policy weights. The learner's forward pass under its current weights gives
`logp_cur`; the engine's recorded `logp_behav` stands in for the old policy. The per-token weight
`w = min(exp(logp_cur - logp_behav), C)` (detached) multiplies the policy-gradient term
`-A * logp_cur`. The same loss runs in both modes; in sync mode `w` only absorbs the small numeric
gap between the vLLM sampler and the trainer.

## CPU tests

```bash
python -m venv /tmp/venv && /tmp/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/venv/bin/pip install transformers safetensors datasets pytest
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir /tmp/models/qwen3.5-0.8b   # config/tokenizer for the tiny model
cd rl && /tmp/venv/bin/python -m pytest -q tests/test_cpu.py
```

## TPU

Model: `Qwen/Qwen3.5-2B` (same `qwen3_5` hybrid Gated DeltaNet architecture as Qwen3.8-27B).

```bash
bash rl/tpu/install_gcloud.sh            # only if gcloud is missing; installs to ~/google-cloud-sdk
export PATH="$HOME/google-cloud-sdk/bin:$PATH"
# GCP_SA_KEY_JSON (Cloud Agents secret) holds the service-account key, raw JSON or base64.
# flex_run.sh sources tpu/gcp_auth.sh, which writes it to a 0600 temp file, activates it, and deletes it.
bash rl/tpu/flex_run.sh                  # auth + preflight: lists TPU VMs, queued resources, clusters, ct6e VMs
ALLOW_CREATE=1 bash rl/tpu/flex_run.sh   # Flex-start europe-west4-a: creates qrl-bc4a-*, runs, deletes on exit
ALLOW_CREATE=1 PROVISIONING=SPOT ZONE=us-east1-d PRICE_PER_H=2.685 bash rl/tpu/flex_run.sh   # Spot fallback
```

`REMOTE_ENV="STEPS=40 SEEDS='0 1'"` overrides `remote.sh` settings. Spend is metered from the
RUNNING time and the driver stops 15 minutes before `BUDGET_USD` (pass `SPENT_BEFORE` across sessions).
