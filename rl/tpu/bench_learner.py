"""Learner throughput on one chip with a synthetic batch: forward+backward vs. optimizer time.

Pilot sweep (one chip, ~20 min), each config a separate process so compile caches don't mix:
  for c in "64 0" "256 0" "256 16384" "256 32768"; do set -- $c
    python /rl/tpu/bench_learner.py --model M --grad-ckpt --len 6000 --seqs 16 --gdn-chunk $1 --mb-tokens $2; done
  python /rl/tpu/bench_learner.py ... --gdn-chunk 256 --mb-tokens 16384 --gdn-contig-grad --compile
"""
import argparse
import json
import sys
import time

sys.path.insert(0, "/rl")
import torch  # noqa: E402

from qrl import gdn_patch  # noqa: E402
from qrl.grpo import LossConfig, policy_loss  # noqa: E402
from qrl.learner import HostAdamW, get_device, make_microbatches, token_logprobs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--seqs", type=int, default=16)
ap.add_argument("--len", type=int, default=1700)
ap.add_argument("--micro-batch", type=int, default=2)
ap.add_argument("--mb-tokens", type=int, default=0)
ap.add_argument("--quantum", type=int, default=512)
ap.add_argument("--max-len", type=int, default=9728)
ap.add_argument("--grad-ckpt", action="store_true")
ap.add_argument("--compile", action="store_true")
ap.add_argument("--gdn-chunk", type=int, default=0)
ap.add_argument("--gdn-solve", default="")
ap.add_argument("--gdn-contig-grad", action="store_true")
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--json-out", default=None)
args = ap.parse_args()

desc = gdn_patch.apply(args.gdn_chunk or None, args.gdn_solve or None, args.gdn_contig_grad)
print(desc, flush=True)
dev = get_device("tpu")
from transformers import AutoModelForCausalLM  # noqa: E402

m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev)
m.train()
if args.grad_ckpt:
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
params = [p for p in m.parameters() if p.requires_grad]
opt = HostAdamW(params, lr=1e-6)
fwd = torch.compile(token_logprobs, backend="tpu", dynamic=False) if args.compile else token_logprobs
g = torch.Generator().manual_seed(0)
seqs = [{"prompt_ids": torch.randint(0, 200000, (120,), generator=g).tolist(),
         "completion_ids": torch.randint(0, 200000, (args.len - 120,), generator=g).tolist(),
         "behav_logp": [-1.0] * (args.len - 120)} for _ in range(args.seqs)]
adv = torch.randn(args.seqs)
cfg = LossConfig()
res = []
for step in range(args.steps):
    t0 = time.time()
    n_tok = 0
    for ids, tgt, behav, mask, a in make_microbatches(seqs, adv, args.micro_batch, args.quantum, args.max_len, 0,
                                                      args.mb_tokens):
        ids, tgt, behav, mask, a = (x.to(dev) for x in (ids, tgt, behav, mask, a))
        lp = fwd(m, ids, tgt)
        loss, st = policy_loss(lp, behav, a, mask, cfg)
        loss.backward()
        n_tok += st["tokens"]
    float(params[0].grad.float().sum())
    t1 = time.time()
    opt.step(1.0)
    float(params[0].float().sum())
    t2 = time.time()
    res.append({"step": step, "fwd_bwd_s": t1 - t0, "tok_s": n_tok / (t1 - t0), "opt_s": t2 - t1})
    print(f"step {step}: fwd+bwd {t1 - t0:.1f}s ({n_tok / (t1 - t0):.0f} tok/s), host optimizer {t2 - t1:.1f}s", flush=True)
if args.json_out:
    json.dump({"config": vars(args), "gdn": desc, "steps": res}, open(args.json_out, "w"), indent=1)
