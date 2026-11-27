"""Rollout process: generates GRPO groups and records behavior log-probs.

Staleness rule: batch b is consumed by learner step b, which runs on weights
version b. The worker may start batch b only once version >= b - k is
published, so every batch is at most k versions stale. k = 0 is the
synchronous baseline: the engine waits for each weight push before sampling.
New weights are picked up at batch boundaries without the learner waiting.
"""
import argparse
import os
import time

from . import data
from .channel import Channel, EventLog, wait_until


class HFBackend:
    """CPU/test backend using transformers generate."""

    def __init__(self, model_path, max_model_len, seed, dtype="float32", device="cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, dtype)).to(device).eval()
        self.max_model_len = max_model_len

    def load_weights(self, path):
        from safetensors.torch import load_file

        sd = load_file(path)
        own = self.model.state_dict()
        mapped = {}
        for k, v in sd.items():
            for cand in (k, k.replace("model.language_model.", "model.")):
                if cand in own:
                    mapped[cand] = v.to(own[cand].dtype)
                    break
        missing = [k for k in own if k not in mapped and not k.endswith("lm_head.weight")]
        assert not missing, f"weights missing for {missing[:5]}"
        self.model.load_state_dict(mapped, strict=False)

    def generate(self, prompts, group_size, max_tokens, seeds, temperature=1.0, top_p=1.0):
        torch = self.torch
        out = []
        for p, s in zip(prompts, seeds):
            ids = self.tok(p, return_tensors="pt").input_ids.to(self.device)
            for j in range(group_size):
                torch.manual_seed(s + j)
                g = self.model.generate(
                    ids, do_sample=True, temperature=temperature, top_k=0, top_p=top_p, max_new_tokens=max_tokens,
                    output_scores=True, return_dict_in_generate=True, pad_token_id=self.tok.eos_token_id,
                )
                comp = g.sequences[0, ids.shape[1]:]
                lps = [torch.log_softmax(sc[0].float(), -1)[t].item() for sc, t in zip(g.scores, comp)]
                eos = self.tok.eos_token_id
                finished = len(comp) > 0 and int(comp[-1]) in (eos if isinstance(eos, list) else [eos])
                out.append({
                    "prompt_ids": ids[0].tolist(), "completion_ids": comp.tolist(), "behav_logp": lps,
                    "text": self.tok.decode(comp, skip_special_tokens=True), "truncated": not finished,
                })
        return out


def _vllm_load_weights(worker, path):
    """Runs inside the vLLM worker. In-place reload in checkpoint format, so vLLM re-applies its
    TPU layout transforms (plain model.load_weights fails on already-fused parameters)."""
    import glob

    from safetensors import safe_open
    from safetensors.torch import load_file
    from vllm.model_executor.model_loader.reload.inplace import reload_weights

    runner = getattr(worker, "model_runner", worker)
    model = runner.get_model() if hasattr(runner, "get_model") else runner.model
    cfg = getattr(runner, "vllm_config", None) or worker.vllm_config
    sd = load_file(path)
    if not hasattr(worker, "_qrl_frozen"):
        frozen = {}
        for f in glob.glob(os.path.join(cfg.model_config.model, "*.safetensors")):
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    if not k.startswith("model.language_model."):
                        frozen[k] = h.get_tensor(k)
        worker._qrl_frozen = frozen
    weights = {**worker._qrl_frozen, **sd}
    try:
        from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
    except ImportError:
        WEIGHT_FLIPPED_ATTR = None
    if WEIGHT_FLIPPED_ATTR:
        # The reload restores checkpoint-layout [n_out, n_in] weights and re-runs
        # process_weights_after_loading, which only transposes to (k, n) when this flag is unset.
        for m in model.modules():
            if getattr(m, WEIGHT_FLIPPED_ATTR, False):
                setattr(m, WEIGHT_FLIPPED_ATTR, False)
    reload_weights(cfg, model, weights_iterator=iter(weights.items()), weights_path=None, is_checkpoint_format=True)
    return len(sd)


def _vllm_param_snapshot(worker):
    runner = getattr(worker, "model_runner", worker)
    model = runner.get_model() if hasattr(runner, "get_model") else runner.model
    return {n: p.detach().to("cpu").clone() for n, p in model.named_parameters()}


class VLLMBackend:
    def __init__(self, model_path, max_model_len, seed, gpu_mem=0.85, max_num_seqs=256, enforce_eager=False):
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        from vllm import LLM

        self.llm = LLM(
            model=model_path, tensor_parallel_size=1, max_model_len=max_model_len, seed=seed,
            gpu_memory_utilization=gpu_mem, max_num_seqs=max_num_seqs, enable_prefix_caching=False,
            enforce_eager=enforce_eager, limit_mm_per_prompt={"image": 0, "video": 0},
        )
        self.tok = self.llm.get_tokenizer()

    def load_weights(self, path):
        self.llm.collective_rpc(_vllm_load_weights, args=(path,))
        try:
            self.llm.reset_prefix_cache()
        except Exception:
            pass

    def generate(self, prompts, group_size, max_tokens, seeds, temperature=1.0, top_p=1.0):
        from vllm import SamplingParams

        # vLLM TorchTPU has no per-request seeds; sampling is seeded once per engine (LLM(seed=...)).
        sp = SamplingParams(n=group_size, temperature=temperature, top_p=top_p, top_k=-1, max_tokens=max_tokens,
                            logprobs=0)
        outs = self.llm.generate(list(prompts), sp, use_tqdm=False)
        res = []
        for o in outs:
            for c in sorted(o.outputs, key=lambda c: c.index):
                lps = [d[t].logprob for d, t in zip(c.logprobs, c.token_ids)] if c.logprobs is not None else None
                res.append({
                    "prompt_ids": list(o.prompt_token_ids), "completion_ids": list(c.token_ids), "behav_logp": lps,
                    "text": c.text, "truncated": c.finish_reason == "length",
                })
        return res


def make_backend(args):
    seed = args.seed + 7919 * args.shard_index
    if args.backend == "vllm":
        return VLLMBackend(args.model, args.max_model_len, seed, gpu_mem=args.gpu_mem,
                           max_num_seqs=args.max_num_seqs, enforce_eager=args.enforce_eager)
    return HFBackend(args.model, args.max_model_len, seed)


def parse_schedule(spec, default):
    """'0:4096,30:8192' -> step -> max_tokens (piecewise constant from each start step)."""
    if not spec:
        return lambda b: default
    pts = sorted((int(s), int(v)) for s, v in (p.split(":") for p in spec.split(",")))

    def at(b):
        val = pts[0][1]
        for s, v in pts:
            if b >= s:
                val = v
        return val
    return at


def target_version(b, k, mode, latest):
    """Policy version to sample batch b with: bounded = newest available once >= b-k is out;
    fixed = exactly max(0, b-k), so every batch is exactly k versions stale (after warm-up)."""
    need = max(0, b - k)
    return need if mode == "fixed" else max(need, latest)


def _dom_stats(seqs, d):
    sub = [s for s in seqs if s.get("domain") == d]
    if not sub:
        return {}
    return {"reward": sum(s["reward"] for s in sub) / len(sub), "truncated": sum(s["truncated"] for s in sub) / len(sub),
            "len": sum(len(s["completion_ids"]) for s in sub) / len(sub)}


def teacher_weights_path(spec):
    return os.path.join(spec, "model.safetensors") if os.path.isdir(spec) else spec


def teacher_gen_loop(args, ch, ev, backend, by_dom, spec, doms, shard, max_tokens_at):
    """Batches of teacher samples on the same prompt stream as the on-policy arms (data.mixed_batch, same seeds).
    Per block of batches, each domain's teacher is loaded once and samples all of that domain's prompts in one call;
    the batches are then written in order, each keeping mixed_batch's row order and group layout."""
    from .teacher import parse_teachers

    teachers = parse_teachers(args.teacher_gen)
    missing = [d for d in doms if d not in teachers]
    assert not missing, f"--teacher-gen has no teacher for {missing}"
    loaded, b = None, 0
    blocks = []
    while b < args.steps:
        n = 1 if not blocks else args.teacher_gen_block
        blocks.append(list(range(b, min(args.steps, b + n))))
        b += n
    for blk in blocks:
        rows = {}
        for bb in blk:
            rows_all = data.mixed_batch(by_dom, spec, bb, args.seed)
            rows[bb] = rows_all[args.shard_index::args.num_shards]
        seqs_of = {bb: [None] * len(rows[bb]) for bb in blk}
        tb = time.time()
        gen_s = {}
        for d in doms:
            if loaded != d:
                ts = time.time()
                backend.load_weights(teacher_weights_path(teachers[d]))
                loaded = d
                ev.interval("load_weights", ts, time.time(), teacher=d)
            idx = [(bb, i) for bb in blk for i, r in enumerate(rows[bb]) if data.domain_of(r) == d]
            if not idx:
                continue
            prompts = [data.row_prompt(backend.tok, rows[bb][i], args.enable_thinking) for bb, i in idx]
            seeds = [(args.seed * 1_000_003 + bb * 1009 + (args.shard_index + i * args.num_shards) * 97) % (2**31)
                     for bb, i in idx]
            max_tokens = max_tokens_at(blk[0])
            ts = time.time()
            out = backend.generate(prompts, args.group_size, max_tokens, seeds, temperature=args.temperature)
            gen_s[d] = time.time() - ts
            for j, (bb, i) in enumerate(idx):
                seqs_of[bb][i] = out[j * args.group_size:(j + 1) * args.group_size]
        te = time.time()
        span = (te - tb) / len(blk)
        for k, bb in enumerate(blk):
            seqs = []
            for g, group in enumerate(seqs_of[bb]):
                r = rows[bb][g]
                for s in group:
                    s["group"] = g
                    s["reward"] = data.row_reward(s["text"], r)
                    s["problem_id"] = r["id"]
                    s["domain"] = data.domain_of(r)
                    s["teacher"] = s["domain"]
                    seqs.append(s)
            ch.put_batch(bb, {"idx": bb, "version": 0, "seqs": seqs, "group_size": args.group_size,
                              "gen_start": tb + k * span, "gen_end": tb + (k + 1) * span,
                              "max_tokens": max_tokens_at(bb), "teacher_gen": True}, shard=shard)
            lens = sorted(len(s["completion_ids"]) for s in seqs)
            ev.interval("generate", tb + k * span, tb + (k + 1) * span, batch=bb, version=0, teacher_gen=True,
                        gen_tokens=sum(len(s["completion_ids"]) for s in seqs), max_tokens=max_tokens_at(bb),
                        reward_mean=sum(s["reward"] for s in seqs) / len(seqs),
                        truncated_frac=sum(s["truncated"] for s in seqs) / len(seqs),
                        len_p50=lens[len(lens) // 2], len_p90=lens[int(len(lens) * 0.9)], block=len(blk),
                        **{f"gen_s_{d}": v for d, v in gen_s.items()},
                        **{f"{kk}_{d}": v for d in doms for kk, v in _dom_stats(seqs, d).items()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--staleness", type=int, default=0, help="k; 0 = synchronous")
    ap.add_argument("--prompts-per-batch", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--train-rows", default=None, help="json file of rows (tests); default AIME<=2023")
    ap.add_argument("--test-random-reward", action="store_true", help="tests only: alternate 0/1 rewards")
    ap.add_argument("--staleness-mode", choices=["bounded", "fixed"], default="bounded")
    ap.add_argument("--max-tokens-schedule", default="", help="length curriculum, e.g. '0:4096,30:8192'")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--train-filter", default=None, help="prompt profile JSON from qrl.profile_prompts")
    ap.add_argument("--keep-zero-frac", type=float, default=0.2, help="share of 0/n base prompts kept by the filter")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--domains", default="", help="prompts per batch per domain, e.g. 'math:8,if:8' (MOPD mixed "
                    "batches) or 'if:16'; default: math only, --prompts-per-batch of them")
    ap.add_argument("--teacher-gen", default="", help="offline distillation: math=DIR,if=DIR. Each domain's prompts "
                    "are sampled by that teacher (weights swapped in place), never by the student; student "
                    "weights are ignored")
    ap.add_argument("--teacher-gen-block", type=int, default=4, help="batches per teacher swap (the first block "
                    "is one batch, so the learner starts early)")
    args = ap.parse_args(argv)

    ch = Channel(args.run_dir)
    shard = args.shard_index if args.num_shards > 1 else None
    ev = EventLog(ch, "rollout" if shard is None else f"rollout.s{shard}")
    t0 = time.time()
    backend = make_backend(args)
    ev.interval("startup", t0, time.time())
    spec = args.domains or f"math:{args.prompts_per_batch}"
    doms = [d for d, _ in data.parse_domains(spec)]
    if args.train_rows:
        import json
        rows = json.load(open(args.train_rows))
        by_dom = {d: [r for r in rows if data.domain_of(r) == d] for d in doms}
    else:
        by_dom = {}
        if "math" in doms:
            by_dom["math"] = data.load_train()
        if "if" in doms:
            from . import ifeval

            by_dom["if"] = ifeval.load_train()
    if args.train_filter and "math" in by_dom:
        by_dom["math"] = data.filter_by_profile(by_dom["math"], args.train_filter, args.keep_zero_frac)
    max_tokens_at = parse_schedule(args.max_tokens_schedule, args.max_tokens)
    if args.teacher_gen:
        teacher_gen_loop(args, ch, ev, backend, by_dom, spec, doms, shard, max_tokens_at)
        ev.point("done")
        return
    version = 0
    for b in range(args.steps):
        need = max(0, b - args.staleness)
        ts = time.time()
        wait_until(lambda: ch.latest_version() >= need, ch)
        if time.time() - ts > 1e-3:
            ev.interval("idle_wait_weights", ts, time.time(), batch=b)
        want = target_version(b, args.staleness, args.staleness_mode, ch.latest_version())
        if want != version:
            ts = time.time()
            backend.load_weights(ch.weights_path(want))
            version = want
            ev.interval("load_weights", ts, time.time(), version=version)
        rows_all = data.mixed_batch(by_dom, spec, b, args.seed)
        rows_b = rows_all[args.shard_index::args.num_shards]
        first = args.shard_index
        prompts = [data.row_prompt(backend.tok, r, args.enable_thinking) for r in rows_b]
        seeds = [(args.seed * 1_000_003 + b * 1009 + (first + i * args.num_shards) * 97) % (2**31)
                 for i in range(len(rows_b))]
        max_tokens = max_tokens_at(b)
        ts = time.time()
        seqs = backend.generate(prompts, args.group_size, max_tokens, seeds, temperature=args.temperature)
        te = time.time()
        for i, s in enumerate(seqs):
            g = i // args.group_size
            s["group"] = g
            if args.test_random_reward:
                s["reward"] = float((seeds[g] + i) % 2)
            else:
                s["reward"] = data.row_reward(s["text"], rows_b[g])
            s["problem_id"] = rows_b[g]["id"]
            s["domain"] = data.domain_of(rows_b[g])
        gen_tokens = sum(len(s["completion_ids"]) for s in seqs)
        ch.put_batch(b, {"idx": b, "version": version, "seqs": seqs, "group_size": args.group_size,
                         "gen_start": ts, "gen_end": te, "max_tokens": max_tokens}, shard=shard)
        lens = sorted(len(s["completion_ids"]) for s in seqs)
        ev.interval("generate", ts, te, batch=b, version=version, gen_tokens=gen_tokens, max_tokens=max_tokens,
                    reward_mean=sum(s["reward"] for s in seqs) / len(seqs),
                    truncated_frac=sum(s["truncated"] for s in seqs) / len(seqs),
                    len_p50=lens[len(lens) // 2], len_p90=lens[int(len(lens) * 0.9)],
                    finished_len_mean=(sum(len(s["completion_ids"]) for s in seqs if not s["truncated"])
                                       / max(1, sum(not s["truncated"] for s in seqs))),
                    **{f"{k}_{d}": v for d in doms for k, v in _dom_stats(seqs, d).items()})
    ev.point("done")


if __name__ == "__main__":
    main()
