"""Phase-0 pilot: profile the base model on the training prompts.

Writes, per prompt, the number of correct samples (for the difficulty filter) and, per sample, the
completion length, whether it hit the cap and a repetition score (to tell degenerate loops from long
reasoning). Shard across chips with --shard-index/--num-shards and combine with --merge.

    python -m qrl.profile_prompts --model M --samples 8 --max-tokens 8192 --shard-index 0 --num-shards 3 --out p0.json
    python -m qrl.profile_prompts --merge p0.json p1.json p2.json --out profile.json
    python -m qrl.profile_prompts --model M --timing-probe --out timing.json   # decode s/step vs concurrency
"""
import argparse
import json
import time

from . import data


def repetition_score(ids, window=2048, n=16):
    """Share of repeated n-grams in the last `window` tokens: ~0 for normal text, ->1 for a loop."""
    tail = ids[-window:]
    grams = [tuple(tail[i:i + n]) for i in range(0, max(0, len(tail) - n + 1))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def survival(lens, truncated, caps):
    """P(length > L) for each L in caps, from samples generated with a cap >= max(caps)."""
    tot = len(lens)
    return {str(L): sum(1 for x, t in zip(lens, truncated) if x >= L or t) / tot for L in caps}


def summarize(prompts, cap):
    lens = [x for p in prompts.values() for x in p["lens"]]
    trunc = [t for p in prompts.values() for t in p["truncated"]]
    rep = [r for p in prompts.values() for r, t in zip(p["rep"], p["truncated"]) if t]
    counts = {}
    for p in prompts.values():
        counts[p["correct"]] = counts.get(p["correct"], 0) + 1
    n = next(iter(prompts.values()))["n"] if prompts else 0
    caps = [c for c in (2048, 3072, 4096, 6144, 8192, 12288, 16384) if c <= cap]
    return {
        "prompts": len(prompts), "samples": len(lens), "cap": cap,
        "survival": survival(lens, trunc, caps),
        "truncated_loop_frac": (sum(r > 0.5 for r in rep) / len(rep)) if rep else 0.0,
        "pass_count_hist": {str(k): counts.get(k, 0) for k in range(n + 1)},
        "informative_prompt_frac": sum(v for k, v in counts.items() if 0 < k < n) / max(1, len(prompts)),
        "base_reward": sum(p["correct"] for p in prompts.values()) / max(1, len(lens)),
    }


def timing_probe(be, out, concurrencies=(32, 64, 128, 256), steps=1024):
    """Seconds per decode step at each concurrency, with every sequence forced to `steps` tokens."""
    from vllm import SamplingParams

    prompt = data.chat_prompt(be.tok, "Count upward from 1, one number per line, forever.")
    res = {}
    for c in concurrencies:
        sp = SamplingParams(n=1, temperature=1.0, max_tokens=steps, ignore_eos=True, logprobs=0)
        be.llm.generate([prompt] * 8, SamplingParams(n=1, max_tokens=16), use_tqdm=False)
        t = time.time()
        be.llm.generate([prompt] * c, sp, use_tqdm=False)
        dt = time.time() - t
        res[str(c)] = {"s_per_step": dt / steps, "tok_per_s": c * steps / dt}
        print(c, res[str(c)], flush=True)
    json.dump(res, open(out, "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=9728)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts of the shard (tests)")
    ap.add_argument("--train-rows", default=None)
    ap.add_argument("--timing-probe", action="store_true")
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.merge is not None:
        parts = [json.load(open(p)) for p in args.merge]
        prompts = {k: v for p in parts for k, v in p["prompts"].items()}
        cap = parts[0]["summary"]["cap"]
        res = {"prompts": prompts, "summary": summarize(prompts, cap), "gen_s": sum(p.get("gen_s", 0) for p in parts)}
        json.dump(res, open(args.out, "w"))
        print(json.dumps(res["summary"], indent=1))
        return

    from .rollout import HFBackend, VLLMBackend

    be = (VLLMBackend(args.model, args.max_model_len, args.seed + 7919 * args.shard_index)
          if args.backend == "vllm" else HFBackend(args.model, args.max_model_len, args.seed))
    if args.timing_probe:
        return timing_probe(be, args.out)
    rows = json.load(open(args.train_rows)) if args.train_rows else data.load_train()
    rows = rows[args.shard_index::args.num_shards]
    if args.limit:
        rows = rows[:args.limit]
    prompts = [data.chat_prompt(be.tok, r["question"]) for r in rows]
    t = time.time()
    seqs = be.generate(prompts, args.samples, args.max_tokens, [args.seed + i for i in range(len(rows))],
                       temperature=args.temperature)
    gen_s = time.time() - t
    out = {}
    for i, r in enumerate(rows):
        grp = seqs[i * args.samples:(i + 1) * args.samples]
        out[str(r["id"])] = {
            "n": args.samples, "correct": int(sum(data.reward(s["text"], r["answer"]) for s in grp)),
            "lens": [len(s["completion_ids"]) for s in grp], "truncated": [bool(s["truncated"]) for s in grp],
            "rep": [round(repetition_score(s["completion_ids"]), 3) for s in grp],
        }
    res = {"prompts": out, "summary": summarize(out, args.max_tokens), "gen_s": gen_s,
           "gen_tokens": sum(len(s["completion_ids"]) for s in seqs)}
    json.dump(res, open(args.out, "w"))
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
