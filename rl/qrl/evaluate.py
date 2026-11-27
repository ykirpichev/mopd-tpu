"""pass@1 (mean over n samples per problem) for saved checkpoints on one vLLM engine, on AIME 2024+2025
(eval set v1) or the 136-problem v2 set, or instruction following: IFEval (541 prompts) / IFBench_test (300),
where a sample counts as correct when it follows every instruction of its prompt (prompt-level strict accuracy).
Shard problems across chips with --shard-index/--num-shards and combine with --merge."""
import argparse
import json
import os
import time

from . import data
from .rollout import HFBackend, VLLMBackend


def metrics(per, samples, eval_s=0.0):
    n = len(per) * samples
    out = {
        "pass@1": sum(p["correct"] for p in per) / n,
        "truncated_frac": sum(p["truncated"] for p in per) / n,
        "mean_len": sum(sum(p["lens"]) for p in per) / n if per and "lens" in per[0] else None,
        "eval_s": eval_s, "samples": samples, "problems": len(per), "per_problem": per,
    }
    for year in ("2024", "2025"):
        sub = [p for p in per if p["id"].startswith(year)]
        if sub:
            out[f"pass@1_{year}"] = sum(p["correct"] for p in sub) / (len(sub) * samples)
    if per and "inst" in per[0]:
        out["inst_acc"] = sum(sum(p["inst"]) for p in per) / sum(len(p["inst"]) * samples for p in per)
    for tag in sorted({p["id"].split("-")[0] for p in per}):
        sub = [p for p in per if p["id"].split("-")[0] == tag]
        out[f"pass@1[{tag}]"] = sum(p["correct"] for p in sub) / (len(sub) * samples)
    return out


def merge(paths, out):
    parts = [json.load(open(p)) for p in paths]
    res = {}
    for label in parts[0]:
        per = [q for p in parts for q in p[label]["per_problem"]]
        res[label] = metrics(per, parts[0][label]["samples"], max(p[label]["eval_s"] for p in parts))
    json.dump(res, open(out, "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="base checkpoint dir (config/tokenizer)")
    ap.add_argument("--ckpts", nargs="+", help="label=path.safetensors, or label=base")
    ap.add_argument("--backend", default="vllm")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--eval-set", choices=["v1", "v2", "ifeval", "ifbench"], default="v1")
    ap.add_argument("--texts", default="", help="also write the first sample of every prompt here (jsonl)")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", nargs="*", default=None, help="combine shard outputs into --out")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.merge is not None:
        for label, r in merge(args.merge, args.out).items():
            print(label, json.dumps({k: v for k, v in r.items() if k != "per_problem"}))
        return
    be = (VLLMBackend(args.model, args.max_model_len, args.seed + 7919 * args.shard_index,
                      max_num_seqs=args.max_num_seqs)
          if args.backend == "vllm"
          else HFBackend(args.model, args.max_model_len, args.seed))
    is_if = args.eval_set in ("ifeval", "ifbench")
    if is_if:
        from . import ifeval

        ifeval.setup_nltk()
        rows = ifeval.load_eval(args.eval_set)
    else:
        rows = data.load_eval(args.eval_set)
    idx = list(range(len(rows)))[args.shard_index::args.num_shards]
    rows = [rows[i] for i in idx]
    prompts = [data.row_prompt(be.tok, r, args.enable_thinking) for r in rows]
    seeds = [args.seed + 1000 * i for i in idx]
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for spec in args.ckpts:
        label, path = spec.split("=", 1)
        if label in results:
            continue
        if path != "base":
            be.load_weights(path)
        t0 = time.time()
        kw = {"temperature": args.temperature, "top_p": args.top_p} if args.backend == "vllm" else {}
        seqs = be.generate(prompts, args.samples, args.max_tokens, seeds, **kw)
        per = []
        texts = []
        for i, r in enumerate(rows):
            group = seqs[i * args.samples:(i + 1) * args.samples]
            q = {"id": r["id"], "truncated": sum(s["truncated"] for s in group),
                 "lens": [len(s["completion_ids"]) for s in group]}
            if is_if:
                fl = [ifeval.follow_list(r, s["text"], args.eval_set) for s in group]
                q["correct"] = sum(all(f) for f in fl)
                q["inst"] = [sum(f[j] for f in fl) for j in range(len(r["instruction_id_list"]))]
            else:
                q["correct"] = sum(data.reward(s["text"], r["answer"]) for s in group)
            per.append(q)
            texts.append({"label": label, "id": r["id"], "correct0": (all(fl[0]) if is_if else
                          bool(data.reward(group[0]["text"], r["answer"]))), "text": group[0]["text"][:6000]})
        if args.texts:
            with open(args.texts, "a") as f:
                for t in texts:
                    f.write(json.dumps(t) + "\n")
        results[label] = metrics(per, args.samples, time.time() - t0)
        json.dump(results, open(args.out, "w"), indent=1)
        print(label, json.dumps({k: v for k, v in results[label].items() if k != "per_problem"}), flush=True)


if __name__ == "__main__":
    main()
