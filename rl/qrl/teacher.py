"""Teacher prefill service for on-policy distillation (MOPD, OPD), on its own chip.

For each rollout batch b it routes every sequence to the teacher of its domain (or to the single teacher "all"),
runs one forward pass over prompt + completion, and writes scored/b with the teacher's log-prob of every sampled
completion token (s["teacher_logp"]). With --ship-hidden all (full-distribution arm) it also writes the teacher's
final hidden state (after the final norm) at every completion position (bf16 on TPU), to scored/b.hid.pt, and once at
startup each teacher's output layer to teacher_heads/<name>.safetensors, so the learner can rebuild the exact
teacher logits. --ship-hidden diag does that only every --diag-every batches for a few sequences per domain
(exact-KL diagnostics in the sampled-token arm). Every --diag-every batches it also scores each sequence with the
teacher(s) it was not routed to (KL to the other teacher). Teacher outputs depend only on the tokens, so the lag
between student versions does not touch them.

Sharding: with --num-shards n, shard i scores sequences i::n of every batch and writes scored/b.s<i>; the rollout
batch is kept for the other shards and the learner deletes it.
"""
import argparse
import os
import time

import torch

from .channel import Channel, EventLog, wait_until
from .learner import _plan_microbatches, get_device


def parse_teachers(spec):
    out = {}
    for part in spec.split(","):
        name, path = part.split("=", 1)
        out[name] = path
    return out


def route(seq, names):
    d = seq.get("domain", "math")
    if d in names:
        return d
    if "all" in names:
        return "all"
    raise KeyError(f"no teacher for domain {d} (teachers {sorted(names)})")


def score_seqs(model, seqs, dev, args, want_hidden=()):
    """Teacher log-prob of every completion token of each sequence; hidden states for the indices in want_hidden.
    Positions are gathered across the microbatch and scored in fixed chunks, so the [chunk, vocab] fp32 tensor
    stays small whatever the row count, and every shape repeats."""
    lengths = [len(s["prompt_ids"]) + len(s["completion_ids"]) for s in seqs]
    lps, hids = [None] * len(seqs), {}
    lm_head = model.get_output_embeddings()
    tokens = 0
    for idx, L, rows in _plan_microbatches(lengths, None, args.len_quantum, args.max_model_len, args.mb_tokens):
        ids = torch.full((rows, L), args.pad_id, dtype=torch.long)
        tgt = torch.full((rows, L), args.pad_id, dtype=torch.long)
        pos, spans = [], []
        for r, i in enumerate(idx):
            p, c = seqs[i]["prompt_ids"], seqs[i]["completion_ids"]
            full = (p + c)[:L + 1]
            n = len(full) - 1
            ids[r, :n] = torch.tensor(full[:-1])
            tgt[r, :n] = torch.tensor(full[1:])
            start = len(p) - 1
            m = max(0, min(len(c), L - start))
            spans.append((i, len(pos), m))
            pos += range(r * L + start, r * L + start + m)
        M = len(pos)
        tokens += M
        Mp = -(-max(M, 1) // args.chunk) * args.chunk
        P = torch.tensor(pos + [0] * (Mp - M), dtype=torch.long)
        with torch.no_grad():
            hid = model.model(input_ids=ids.to(dev)).last_hidden_state
            flat = hid.reshape(-1, hid.shape[-1])
            sel = flat.index_select(0, P.to(dev))
            y = tgt.reshape(-1).index_select(0, P).to(dev)
            out = []
            for c in range(0, Mp, args.chunk):
                lp = torch.log_softmax(lm_head(sel[c:c + args.chunk]).float(), -1)
                out.append(torch.gather(lp, -1, y[c:c + args.chunk].unsqueeze(-1)).squeeze(-1))
            lp = torch.cat(out)[:M].to("cpu")
            sel_cpu = sel[:M].to("cpu") if any(i in want_hidden for i, _, _ in spans) else None
        for i, o, m in spans:
            lps[i] = lp[o:o + m].tolist()
            if i in want_hidden:
                hids[i] = sel_cpu[o:o + m].clone()
    return lps, hids, tokens


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", required=True, help="student checkpoint dir (tokenizer, pad id)")
    ap.add_argument("--teachers", required=True, help="math=DIR,if=DIR or all=DIR (HF model dirs)")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="tpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--mb-tokens", type=int, default=8192)
    ap.add_argument("--len-quantum", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=1024, help="positions per log-softmax chunk")
    ap.add_argument("--gdn-chunk", type=int, default=0)
    ap.add_argument("--ship-hidden", choices=["none", "all", "diag"], default="none")
    ap.add_argument("--diag-every", type=int, default=10)
    ap.add_argument("--diag-seqs", type=int, default=4, help="sequences per domain with hidden states (diag)")
    ap.add_argument("--rollout-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--first-step", type=int, default=0)
    args = ap.parse_args(argv)

    from . import gdn_patch

    gdn_desc = gdn_patch.apply(args.gdn_chunk or None, None, False)
    ch = Channel(args.run_dir)
    sh = args.shard_index if args.num_shards > 1 else None
    ev = EventLog(ch, "teacher" if sh is None else f"teacher.s{sh}")
    t0 = time.time()
    dev = get_device(args.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    args.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    paths = parse_teachers(args.teachers)
    models = {}
    for name, path in paths.items():
        models[name] = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, args.dtype)).to(dev).eval()
    if args.ship_hidden != "none" and sh in (None, 0):
        for name, m in models.items():
            ch.put_head(name, m.get_output_embeddings().weight.detach().to("cpu").contiguous())
    ev.interval("startup", t0, time.time(), gdn=gdn_desc, teachers=paths, ship_hidden=args.ship_hidden)

    for b in range(args.first_step, args.steps):
        ts = time.time()
        wait_until(lambda: ch.batch_ready(b, args.rollout_shards), ch)
        if time.time() - ts > 1e-3:
            ev.interval("idle_wait_batch", ts, time.time(), batch=b)
        keep = args.num_shards > 1
        batch = ch.take_batch(b, args.rollout_shards, keep=keep)
        t1 = time.time()
        seqs = batch["seqs"]
        mine = list(range(len(seqs)))[args.shard_index::args.num_shards]
        diag = args.diag_every > 0 and b % args.diag_every == 0
        by_t = {}
        for i in mine:
            by_t.setdefault(route(seqs[i], models), []).append(i)
        want = set()
        if args.ship_hidden == "all":
            want = set(mine)
        elif args.ship_hidden == "diag" and diag:
            for ix in by_t.values():
                want |= set(ix[:args.diag_seqs])
        hidden, stats, tokens = {}, {}, 0
        for name, ix in by_t.items():
            sub = [seqs[i] for i in ix]
            lps, hids, n = score_seqs(models[name], sub, dev, args, {j for j, i in enumerate(ix) if i in want})
            tokens += n
            kl_sum = kl_n = 0.0
            for j, i in enumerate(ix):
                s = seqs[i]
                s["teacher_logp"], s["teacher"] = lps[j], name
                beh = s["behav_logp"][:len(lps[j])]
                kl_sum += sum(bb - tt for bb, tt in zip(beh, lps[j]))
                kl_n += len(beh)
                if j in hids:
                    hidden[f"{b}.{i}"] = hids[j]
                    s["hid"] = (None, f"{b}.{i}")
            stats[f"kl_behav_teacher_{name}"] = kl_sum / max(1.0, kl_n)
            stats[f"tokens_{name}"] = kl_n
            if diag and len(models) > 1:
                for other in models:
                    if other == name:
                        continue
                    olps, _, _ = score_seqs(models[other], sub, dev, args)
                    o_sum = sum(sum(bb - tt for bb, tt in zip(seqs[i]["behav_logp"][:len(olps[j])], olps[j]))
                                for j, i in enumerate(ix))
                    stats[f"kl_behav_other_{name}_vs_{other}"] = o_sum / max(1.0, kl_n)
        t2 = time.time()
        out = dict(batch)
        out["seqs"] = [seqs[i] for i in mine]
        if hidden:
            fname = ch.put_hidden(b, hidden, sh)
            for s in out["seqs"]:
                if "hid" in s:
                    s["hid"] = (fname, s["hid"][1])
        ch.put_batch(b, out, shard=sh, stage="scored")
        ev.interval("score", t1, t2, batch=b, tokens=tokens, tok_s=tokens / max(1e-6, t2 - t1), seqs=len(mine),
                    hidden_seqs=len(hidden), write_s=time.time() - t2, diag=diag, **stats)
    ev.point("done")


if __name__ == "__main__":
    main()
