"""Learner process: TorchTPU (device "tpu") or CPU trainer for the GRPO loop.

One optimizer step per rollout batch. The learner never holds an old-policy
copy: the loss compares its own forward pass with the behavior log-probs the
rollout engine recorded. Master weights and AdamW state live in fp32 on the
host; the device holds bf16 weights and gradients.
"""
import argparse
import json
import math
import os
import time

import torch

from .channel import Channel, EventLog, wait_until
from .grpo import LossConfig, behavior_diagnostics, group_advantages, policy_loss, recomputed_old_loss


def get_device(name):
    if name == "tpu":
        import torch_tpu  # noqa: F401  registers the "tpu" device

        return torch.device("tpu")
    return torch.device(name)


def checkpoint_keys(model_path):
    idx = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(idx):
        return set(json.load(open(idx))["weight_map"])
    from safetensors import safe_open

    with safe_open(os.path.join(model_path, "model.safetensors"), "pt") as f:
        return set(f.keys())


def export_name_map(param_names, ckpt_keys):
    """Map trainer parameter names to the checkpoint's names, which vLLM's loader expects."""
    out = {}
    for n in param_names:
        cands = [n]
        if n.startswith("model."):
            cands.append("model.language_model." + n[len("model."):])
        hit = next((c for c in cands if c in ckpt_keys), None)
        if hit is None:
            if n == "lm_head.weight":
                continue  # tied to embed_tokens
            raise KeyError(f"no checkpoint key for {n}")
        out[n] = hit
    return out


def token_logprobs(model, input_ids, targets, chunk=512, with_entropy=False):
    """Log-probs of targets[:, t] given input_ids[:, :t+1], chunked over the vocab projection: each chunk holds
    about `chunk` positions over all rows (a 16-row microbatch of short sequences would otherwise materialize
    16 x chunk x 248k fp32 logits, ~8 GB). with_entropy also returns the per-position policy entropy (no grad;
    one extra lm_head pass)."""
    from torch.utils.checkpoint import checkpoint

    hidden = model.model(input_ids=input_ids).last_hidden_state
    lm_head = model.get_output_embeddings()

    def piece(h, y):
        logits = lm_head(h).float()
        return torch.gather(logits.log_softmax(-1), -1, y.unsqueeze(-1)).squeeze(-1)

    outs, ents = [], []
    step = max(16, 1 << (max(1, chunk // hidden.shape[0]).bit_length() - 1))  # a power of 2 divides the bucketed lengths
    for s in range(0, hidden.shape[1], step):
        h, y = hidden[:, s:s + step], targets[:, s:s + step]
        outs.append(checkpoint(piece, h, y, use_reentrant=False) if torch.is_grad_enabled() else piece(h, y))
        if with_entropy:
            with torch.no_grad():
                lp_all = lm_head(h.detach()).float().log_softmax(-1)
                ents.append(-(lp_all.exp() * lp_all).sum(-1))
    lp = torch.cat(outs, dim=1)
    return (lp, torch.cat(ents, dim=1)) if with_entropy else lp


def aggregate_stats(stats_acc):
    """Token-weighted means across microbatches; *_max / *_min keys take the extreme."""
    if not stats_acc:
        return {}
    tot = sum(s["tokens"] for s in stats_acc) or 1.0
    out = {}
    for k in stats_acc[0]:
        if k == "tokens":
            continue
        vals = [s[k] for s in stats_acc if k in s]
        if k.endswith("_max"):
            out[k] = max(vals)
        elif k.endswith("_min"):
            out[k] = min(vals)
        else:
            out[k] = sum(s[k] * s["tokens"] for s in stats_acc if k in s) / tot
    return out


def bucket_len(n, quantum, cap):
    return min(cap, int(math.ceil(n / quantum) * quantum))


def _plan_microbatches(lengths, mb_size, quantum, cap, mb_tokens=None, max_rows=16):
    """Groups of indices (sorted by length) and their padded length. With mb_tokens, each group holds
    as many rows as fit in the token budget at its bucketed length, so short sequences share a dispatch."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    plan, s = [], 0
    while s < len(order):
        if mb_tokens:
            # Sorted ascending, so the group's padded length is set by its last member.
            rows = 1
            while (s + rows < len(order) and rows < max_rows
                   and (rows + 1) * bucket_len(lengths[order[s + rows]], quantum, cap) <= mb_tokens):
                rows += 1
        else:
            rows = mb_size
        idx = order[s:s + rows]
        plan.append((idx, bucket_len(max(lengths[i] for i in idx), quantum, cap), rows))
        s += rows
    return plan


def make_microbatches(seqs, adv, mb_size, quantum, cap, pad_id, mb_tokens=None):
    """Right-padded fixed-shape microbatches. Causal attention makes right padding harmless."""
    lengths = [len(s["prompt_ids"]) + len(s["completion_ids"]) for s in seqs]
    for idx, L, rows in _plan_microbatches(lengths, mb_size, quantum, cap, mb_tokens):
        mb_size = rows
        ids = torch.full((mb_size, L), pad_id, dtype=torch.long)
        tgt = torch.full((mb_size, L), pad_id, dtype=torch.long)
        behav = torch.zeros((mb_size, L))
        mask = torch.zeros((mb_size, L))
        a = torch.zeros(mb_size)
        for r, i in enumerate(idx):
            p, c = seqs[i]["prompt_ids"], seqs[i]["completion_ids"]
            full = (p + c)[:L + 1]
            n = len(full) - 1
            ids[r, :n] = torch.tensor(full[:-1])
            tgt[r, :n] = torch.tensor(full[1:])
            start = len(p) - 1
            m = min(len(c), L - start)
            behav[r, start:start + m] = torch.tensor(seqs[i]["behav_logp"][:m])
            mask[r, start:start + m] = 1.0
            a[r] = adv[i]
        yield ids, tgt, behav, mask, a


class HostAdamW:
    """fp32 master weights + AdamW on the host; device params are bf16 copies."""

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.dev = list(params)
        self.master = [p.detach().to("cpu", torch.float32).clone().requires_grad_(True) for p in self.dev]
        self.opt = torch.optim.AdamW(self.master, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def step(self, max_grad_norm):
        for m, p in zip(self.master, self.dev):
            m.grad = None if p.grad is None else p.grad.detach().to("cpu", torch.float32)
        gn = torch.nn.utils.clip_grad_norm_([m for m in self.master if m.grad is not None], max_grad_norm)
        self.opt.step()
        with torch.no_grad():
            for m, p in zip(self.master, self.dev):
                p.copy_(m.detach().to(p.device, p.dtype))
                p.grad = None
        return float(gn)


def train_on_batch(args, model, fwd, opt, kseqs, kadv, loss_cfg, pad_id, dev, step):
    """One rollout batch -> args.updates_per_batch optimizer steps on random disjoint minibatches.

    behavior_logprobs="use": truncated-IS (or PPO) against the sampler's log-probs (baseline).
    behavior_logprobs="none": PPO clip against log-probs recomputed by the trainer before the first
    update; with one update that is lp.detach() (ratio == 1). Sampler log-probs are only diagnostics."""
    U = max(1, min(args.updates_per_batch, len(kseqs)))
    perm = torch.randperm(len(kseqs), generator=torch.Generator().manual_seed(args.seed * 100_003 + step)).tolist()
    parts = [perm[u::U] for u in range(U)]

    def mbs(part):
        ps, pa = [kseqs[i] for i in part], kadv[part]
        return list(make_microbatches(ps, pa, args.micro_batch, args.len_quantum, args.max_model_len, pad_id,
                                      args.mb_tokens))

    part_mbs = [mbs(p) for p in parts]
    old = None
    if args.behavior_logprobs == "none" and U > 1:
        with torch.no_grad():
            old = [[fwd(model, ids.to(dev), tgt.to(dev)).detach() for ids, tgt, *_ in pm] for pm in part_mbs]
    stats_acc, gns = [], []
    for u, (part, pm) in enumerate(zip(parts, part_mbs)):
        denom = torch.tensor(float(sum(min(len(kseqs[i]["completion_ids"]), args.max_model_len) for i in part)))
        for j, (ids, tgt, behav, mask, a) in enumerate(pm):
            ids, tgt, behav, mask, a = (x.to(dev) for x in (ids, tgt, behav, mask, a))
            out = fwd(model, ids, tgt, with_entropy=True) if args.log_entropy else fwd(model, ids, tgt)
            lp, ent = out if args.log_entropy else (out, None)
            if args.behavior_logprobs == "use":
                loss, st = policy_loss(lp, behav, a, mask, loss_cfg, denom=denom.to(dev))
            else:
                ref = lp.detach() if old is None else old[u][j]
                loss, st = recomputed_old_loss(lp, ref, a, mask, loss_cfg, denom=denom.to(dev))
            st.update(behavior_diagnostics(lp, behav, mask, loss_cfg.is_cap))
            if ent is not None:
                st["entropy"] = float((ent * mask).sum() / mask.sum().clamp(min=1.0))
            loss.backward()
            stats_acc.append(st)
        gns.append(opt.step(args.max_grad_norm))
    return stats_acc, gns


def distill_microbatches(seqs, quantum, cap, pad_id, mb_tokens):
    """Like make_microbatches, plus the shipped teacher log-probs on the same positions, and per row the
    sequence index, its first completion position and its completion-token count."""
    lengths = [len(s["prompt_ids"]) + len(s["completion_ids"]) for s in seqs]
    for idx, L, rows in _plan_microbatches(lengths, None, quantum, cap, mb_tokens):
        ids = torch.full((rows, L), pad_id, dtype=torch.long)
        tgt = torch.full((rows, L), pad_id, dtype=torch.long)
        behav, tlp, mask = torch.zeros((rows, L)), torch.zeros((rows, L)), torch.zeros((rows, L))
        starts, ms = [], []
        for r, i in enumerate(idx):
            p, c = seqs[i]["prompt_ids"], seqs[i]["completion_ids"]
            full = (p + c)[:L + 1]
            n = len(full) - 1
            ids[r, :n] = torch.tensor(full[:-1])
            tgt[r, :n] = torch.tensor(full[1:])
            start = len(p) - 1
            m = min(len(c), L - start, len(seqs[i].get("teacher_logp") or c))
            behav[r, start:start + m] = torch.tensor(seqs[i]["behav_logp"][:m])
            if seqs[i].get("teacher_logp") is not None:
                tlp[r, start:start + m] = torch.tensor(seqs[i]["teacher_logp"][:m])
            mask[r, start:start + m] = 1.0
            starts.append(start)
            ms.append(m)
        yield {"ids": ids, "tgt": tgt, "behav": behav, "tlp": tlp, "mask": mask, "idx": idx, "L": L,
               "starts": starts, "ms": ms}


class DomainSums:
    """Token-weighted per-domain means of per-token quantities across microbatches."""

    def __init__(self):
        self.s = {}

    def add(self, key, dom, total, n):
        k = f"{key}_{dom}"
        a = self.s.setdefault(k, [0.0, 0.0])
        a[0] += float(total)
        a[1] += float(n)

    def out(self):
        return {k: v[0] / v[1] for k, v in self.s.items() if v[1] > 0}


def dist_microbatch(model, mb, seqs, hid, heads, dev, denom, chunk, ds, grad=True):
    """Exact full-vocabulary reverse KL to the routed teacher on one microbatch, from shipped teacher hidden states.
    Positions are flattened over rows, grouped by teacher, padded to whole chunks (pads weigh 0), and each chunk's
    [chunk, vocab] fp32 work is recomputed in backward (checkpoint), so memory does not grow with the row count.
    Returns the loss (sum over positions / denom; None without grad) and token-weighted stats."""
    from torch.utils.checkpoint import checkpoint

    from .grpo import reverse_kl_piece

    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        hidden = model.model(input_ids=mb["ids"].to(dev)).last_hidden_state
        H = hidden.shape[-1]
        flat = hidden.reshape(-1, H)
        w_s = model.get_output_embeddings().weight
        L = mb["L"]
        groups = {}
        for r, (i, st, m) in enumerate(zip(mb["idx"], mb["starts"], mb["ms"])):
            if m <= 0 or "hid" not in seqs[i]:
                continue
            groups.setdefault(seqs[i]["teacher"], []).append((r, i, st, m))
        loss, stats = None, {"tokens": 0.0}
        acc = {"kl_exact": 0.0, "kl_teacher": 0.0, "entropy": 0.0, "hid_check_abs": 0.0}
        hc_max, ratios = 0.0, []
        for tname, rows in groups.items():
            pos, ys, hts, beh, tl, doms = [], [], [], [], [], []
            for r, i, st, m in rows:
                h_t = hid[seqs[i]["hid"][1]]
                m = min(m, h_t.shape[0])
                pos += range(r * L + st, r * L + st + m)
                ys.append(mb["tgt"][r, st:st + m])
                hts.append(h_t[:m])
                beh.append(mb["behav"][r, st:st + m])
                tl.append(mb["tlp"][r, st:st + m])
                doms += [seqs[i].get("domain", "math")] * m
            M = len(pos)
            Mp = -(-M // chunk) * chunk
            P = torch.tensor(pos + [0] * (Mp - M), dtype=torch.long).to(dev)
            Y = torch.cat(ys + [torch.zeros(Mp - M, dtype=torch.long)]).to(dev)
            HT = torch.cat(hts + [torch.zeros(Mp - M, H, dtype=hts[0].dtype)]).to(dev)
            W = torch.cat([torch.ones(M), torch.zeros(Mp - M)]).to(dev)
            kls, lps, ents, tlps = [], [], [], []
            for c in range(0, Mp, chunk):
                h_c = flat.index_select(0, P[c:c + chunk])
                args_ = (h_c, HT[c:c + chunk], w_s, heads[tname], Y[c:c + chunk])
                kl, lp, ent, tlp_re = (checkpoint(reverse_kl_piece, *args_, use_reentrant=False) if grad
                                       else reverse_kl_piece(*args_))
                part = (kl * W[c:c + chunk]).sum() / denom
                loss = part if loss is None else loss + part
                kls.append(kl.detach())
                lps.append(lp)
                ents.append(ent)
                tlps.append(tlp_re)
            kl_all = torch.cat(kls)[:M].float().cpu()
            lp_all = torch.cat(lps)[:M].float().cpu()
            ent_all = torch.cat(ents)[:M].float().cpu()
            tlp_re = torch.cat(tlps)[:M].float().cpu()
            beh_all, tl_all = torch.cat(beh), torch.cat(tl)
            dk = lp_all - tl_all
            hc = (tlp_re - tl_all).abs()
            acc["kl_exact"] += float(kl_all.sum())
            acc["kl_teacher"] += float(dk.sum())
            acc["entropy"] += float(ent_all.sum())
            acc["hid_check_abs"] += float(hc.sum())
            hc_max = max(hc_max, float(hc.max()))
            stats["tokens"] += M
            ratios.append((lp_all - beh_all).clamp(-20, 20))
            for d in set(doms):
                sel = torch.tensor([x == d for x in doms])
                ds.add("kl_exact", d, kl_all[sel].sum(), sel.sum())
                ds.add("kl_teacher", d, dk[sel].sum(), sel.sum())
                ds.add("entropy", d, ent_all[sel].sum(), sel.sum())
        n = max(1.0, stats["tokens"])
        stats.update({k: v / n for k, v in acc.items()})
        stats["hid_check_max"] = hc_max
        if ratios:
            lr_ = torch.cat(ratios)
            r = lr_.exp()
            stats.update({"behav_ratio_mean": float(r.mean()), "behav_ratio_max": float(r.max()),
                          "kl_behav_cur": float((r - 1 - lr_).mean())})
    return loss, stats


def load_hidden(ch, seqs):
    hid = {}
    for name in sorted({s["hid"][0] for s in seqs if "hid" in s}):
        hid.update(ch.take_hidden(name))
    return hid


def load_heads(ch, names, dev, timeout=3600):
    from safetensors.torch import load_file

    for n in names:
        wait_until(lambda: os.path.exists(ch.head_path(n)), ch, timeout=timeout)
    return {n: load_file(ch.head_path(n))["weight"].to(dev) for n in names}


def train_distill(args, model, fwd, opt, seqs, hid, heads, loss_cfg, pad_id, dev):
    """One optimizer step on every sequence of the batch (every token carries a teacher signal).
    opd: truncated-IS policy gradient with the per-token advantage clip(log pi_T - log pi_S, +-adv_clip).
    opd_dist: exact per-position reverse KL over the vocabulary to the routed teacher (no clip, no IS weight)."""
    from .grpo import distill_advantage

    denom = torch.tensor(float(sum(min(len(s["completion_ids"]), args.max_model_len) for s in seqs))).to(dev)
    stats_acc, ds = [], DomainSums()
    for mb in distill_microbatches(seqs, args.len_quantum, args.max_model_len, pad_id, args.mb_tokens):
        if args.objective == "opd_dist":
            loss, st = dist_microbatch(model, mb, seqs, hid, heads, dev, denom, args.dist_chunk, ds, grad=True)
            if loss is not None:
                loss.backward()
            stats_acc.append(st)
            continue
        ids, tgt, behav, tlp, mask = (mb[k].to(dev) for k in ("ids", "tgt", "behav", "tlp", "mask"))
        out = fwd(model, ids, tgt, with_entropy=True) if args.log_entropy else fwd(model, ids, tgt)
        lp, ent = out if args.log_entropy else (out, None)
        adv, dst = distill_advantage(lp, tlp, mask, args.adv_clip)
        loss, st = policy_loss(lp, behav, adv, mask, loss_cfg, denom=denom)
        st.update(behavior_diagnostics(lp, behav, mask, loss_cfg.is_cap))
        st.update(dst)
        with torch.no_grad():
            per_row = ((lp.detach() - tlp) * mask).sum(1).cpu()
            n_row = mask.sum(1).cpu()
            ent_row = (ent * mask).sum(1).cpu() if ent is not None else None
        for r, i in enumerate(mb["idx"]):
            d = seqs[i].get("domain", "math")
            ds.add("kl_teacher", d, per_row[r], n_row[r])
            if ent_row is not None:
                ds.add("entropy", d, ent_row[r], n_row[r])
        if ent is not None:
            st["entropy"] = float((ent * mask).sum() / mask.sum().clamp(min=1.0))
        loss.backward()
        stats_acc.append(st)
    return stats_acc, [opt.step(args.max_grad_norm)], ds.out()


def train_sft(args, model, fwd, opt, seqs, pad_id, dev):
    """Offline distillation (sequence-level KD): one optimizer step of token-level cross-entropy on every
    completion of the batch. The completions were sampled by the routed teacher (rollout --teacher-gen), so
    behav_logp is the teacher's own log-prob of each token and mean(behav - lp) estimates KL(pi_T || pi_S)."""
    denom = torch.tensor(float(sum(min(len(s["completion_ids"]), args.max_model_len) for s in seqs))).to(dev)
    stats_acc, ds = [], DomainSums()
    for mb in distill_microbatches(seqs, args.len_quantum, args.max_model_len, pad_id, args.mb_tokens):
        ids, tgt, behav, mask = (mb[k].to(dev) for k in ("ids", "tgt", "behav", "mask"))
        out = fwd(model, ids, tgt, with_entropy=True) if args.log_entropy else fwd(model, ids, tgt)
        lp, ent = out if args.log_entropy else (out, None)
        loss = -(lp * mask).sum() / denom
        with torch.no_grad():
            n = mask.sum().clamp(min=1.0)
            d = (behav - lp.detach()) * mask
            st = {"tokens": float(n), "ce": float(-(lp.detach() * mask).sum() / n), "fkl_teacher": float(d.sum() / n)}
            if ent is not None:
                st["entropy"] = float((ent * mask).sum() / n)
            per_row, n_row = d.sum(1).cpu(), mask.sum(1).cpu()
            ce_row = (-(lp.detach() * mask)).sum(1).cpu()
            ent_row = (ent * mask).sum(1).cpu() if ent is not None else None
        for r, i in enumerate(mb["idx"]):
            dom = seqs[i].get("domain", "math")
            ds.add("fkl_teacher", dom, per_row[r], n_row[r])
            ds.add("ce", dom, ce_row[r], n_row[r])
            if ent_row is not None:
                ds.add("entropy", dom, ent_row[r], n_row[r])
        loss.backward()
        stats_acc.append(st)
    return stats_acc, [opt.step(args.max_grad_norm)], ds.out()


def exact_kl_diag(args, model, seqs, hid, heads, pad_id, dev):
    """Exact reverse KL (no grad) for the sequences that came with hidden states (sampled-token arm diagnostics)."""
    sub = [s for s in seqs if "hid" in s]
    if not sub:
        return {}
    ds, acc, n = DomainSums(), 0.0, 0.0
    for mb in distill_microbatches(sub, args.len_quantum, args.max_model_len, pad_id, args.mb_tokens):
        _, st = dist_microbatch(model, mb, sub, hid, heads, dev, torch.tensor(1.0).to(dev), args.dist_chunk, ds,
                                grad=False)
        acc += st["kl_exact"] * st["tokens"]
        n += st["tokens"]
    return {"diag_kl_exact": acc / max(1.0, n), "diag_tokens": n,
            **{f"diag_{k}": v for k, v in ds.out().items() if k.startswith("kl_exact")}}


def domain_batch_stats(seqs):
    out = {}
    for d in sorted({s.get("domain", "math") for s in seqs}):
        sub = [s for s in seqs if s.get("domain", "math") == d]
        out[f"reward_{d}"] = sum(s["reward"] for s in sub) / len(sub)
        out[f"truncated_{d}"] = sum(bool(s.get("truncated")) for s in sub) / len(sub)
        out[f"len_{d}"] = sum(len(s["completion_ids"]) for s in sub) / len(sub)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="tpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--len-quantum", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--loss", default="tis", choices=["tis", "ppo"])
    ap.add_argument("--is-cap", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action="store_true", help="torch.compile(backend='tpu') the forward")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--ckpt-steps", default="", help="comma-separated optimizer steps to save for eval")
    ap.add_argument("--ckpt-every", type=int, default=0, help="also save every N steps")
    ap.add_argument("--mb-tokens", type=int, default=0, help="token budget per microbatch (0: fixed --micro-batch)")
    ap.add_argument("--gdn-chunk", type=int, default=0, help="Gated DeltaNet chunk size (0: transformers default)")
    ap.add_argument("--gdn-solve", default="", choices=["", "triangular", "neumann"])
    ap.add_argument("--gdn-contig-grad", action="store_true", help="contiguous beta/g grads (torch.compile fix)")
    ap.add_argument("--num-shards", type=int, default=1, help="rollout engines contributing to each batch")
    ap.add_argument("--keep-weights", type=int, default=2, help="published versions kept (fixed lag k needs k+2)")
    ap.add_argument("--behavior-logprobs", choices=["use", "none"], default="use",
                    help="none = no-IS ablation: PPO clip against trainer-recomputed log-probs; sampler "
                         "log-probs are only logged as diagnostics")
    ap.add_argument("--updates-per-batch", type=int, default=1,
                    help="optimizer steps per rollout batch (random disjoint minibatches)")
    ap.add_argument("--log-entropy", action="store_true", help="log mean policy entropy on trained tokens")
    ap.add_argument("--objective", choices=["grpo", "opd", "opd_dist", "sft"], default="grpo",
                    help="opd: sampled-token distillation (MOPD/OPD); opd_dist: exact full-vocabulary reverse KL; "
                         "sft: cross-entropy on teacher-generated completions (rollout --teacher-gen)")
    ap.add_argument("--adv-clip", type=float, default=5.0, help="opd advantage clip (MOPD paper: 5)")
    ap.add_argument("--dist-chunk", type=int, default=512, help="opd_dist positions per full-vocabulary chunk")
    ap.add_argument("--teacher-shards", type=int, default=1, help="scored shards per batch (teacher processes)")
    ap.add_argument("--first-step", type=int, default=0, help="tests/bring-up: start at this batch index")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    from . import gdn_patch

    gdn_desc = gdn_patch.apply(args.gdn_chunk or None, args.gdn_solve or None, args.gdn_contig_grad)
    ch = Channel(args.run_dir, keep_weights=args.keep_weights)
    ev = EventLog(ch, "learner")
    t0 = time.time()
    dev = get_device(args.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(dev)
    model.train()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    name_map = export_name_map([n for n, _ in named], checkpoint_keys(args.model))
    opt = HostAdamW([p for _, p in named], lr=args.lr)
    fwd = torch.compile(token_logprobs, backend="tpu", dynamic=False) if args.compile else token_logprobs
    loss_cfg = LossConfig(kind=args.loss, is_cap=args.is_cap)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ckpt_steps = {int(s) for s in args.ckpt_steps.split(",") if s}
    if args.ckpt_every:
        ckpt_steps |= set(range(0, args.steps + 1, args.ckpt_every))
    ev.interval("startup", t0, time.time(), gdn=gdn_desc)

    def export():
        return {name_map[n]: p.detach().to("cpu", torch.bfloat16).contiguous() for n, p in named if n in name_map}

    if 0 in ckpt_steps and args.ckpt_dir:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        from safetensors.torch import save_file
        save_file(export(), os.path.join(args.ckpt_dir, "step00000.safetensors"))

    distill = args.objective in ("opd", "opd_dist")
    heads = None
    for step in range(args.first_step, args.steps):
        ts = time.time()
        if distill:
            wait_until(lambda: ch.batch_ready(step, args.teacher_shards, stage="scored"), ch)
        else:
            wait_until(lambda: ch.batch_ready(step, args.num_shards), ch)
        ev.interval("idle_wait_batch", ts, time.time(), step=step)
        tr0 = time.time()
        extra = {}
        if distill:
            batch = ch.take_batch(step, args.teacher_shards, stage="scored")
            if args.teacher_shards > 1:
                ch.remove_batch(step, args.num_shards)
            seqs, G = batch["seqs"], batch["group_size"]
            hid = load_hidden(ch, seqs)
            extra["load_hidden_s"] = time.time() - tr0
            need = sorted({s["teacher"] for s in seqs if "hid" in s})
            if need and (heads is None or not set(need) <= set(heads)):
                heads = {**(heads or {}), **load_heads(ch, [n for n in need if n not in (heads or {})], dev)}
            if args.objective == "opd" and hid:
                extra.update(exact_kl_diag(args, model, seqs, hid, heads, pad_id, dev))
            rewards = torch.tensor([s["reward"] for s in seqs])
            keep = list(range(len(seqs)))
            stats_acc, gns, dstats = train_distill(args, model, fwd, opt, seqs, hid, heads, loss_cfg, pad_id, dev)
            extra.update(dstats)
            del hid
        elif args.objective == "sft":
            batch = ch.take_batch(step, args.num_shards)
            seqs, G = batch["seqs"], batch["group_size"]
            rewards = torch.tensor([s["reward"] for s in seqs])
            keep = list(range(len(seqs)))
            stats_acc, gns, dstats = train_sft(args, model, fwd, opt, seqs, pad_id, dev)
            extra.update(dstats)
        else:
            batch = ch.take_batch(step, args.num_shards)
            seqs, G = batch["seqs"], batch["group_size"]
            rewards = torch.tensor([s["reward"] for s in seqs]).view(-1, G)
            adv = group_advantages(rewards).flatten()
            keep = [i for i in range(len(seqs)) if abs(float(adv[i])) > 1e-8]
            stats_acc, gns = [], []
            if keep:
                stats_acc, gns = train_on_batch(args, model, fwd, opt, [seqs[i] for i in keep], adv[keep], loss_cfg,
                                                pad_id, dev, step)
        extra.update(domain_batch_stats(seqs))
        version = step + 1
        tr1 = time.time()
        ch.publish_weights(version, export())
        tp1 = time.time()
        if version in ckpt_steps and args.ckpt_dir:
            from safetensors.torch import save_file
            os.makedirs(args.ckpt_dir, exist_ok=True)
            path = os.path.join(args.ckpt_dir, f"step{version:05d}.safetensors")
            save_file(export(), path + ".tmp")
            os.replace(path + ".tmp", path)  # evaluators poll for the file while training goes on
        tot = sum(s["tokens"] for s in stats_acc) or 1.0
        agg = aggregate_stats(stats_acc)
        agg.setdefault("ratio_max", 1.0)
        ev.interval("train", tr0, tr1, step=step, version=version, staleness=step - batch["version"],
                    behavior_logprobs=args.behavior_logprobs,
                    objective=args.objective,
                    reward_mean=float(rewards.mean()),
                    informative_groups=int((rewards.view(-1, G).std(1) > 0).sum()),
                    groups=int(rewards.numel() // G), train_seqs=len(keep),
                    gen_tokens=sum(len(s["completion_ids"]) for s in seqs),
                    train_tokens=float(tot) if stats_acc else 0.0, grad_norm=gns[-1] if gns else 0.0,
                    grad_norms=gns, **agg, **extra)
        ev.interval("publish", tr1, tp1, version=version)
    ev.point("done")


if __name__ == "__main__":
    main()
