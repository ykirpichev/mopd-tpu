"""Summarize runs from their event logs: wall-clock, idle fractions, throughput, staleness, rewards."""
import argparse
import json
import os


def load_events(run_dir):
    """All events; sharded rollout engines (rollout.s<i>.jsonl) get proc "rollout" and a "shard" field."""
    import glob

    ev = []
    for p in sorted(glob.glob(os.path.join(run_dir, "events", "*.jsonl"))):
        name = os.path.basename(p)[:-len(".jsonl")]
        shard = int(name.split(".s")[1]) if name.startswith("rollout.s") else None
        for line in open(p):
            e = json.loads(line)
            if shard is not None:
                e["shard"] = shard
            ev.append(e)
    return ev


def summarize(run_dir):
    ev = load_events(run_dir)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    gen = sorted((e for e in ev if e["kind"] == "generate"), key=lambda e: (e["batch"], e.get("shard", 0)))
    n_shards = len({e.get("shard", 0) for e in gen}) or 1
    train = sorted((e for e in ev if e["kind"] == "train"), key=lambda e: e["step"])
    pub = [e for e in ev if e["kind"] == "publish"]
    loads = [e for e in ev if e["kind"] == "load_weights"]
    if not gen or not train:
        return {"run_dir": run_dir, "complete": False}
    # Steady-state window: first generation start -> last publish end. Startup/compile is reported separately.
    t_start = gen[0]["start"]
    t_end = max(e["end"] for e in pub)
    wall = t_end - t_start
    # Per-engine busy time, averaged over engines when the rollout is sharded across chips.
    gen_busy = (sum(e["end"] - e["start"] for e in gen) + sum(e["end"] - e["start"] for e in loads)) / n_shards
    learn_busy = sum(e["end"] - e["start"] for e in train) + sum(e["end"] - e["start"] for e in pub)
    gen_tokens = sum(e["gen_tokens"] for e in gen)
    startup = {e["proc"]: round(e["end"] - e["start"], 1) for e in ev if e["kind"] == "startup"}
    return {
        "run_dir": run_dir, "complete": len(train) == cfg["steps"], "mode": cfg["mode"], "k": cfg["k"],
        "seed": cfg["seed"], "steps": len(train), "wall_s": wall, "s_per_step": wall / len(train),
        "rollout_idle_frac": 1 - gen_busy / wall, "learner_idle_frac": 1 - learn_busy / wall,
        "gen_tokens": gen_tokens, "gen_tok_per_s": gen_tokens / wall,
        "train_tokens": sum(e["train_tokens"] for e in train),
        "gen_s_mean": sum(e["end"] - e["start"] for e in gen) / len(gen), "rollout_engines": n_shards,
        "truncated_frac_gen": sum(e.get("truncated_frac", 0.0) for e in gen) / len(gen),
        "train_s_mean": sum(e["end"] - e["start"] for e in train) / len(train),
        "publish_s_mean": sum(e["end"] - e["start"] for e in pub) / len(pub),
        "load_weights_s_mean": (sum(e["end"] - e["start"] for e in loads) / len(loads)) if loads else 0.0,
        "informative_groups_mean": sum(e.get("informative_groups", 0) for e in train) / len(train),
        "behavior_logprobs": train[0].get("behavior_logprobs", "use"),
        "grad_norm_mean": sum(e.get("grad_norm", 0.0) for e in train) / len(train),
        "grad_norm_max": max(e.get("grad_norm", 0.0) for e in train),
        **{f"{k}_mean": sum(e[k] for e in train if k in e) / max(1, sum(k in e for e in train))
           for k in ("entropy", "behav_would_truncate_frac", "behav_ratio_mean", "clip_frac")
           if any(k in e for e in train)},
        "behav_ratio_max": max((e.get("behav_ratio_max", e.get("ratio_max", 1.0)) for e in train), default=1.0),
        "kl_curve": [e.get("kl_behav_cur", 0.0) for e in train],
        "entropy_curve": [e["entropy"] for e in train if "entropy" in e],
        "staleness_mean": sum(e["staleness"] for e in train) / len(train),
        "staleness_max": max(e["staleness"] for e in train),
        "staleness_hist": {str(s): sum(1 for e in train if e["staleness"] == s) for s in sorted({e["staleness"] for e in train})},
        "reward_curve": [e["reward_mean"] for e in train],
        "kl_behav_cur_mean": sum(e.get("kl_behav_cur", 0.0) for e in train) / len(train),
        "truncated_frac_mean": sum(e.get("truncated_frac", 0.0) for e in train) / len(train),
        "startup_s": startup,
        "step_end_times": [e["end"] - t_start for e in sorted(pub, key=lambda e: e["version"])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = [summarize(d) for d in args.run_dirs]
    for r in res:
        brief = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                 if k not in ("reward_curve", "step_end_times")}
        print(json.dumps(brief))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
