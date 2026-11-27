"""CPU tests for the v2 plan: fixed-lag staleness, sharded rollouts, length schedule, difficulty filter,
token-budget microbatches, the profiling pilot and the bootstrap statistics."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from qrl import data, stats  # noqa: E402
from qrl.learner import _plan_microbatches  # noqa: E402
from qrl.rollout import parse_schedule, target_version  # noqa: E402
from tests.test_cpu import tiny_model  # noqa: E402,F401


def test_schedule_and_target_version():
    at = parse_schedule("30:8192,0:4096", 3072)
    assert [at(0), at(29), at(30), at(149)] == [4096, 4096, 8192, 8192]
    assert parse_schedule("", 3072)(5) == 3072
    assert [target_version(b, 2, "fixed", latest=b) for b in range(5)] == [0, 0, 0, 1, 2]
    assert target_version(5, 2, "bounded", latest=5) == 5
    assert target_version(5, 2, "bounded", latest=3) == 3


def test_filter_by_profile(tmp_path):
    rows = [{"id": i, "question": "q", "answer": "1"} for i in range(400)]
    prof = {str(i): {"correct": (0 if i < 200 else (8 if i < 300 else 3)), "n": 8} for i in range(400)}
    p = tmp_path / "profile.json"
    p.write_text(json.dumps({"prompts": prof}))
    kept = data.filter_by_profile(rows, str(p), keep_zero_frac=0.25)
    ids = {r["id"] for r in kept}
    assert set(range(300, 400)) <= ids                    # every 1..7/8 prompt kept
    assert not ids & set(range(200, 300))                 # always-solved prompts dropped
    zero = len(ids & set(range(200)))
    assert 30 <= zero <= 70                               # ~25% of never-solved prompts
    assert kept == data.filter_by_profile(rows, str(p), keep_zero_frac=0.25)  # deterministic


def test_token_budget_microbatches():
    lengths = [300, 310, 320, 330, 2000, 2100, 8000]
    plan = _plan_microbatches(lengths, mb_size=2, quantum=512, cap=8192, mb_tokens=4096)
    assert sorted(i for idx, _, _ in plan for i in idx) == list(range(7))
    for idx, L, rows in plan:
        assert len(idx) == rows and (rows == 1 or rows * L <= 4096)
    assert plan[0][2] == 4                                # four short sequences share one dispatch
    fixed = _plan_microbatches(lengths, mb_size=2, quantum=512, cap=8192)
    assert [r for _, _, r in fixed] == [2, 2, 2, 2]


def _loop(tiny_model, tmp_path, mode, k, steps, extra_run=(), extra_rollout="", extra_learner=""):  # noqa: F811
    rows = [{"id": i, "question": f"What is {i}+{i}?", "answer": str(2 * i)} for i in range(8)]
    rows_f = tmp_path / "rows.json"
    rows_f.write_text(json.dumps(rows))
    run_dir = tmp_path / f"run-{mode}-{k}"
    cmd = [sys.executable, "-m", "qrl.run", "--run-dir", str(run_dir), "--model", tiny_model, "--mode", mode,
           "--staleness", str(k), "--steps", str(steps), "--rollout-backend", "hf", "--learner-device", "cpu",
           *extra_run,
           "--rollout-args", f"--prompts-per-batch 2 --group-size 2 --max-tokens 8 --max-model-len 256 "
                             f"--train-rows {rows_f} --test-random-reward {extra_rollout}",
           "--learner-args", "--dtype float32 --micro-batch 2 --len-quantum 16 --max-model-len 256 --lr 1e-3 "
                             f"--gdn-chunk 128 --mb-tokens 512 {extra_learner}"]
    r = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True, text=True,
                       timeout=900)
    logs = "".join(p.read_text() for p in run_dir.glob("*.log"))
    assert r.returncode == 0, r.stdout + r.stderr + logs
    ev = []
    for f in (run_dir / "events").glob("*.jsonl"):
        ev += [json.loads(line) for line in open(f)]
    return run_dir, ev


def test_fixed_lag_is_exact(tiny_model, tmp_path):  # noqa: F811
    _, ev = _loop(tiny_model, tmp_path, "async", 2, 6, extra_run=("--staleness-mode", "fixed"))
    train = sorted((e for e in ev if e["kind"] == "train"), key=lambda e: e["step"])
    assert [e["staleness"] for e in train] == [0, 1, 2, 2, 2, 2]
    gen = sorted((e for e in ev if e["kind"] == "generate"), key=lambda e: e["batch"])
    assert [g["version"] for g in gen] == [0, 0, 0, 1, 2, 3]


def test_sharded_rollout_and_schedule(tiny_model, tmp_path):  # noqa: F811
    run_dir, ev = _loop(tiny_model, tmp_path, "sync", 0, 3, extra_run=("--rollout-chips", "0,1"),
                        extra_rollout="--max-tokens-schedule 0:4,2:8")
    gen = [e for e in ev if e["kind"] == "generate"]
    assert len(gen) == 6 and {g["max_tokens"] for g in gen if g["batch"] < 2} == {4}
    assert {g["max_tokens"] for g in gen if g["batch"] == 2} == {8}
    train = [e for e in ev if e["kind"] == "train"]
    assert [e["step"] for e in train] == [0, 1, 2] and all(e["groups"] == 2 for e in train)
    assert all(e["staleness"] == 0 for e in train)
    from qrl.analyze import summarize

    s = summarize(str(run_dir))
    assert s["rollout_engines"] == 2 and s["complete"]


def test_profile_prompts_and_merge(tiny_model, tmp_path):  # noqa: F811
    rows = [{"id": i, "question": f"What is {i}+{i}?", "answer": str(2 * i)} for i in range(4)]
    rows_f = tmp_path / "rows.json"
    rows_f.write_text(json.dumps(rows))
    outs = []
    for shard in range(2):
        out = tmp_path / f"p{shard}.json"
        cmd = [sys.executable, "-m", "qrl.profile_prompts", "--model", tiny_model, "--backend", "hf",
               "--samples", "2", "--max-tokens", "6", "--max-model-len", "256", "--train-rows", str(rows_f),
               "--shard-index", str(shard), "--num-shards", "2", "--out", str(out)]
        r = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True, text=True,
                           timeout=900)
        assert r.returncode == 0, r.stderr[-3000:]
        outs.append(str(out))
    merged = tmp_path / "profile.json"
    r = subprocess.run([sys.executable, "-m", "qrl.profile_prompts", "--merge", *outs, "--out", str(merged)],
                       cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    prof = json.loads(merged.read_text())
    assert set(prof["prompts"]) == {"0", "1", "2", "3"} and prof["summary"]["samples"] == 8
    assert data.filter_by_profile(rows, str(merged), 1.0)


def test_eval_shards_merge(tiny_model, tmp_path):  # noqa: F811
    outs = []
    for s in range(2):
        out = tmp_path / f"e{s}.json"
        cmd = [sys.executable, "-m", "qrl.evaluate", "--model", tiny_model, "--backend", "hf", "--samples", "1",
               "--max-tokens", "2", "--max-model-len", "512", "--ckpts", "base=base", "--shard-index", str(s),
               "--num-shards", "2", "--out", str(out)]
        r = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True, text=True,
                           timeout=900)
        assert r.returncode == 0, r.stderr[-3000:]
        outs.append(str(out))
    merged = tmp_path / "e.json"
    r = subprocess.run([sys.executable, "-m", "qrl.evaluate", "--merge", *outs, "--out", str(merged)], cwd=ROOT,
                       env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = json.loads(merged.read_text())["base"]
    assert res["problems"] == 60 and len({p["id"] for p in res["per_problem"]}) == 60


def test_repetition_score():
    from qrl.profile_prompts import repetition_score

    assert repetition_score(list(range(3000))) == 0.0
    assert repetition_score([1, 2, 3, 4, 5] * 600) > 0.9


def test_bootstrap_ci_covers_and_tightens():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 0.4, 136)
    a = rng.binomial(16, np.tile(p, (3, 1))).astype(float)
    b = rng.binomial(16, np.tile(p, (3, 1))).astype(float)
    d, draws = stats.bootstrap_diff(a, b, 16, iters=2000)
    lo, hi = stats.ci(draws)
    assert lo < 0 < hi and hi - lo < 0.06
    shifted = rng.binomial(16, np.tile(np.clip(p + 0.08, 0, 1), (3, 1))).astype(float)
    _, draws2 = stats.bootstrap_diff(shifted, b, 16, iters=2000)
    assert stats.ci(draws2)[0] > 0
    assert stats.equivalence(draws, margin=0.05)["equivalent"]
    assert not stats.equivalence(draws2, margin=0.05)["equivalent"]


def test_simulated_halfwidth_matches_first_run():
    """The simulator, fed the first run's design (60 problems x 8 samples x 2 seeds), should reproduce
    its observed hierarchical CI half-width of ~3.4 pp."""
    rng = np.random.default_rng(0)
    rates = np.concatenate([np.zeros(41), rng.uniform(0.05, 0.6, 19)])
    hw = stats.simulate_halfwidth(rates, 60, 8, 2, 0.23, iters=500, reps=20)
    assert 0.022 < hw < 0.05, hw


def test_seed_ratio_ci():
    r, (lo, hi) = stats.seed_ratio_ci([79.5, 79.4, 80.1], [127.2, 129.0, 128.4], iters=2000)
    assert 1.58 < r < 1.63 and lo < r < hi


def test_compare_on_first_run_results():
    R = os.path.join(ROOT, "results", "qwen35-2b-aime")
    if not os.path.exists(R):
        pytest.skip("first-run results not present")
    res = stats.compare({"async": [f"{R}/eval-async-s0.json::s0_40", f"{R}/eval-async-s1.json::s1_40"],
                         "sync": [f"{R}/eval-sync-s0.json::s0_40", f"{R}/eval-sync-s1.json::s1_40"]},
                        "s_40", n_samples=8, iters=3000)
    assert abs(res["diff"] - (-0.0042)) < 1e-3
    assert res["ci95"][0] < -0.025 and res["ci95"][1] > 0.02   # seeds widen it beyond the problem-only CI


@pytest.mark.skipif(os.environ.get("QRL_OFFLINE") == "1", reason="needs the Hugging Face Hub")
def test_eval_v2_set():
    rows = data.load_eval("v2")
    assert len(rows) == 136 and len({r["id"] for r in rows}) == 136
    assert all(r["answer"].lstrip("-").isdigit() for r in rows)
