"""CPU test for qrl.report_v2 on tiny runs laid out like the GCS results tree."""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tests.test_cpu import tiny_model  # noqa: E402,F401
from tests.test_v2 import _loop  # noqa: E402


def _fake_eval(path, label, n, rate, seed):
    import random

    rng = random.Random(seed)
    per = [{"id": f"2025-{i}", "correct": sum(rng.random() < rate for _ in range(n)), "truncated": 0, "lens": [10] * n}
           for i in range(30)]
    json.dump({label: {"pass@1": sum(p["correct"] for p in per) / (30 * n), "truncated_frac": 0.0, "samples": n,
                       "per_problem": per, "eval_s": 1.0}}, open(path, "w"))


def test_report_v2(tiny_model, tmp_path):  # noqa: F811
    root = tmp_path / "v2"
    for sub in "abc":
        (tmp_path / sub).mkdir()
    src_sync, _ = _loop(tiny_model, tmp_path / "a", "sync", 0, 4)
    src_async, _ = _loop(tiny_model, tmp_path / "b", "async", 2, 4, extra_learner="--log-entropy")
    src_none, _ = _loop(tiny_model, tmp_path / "c", "async", 4, 4, extra_run=("--staleness-mode", "fixed"),
                        extra_learner="--behavior-logprobs none --log-entropy")
    layout = {"vm1": [("sync-s0", src_sync), ("async-s0", src_async)],
              "vm2": [("sync-s1", src_sync), ("nois-s0", src_none)],
              "vm3": [("async-s1", src_async), ("nois-s1", src_none)]}
    for vm, runs in layout.items():
        for i, (name, src) in enumerate(runs):
            d = root / vm / name
            shutil.copytree(src, d, ignore=shutil.ignore_patterns("weights*", "*.safetensors", "channel*"))
            for lab, n in (("step2", 16), ("step4", 32)):
                _fake_eval(d / f"eval.{lab}.json", lab, n, 0.2 + 0.05 * i, hash((vm, name, lab)) % 1000)
    os.makedirs(root / "vm1" / "pilot")
    _fake_eval(root / "vm1" / "pilot" / "eval_base.json", "base", 32, 0.15, 7)

    from qrl.report_v2 import build, charts

    rep = build(str(root))
    r1, r2 = rep["runs"]["run1"], rep["runs"]["run2"]
    assert set(r1["arms"]) == {"sync", "async"} and set(r2["arms"]) == {"async", "nois"}
    assert r1["arms"]["sync"]["evals"]["step4"]["n_seeds"] == 2
    assert "gain_ci95" in r1["arms"]["sync"]["evals"]["step2"]
    assert set(r2["compare"]) == {"step2", "step4"} and "speedup_b_over_a" in r1
    assert r2["arms"]["nois"]["staleness_hist"]
    paths = charts(rep, str(tmp_path / "media"))
    assert len(paths) == 4 and all(os.path.getsize(p) > 10_000 for p in paths)
