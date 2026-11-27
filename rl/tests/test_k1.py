"""CPU tests for the plan25k1 arms: async, fixed lag k=1, with truncated IS and without IS."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tests.test_cpu import tiny_model  # noqa: E402,F401
from tests.test_v2 import _loop  # noqa: E402


@pytest.mark.parametrize("behavior", ["use", "none"])
def test_k1_fixed_lag_is_exactly_one(tiny_model, tmp_path, behavior):  # noqa: F811
    _, ev = _loop(tiny_model, tmp_path, "async", 1, 5, extra_run=("--staleness-mode", "fixed"),
                  extra_learner=f"--behavior-logprobs {behavior} --loss tis")
    train = sorted((e for e in ev if e["kind"] == "train"), key=lambda e: e["step"])
    assert [e["staleness"] for e in train] == [0, 1, 1, 1, 1]
    assert all(e["behavior_logprobs"] == behavior for e in train)
    gen = sorted((e for e in ev if e["kind"] == "generate"), key=lambda e: e["batch"])
    assert [e["version"] for e in gen] == [0, 0, 1, 2, 3]
