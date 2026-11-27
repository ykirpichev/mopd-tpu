"""CPU tests for the no-IS ablation (--behavior-logprobs none)."""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from qrl.grpo import LossConfig, behavior_diagnostics, policy_loss, recomputed_old_loss  # noqa: E402
from qrl.learner import aggregate_stats  # noqa: E402
from tests.test_cpu import tiny_model  # noqa: E402,F401
from tests.test_v2 import _loop  # noqa: E402


def _case(seed=0):
    g = torch.Generator().manual_seed(seed)
    lp = torch.randn(3, 6, generator=g) - 2.0
    behav = lp + 0.5 * torch.randn(3, 6, generator=g)
    adv = torch.tensor([1.0, -0.5, 2.0])
    mask = torch.ones(3, 6)
    mask[1, 4:] = 0
    return lp, behav, adv, mask


def test_single_update_is_plain_pg_and_ignores_sampler():
    lp, behav, adv, mask = _case()
    cfg = LossConfig()
    x = lp.clone().requires_grad_(True)
    loss, st = recomputed_old_loss(x, x.detach(), adv, mask, cfg)
    loss.backward()
    expect = -(adv[:, None] * mask) / mask.sum()
    assert torch.allclose(x.grad, expect, atol=1e-6)
    assert st["old_ratio_max"] == 1.0 and st["clip_frac"] == 0.0
    # The IS baseline's gradient does depend on the sampler's log-probs.
    y = lp.clone().requires_grad_(True)
    policy_loss(y, behav, adv, mask, cfg)[0].backward()
    assert not torch.allclose(y.grad, expect, atol=1e-4)


def test_clip_binds_on_later_minibatches():
    lp, _, adv, mask = _case()
    cfg = LossConfig(clip_low=0.2, clip_high=0.28)
    old = lp - 0.5                        # policy moved up by 0.5 nats since the recompute: ratio ~1.65
    x = lp.clone().requires_grad_(True)
    loss, st = recomputed_old_loss(x, old, adv, mask, cfg)
    loss.backward()
    assert st["clip_frac"] > 0.6
    pos = adv > 0
    assert torch.all(x.grad[pos] == 0)    # clipped for positive advantages
    assert torch.all(x.grad[~pos][mask[~pos] > 0] != 0)  # negative advantage keeps the unclipped branch


def test_behavior_diagnostics():
    lp, behav, _, mask = _case()
    d = behavior_diagnostics(lp, behav, mask, is_cap=2.0)
    r = torch.exp(lp - behav)
    m = mask > 0
    assert abs(d["behav_ratio_mean"] - float(r[m].mean())) < 1e-5
    assert abs(d["behav_ratio_max"] - float(r[m].max())) < 1e-5
    assert abs(d["behav_would_truncate_frac"] - float((r[m] > 2).float().mean())) < 1e-6
    assert d["kl_behav_cur"] >= 0
    same = behavior_diagnostics(lp, lp, mask)
    assert same["kl_behav_cur"] == 0.0 and same["behav_ratio_max"] == 1.0


def test_aggregate_stats():
    a = aggregate_stats([{"tokens": 1.0, "x": 1.0, "y_max": 3.0, "z_min": 0.5},
                         {"tokens": 3.0, "x": 5.0, "y_max": 2.0, "z_min": 0.1}])
    assert a == {"x": 4.0, "y_max": 3.0, "z_min": 0.1}


def test_no_is_end_to_end(tiny_model, tmp_path):  # noqa: F811
    run_dir, ev = _loop(tiny_model, tmp_path, "async", 4, 6, extra_run=("--staleness-mode", "fixed"),
                        extra_learner="--behavior-logprobs none --updates-per-batch 2 --log-entropy")
    train = sorted((e for e in ev if e["kind"] == "train"), key=lambda e: e["step"])
    assert [e["staleness"] for e in train] == [0, 1, 2, 3, 4, 4]
    assert all(e["behavior_logprobs"] == "none" for e in train)
    trained = [e for e in train if e["train_seqs"] >= 2]
    assert trained
    for e in trained:
        assert len(e["grad_norms"]) == 2
        for k in ("clip_frac", "old_ratio_max", "behav_ratio_mean", "behav_would_truncate_frac", "kl_behav_cur",
                  "entropy"):
            assert k in e, k
        assert e["entropy"] > 0
    from qrl.analyze import summarize

    s = summarize(str(run_dir))
    assert s["behavior_logprobs"] == "none" and "entropy_mean" in s and len(s["kl_curve"]) == 6


def test_no_is_single_update_ratio_is_one(tiny_model, tmp_path):  # noqa: F811
    _, ev = _loop(tiny_model, tmp_path, "async", 2, 3, extra_learner="--behavior-logprobs none")
    for e in (e for e in ev if e["kind"] == "train" and e["train_seqs"] > 0):
        assert e["old_ratio_max"] == 1.0 and e["clip_frac"] == 0.0
        assert json.dumps(e)  # serializable
