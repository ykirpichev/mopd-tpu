"""CPU tests: loss math, log-prob alignment between sampler and learner, and the sync/async loop."""
import json
import os
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from qrl import data  # noqa: E402
from qrl.grpo import LossConfig, group_advantages, policy_loss  # noqa: E402
from qrl.learner import checkpoint_keys, export_name_map, make_microbatches, token_logprobs  # noqa: E402

BASE = os.environ.get("QRL_BASE_MODEL", "/tmp/models/qwen3.5-2b")  # config + tokenizer + index only


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    """Random 4-layer Qwen3.5 (3 Gated DeltaNet + 1 full attention) saved in the real checkpoint layout."""
    if not os.path.exists(BASE):
        pytest.skip("base model config not downloaded")
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration

    cfg = AutoConfig.from_pretrained(BASE)
    t = cfg.text_config
    t.num_hidden_layers = 4
    t.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    t.hidden_size, t.intermediate_size = 64, 128
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 32
    t.linear_num_key_heads, t.linear_num_value_heads = 2, 2
    t.linear_key_head_dim, t.linear_value_head_dim = 16, 16
    t.mtp_num_hidden_layers = 0
    v = cfg.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 1, 32, 64, 2, 64
    torch.manual_seed(0)
    m = Qwen3_5ForConditionalGeneration(cfg).to(torch.float32)
    out = tmp_path_factory.mktemp("tiny")
    m.save_pretrained(out)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(out)
    return str(out)


def test_reward_and_boxed():
    assert data.last_boxed(r"so \boxed{\frac{1}{2}} and \boxed{042}") == "042"
    assert data.reward(r"answer \boxed{42}", "42") == 1.0
    assert data.reward(r"answer \boxed{ 042 }", "42") == 1.0
    assert data.reward(r"answer \boxed{41}", "42") == 0.0
    assert data.reward("no box 42", "42") == 0.0


def test_batch_prompts_deterministic():
    rows = [{"id": i, "question": str(i), "answer": "1"} for i in range(10)]
    a = [r["id"] for r in data.batch_prompts(rows, 3, 4, seed=1)]
    assert a == [r["id"] for r in data.batch_prompts(rows, 3, 4, seed=1)]
    assert a != [r["id"] for r in data.batch_prompts(rows, 3, 4, seed=2)]


def test_group_advantages():
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    a = group_advantages(r)
    assert torch.allclose(a[0], torch.tensor([1.0, -1.0, -1.0, 1.0]), atol=1e-4)
    assert torch.all(a[1] == 0)


def test_tis_gradient_matches_weighted_pg():
    torch.manual_seed(0)
    lp = torch.randn(3, 5, requires_grad=True)
    behav = (lp.detach() + 0.3 * torch.randn(3, 5))
    adv = torch.tensor([1.0, -0.5, 2.0])
    mask = torch.ones(3, 5)
    mask[1, 3:] = 0
    cfg = LossConfig(kind="tis", is_cap=2.0)
    loss, st = policy_loss(lp, behav, adv, mask, cfg)
    loss.backward()
    w = torch.exp(lp.detach() - behav).clamp(max=2.0)
    expect = -(w * adv[:, None] * mask) / mask.sum()
    assert torch.allclose(lp.grad, expect, atol=1e-6)
    assert 0 <= st["truncated_frac"] <= 1


def test_on_policy_tis_equals_ppo():
    torch.manual_seed(1)
    lp = torch.randn(2, 4, requires_grad=True)
    lp2 = lp.detach().clone().requires_grad_(True)
    adv, mask = torch.tensor([1.0, -1.0]), torch.ones(2, 4)
    policy_loss(lp, lp.detach(), adv, mask, LossConfig("tis"))[0].backward()
    policy_loss(lp2, lp2.detach(), adv, mask, LossConfig("ppo"))[0].backward()
    assert torch.allclose(lp.grad, lp2.grad, atol=1e-6)


def test_export_names_cover_checkpoint(tiny_model):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(tiny_model)
    ck = checkpoint_keys(tiny_model)
    mp = export_name_map([n for n, _ in m.named_parameters()], ck)
    text_keys = {k for k in ck if k.startswith("model.language_model.")}
    assert set(mp.values()) == text_keys


def test_export_names_cover_real_checkpoint():
    """Every trained tensor of the real model maps to a checkpoint key vLLM's loader knows."""
    if not os.path.exists(os.path.join(BASE, "model.safetensors.index.json")):
        pytest.skip("base index not downloaded")
    from transformers import AutoConfig, AutoModelForCausalLM

    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(BASE))
    ck = checkpoint_keys(BASE)
    names = [n for n, _ in m.named_parameters()]
    mp = export_name_map(names, ck)
    text_keys = {k for k in ck if k.startswith("model.language_model.")}
    assert set(mp.values()) == text_keys
    n_params = sum(p.numel() for p in m.parameters())
    assert 1.5e9 < n_params < 2.5e9, n_params


def test_eval_harness_dry_run(tiny_model, tmp_path):
    out = tmp_path / "eval.json"
    cmd = [sys.executable, "-m", "qrl.evaluate", "--model", tiny_model, "--backend", "hf", "--samples", "1",
           "--max-tokens", "2", "--max-model-len", "512", "--ckpts", "base=base", "--out", str(out)]
    r = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(out.read_text())["base"]
    assert len(res["per_problem"]) == 60
    assert 0.0 <= res["pass@1"] <= 1.0


def test_sampler_and_learner_logprobs_align(tiny_model):
    """Learner log-probs on sampled tokens must equal the sampler's behavior log-probs (same weights)."""
    from qrl.rollout import HFBackend
    from transformers import AutoModelForCausalLM

    be = HFBackend(tiny_model, 256, 0)
    p = data.chat_prompt(be.tok, "What is 1+1?")
    seqs = be.generate([p], 3, 12, [7])
    m = AutoModelForCausalLM.from_pretrained(tiny_model).eval()
    adv = torch.ones(len(seqs))
    # Gated DeltaNet's chunked (training) and recurrent (decode) paths differ by ~1e-2 nats here,
    # so compare against the error a one-token misalignment would give.
    with torch.no_grad():
        for ids, tgt, behav, mask, a in make_microbatches(seqs, adv, 2, 16, 256, 0):
            lp = token_logprobs(m, ids, tgt, chunk=8)
            n = mask.sum()
            assert n > 0
            aligned = ((lp - behav).abs() * mask).sum() / n
            shifted = ((lp.roll(1, dims=1) - behav).abs() * mask).sum() / n
            assert aligned < 0.02, float(aligned)
            assert shifted > 5 * aligned, (float(aligned), float(shifted))


def _run(tiny_model, tmp_path, mode, k, steps=4):
    rows = [{"id": i, "question": f"What is {i}+{i}?", "answer": str(2 * i)} for i in range(8)]
    rows_f = tmp_path / "rows.json"
    rows_f.write_text(json.dumps(rows))
    run_dir = tmp_path / f"run-{mode}"
    cmd = [sys.executable, "-m", "qrl.run", "--run-dir", str(run_dir), "--model", tiny_model, "--mode", mode,
           "--staleness", str(k), "--steps", str(steps), "--rollout-backend", "hf", "--learner-device", "cpu",
           "--rollout-args", f"--prompts-per-batch 2 --group-size 2 --max-tokens 8 --max-model-len 256 "
                             f"--train-rows {rows_f} --test-random-reward",
           "--learner-args", "--dtype float32 --micro-batch 2 --len-quantum 16 --max-model-len 256 --lr 1e-3 "
                             f"--ckpt-dir {run_dir}/ckpt --ckpt-steps 0,{steps}"]
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr + (run_dir / "learner.log").read_text() + (run_dir / "rollout.log").read_text()
    ev = [json.loads(line) for f in ("rollout", "learner") for line in open(run_dir / "events" / f"{f}.jsonl")]
    return run_dir, ev


@pytest.mark.parametrize("mode,k", [("sync", 0), ("async", 2)])
def test_loop_end_to_end(tiny_model, tmp_path, mode, k):
    run_dir, ev = _run(tiny_model, tmp_path, mode, k)
    train = [e for e in ev if e["kind"] == "train"]
    gen = [e for e in ev if e["kind"] == "generate"]
    assert [e["step"] for e in train] == [0, 1, 2, 3]
    assert len(gen) == 4
    stal = [e["staleness"] for e in train]
    assert max(stal) <= k
    if mode == "sync":
        assert stal == [0, 0, 0, 0]
        # rollout b starts only after learner published version b
        pub = {e["version"]: e["end"] for e in ev if e["kind"] == "publish"}
        for g in gen[1:]:
            assert g["start"] >= pub[g["batch"]] - 1e-6
    assert any(e["train_seqs"] > 0 and e["grad_norm"] > 0 for e in train)
    assert os.path.exists(run_dir / "ckpt" / "step00000.safetensors")
    assert os.path.exists(run_dir / "ckpt" / "step00004.safetensors")
    loads = [e for e in ev if e["kind"] == "load_weights"]
    assert loads, "rollout engine never received new weights"
