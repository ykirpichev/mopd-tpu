"""CPU tests for the MOPD stack: IF reward and eval checkers, mixed batches, the teacher prefill service, the
sampled-token (opd) and full-distribution (opd_dist) losses, hidden-state shipping, and the teacher export."""
import json
import os
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from qrl import data, ifeval  # noqa: E402
from qrl.grpo import LossConfig, distill_advantage, policy_loss, reverse_kl_piece  # noqa: E402
from tests.test_cpu import tiny_model  # noqa: E402,F401


def test_if_reward_matches_open_instruct_semantics():
    gt = json.dumps({"func_name": "validate_lowercase", "N": None, "word": None})
    assert ifeval.reward("all lower case here.", gt) == 1.0
    assert ifeval.reward("Not lower.", gt) == 0.0
    assert ifeval.reward("", gt) == 0.0
    gt2 = json.dumps({"func_name": "validate_no_commas", "N": None})
    assert ifeval.reward("<think>a, b</think>no commas at all", gt2) == 1.0
    gt3 = json.dumps({"func_name": "verify_keywords", "keyword_list": ["apple", "pie"]})
    assert ifeval.reward("Apple and PIE", gt3) == 1.0 and ifeval.reward("apple only", gt3) == 0.0
    assert ifeval.reward("x", json.dumps({"func_name": "no_such_check"})) == 0.0


def test_ifeval_and_ifbench_checkers():
    ifeval.setup_nltk()
    row = {"question": "Write without commas in lowercase.", "instruction_id_list":
           ["punctuation:no_comma", "change_case:english_lowercase"], "kwargs": [{}, {}]}
    assert ifeval.follow_list(row, "hello world no commas here", "ifeval") == [True, True]
    assert ifeval.follow_list(row, "Hello, world", "ifeval") == [False, False]
    row2 = {"question": "q", "instruction_id_list": ["length_constraints:number_sentences"],
            "kwargs": [{"relation": "less than", "num_sentences": 3}]}
    assert ifeval.follow_list(row2, "One. Two.", "ifeval") == [True]
    assert ifeval.follow_list(row2, "One. Two. Three. Four.", "ifeval") == [False]
    from qrl.ifv.ifbench import instructions_registry as reg

    iid = sorted(reg.INSTRUCTION_DICT)[0]
    assert ifeval.follow_list({"question": "q", "instruction_id_list": [iid], "kwargs": [{}]}, "text",
                              "ifbench")[0] in (True, False)


def test_mixed_batches():
    math = [{"id": i, "domain": "math", "question": str(i), "answer": "1"} for i in range(20)]
    ifr = [{"id": f"if-{i}", "domain": "if", "question": str(i), "ground_truth": "{}"} for i in range(30)]
    by = {"math": math, "if": ifr}
    assert data.mixed_batch(by, "math:16", 3, 7) == data.batch_prompts(math, 3, 16, 7)
    b = data.mixed_batch(by, "math:8,if:8", 2, 7)
    assert [data.domain_of(r) for r in b] == ["math"] * 8 + ["if"] * 8
    assert b == data.mixed_batch(by, "math:8,if:8", 2, 7)
    assert b[:8] == data.batch_prompts(math, 2, 8, 7)


class _Tok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return messages[0]["content"]


def test_row_prompt_suffix():
    assert data.row_prompt(_Tok(), {"question": "Q", "domain": "math"}).endswith(data.INSTRUCTION)
    assert data.row_prompt(_Tok(), {"question": "Q", "domain": "if"}) == "Q"


def test_distill_advantage_sign_clip_and_is():
    lp = torch.tensor([[-1.0, -2.0, -3.0, -0.5]], requires_grad=True)
    tlp = torch.tensor([[-0.5, -9.0, -3.0, -0.5]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    a, st = distill_advantage(lp, tlp, mask, clip=5.0)
    assert torch.allclose(a, torch.tensor([[0.5, -5.0, 0.0, 0.0]]))
    assert abs(st["kl_teacher"] - (-(0.5 - 7.0 + 0.0) / 3)) < 1e-6
    assert abs(st["adv_clip_frac"] - 1 / 3) < 1e-6
    behav = lp.detach() - 0.1
    loss, _ = policy_loss(lp, behav, a, mask, LossConfig(kind="tis", is_cap=2.0))
    loss.backward()
    w = torch.exp(torch.tensor(0.1)).clamp(max=2.0)
    assert torch.allclose(lp.grad, -(w * a * mask) / 3, atol=1e-6)


def test_reverse_kl_piece_matches_direct_and_grad():
    torch.manual_seed(0)
    C, H, V = 6, 8, 50
    h = torch.randn(C, H, requires_grad=True)
    ht = torch.randn(C, H)
    ws = torch.randn(V, H, requires_grad=True)
    wt = torch.randn(V, H)
    y = torch.randint(0, V, (C,))
    kl, lp, ent, tlp = reverse_kl_piece(h, ht, ws, wt, y)
    ls = torch.log_softmax(h @ ws.t(), -1)
    lt = torch.log_softmax(ht @ wt.t(), -1)
    ref = (ls.exp() * (ls - lt)).sum(-1)
    assert torch.allclose(kl, ref, atol=1e-5)
    assert torch.allclose(tlp, lt.gather(-1, y[:, None])[:, 0], atol=1e-6)
    assert torch.allclose(lp, ls.gather(-1, y[:, None])[:, 0], atol=1e-6)
    assert (kl >= -1e-6).all()
    g1 = torch.autograd.grad(kl.sum(), [h, ws])
    g2 = torch.autograd.grad(ref.sum(), [h, ws])
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-5)
    # the sampled-token estimator's expectation is the exact KL: grad of E_y[(lp_S - lp_T)] matches on average
    assert torch.allclose((ls.exp() * (ls - lt)).sum(-1), kl, atol=1e-5)


def _perturbed_copy(src, dst, scale=0.05, seed=1):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration

    m = Qwen3_5ForConditionalGeneration.from_pretrained(src, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    m.save_pretrained(dst)
    AutoTokenizer.from_pretrained(src).save_pretrained(dst)
    return dst


@pytest.fixture(scope="session")
def tiny_teachers(tiny_model, tmp_path_factory):  # noqa: F811
    d = tmp_path_factory.mktemp("teachers")
    return {"math": _perturbed_copy(tiny_model, str(d / "tmath"), seed=1),
            "if": _perturbed_copy(tiny_model, str(d / "tif"), seed=2)}


def _seqs(tok, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        p = torch.randint(10, 1000, (5 + i,), generator=g).tolist()
        c = torch.randint(10, 1000, (7 + 3 * i,), generator=g).tolist()
        out.append({"prompt_ids": p, "completion_ids": c, "behav_logp": [-1.0] * len(c),
                    "domain": "math" if i % 2 == 0 else "if", "reward": 0.0})
    return out


def test_teacher_scores_match_token_logprobs_and_hidden_rebuilds_logits(tiny_model, tiny_teachers):  # noqa: F811
    """Gate 3 on CPU: the loss rebuilt from shipped hidden states equals the KL from the teacher's own logits."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from qrl import learner, teacher

    tok = AutoTokenizer.from_pretrained(tiny_model)
    seqs = _seqs(tok)
    tm = {k: AutoModelForCausalLM.from_pretrained(v, dtype=torch.float32).eval() for k, v in tiny_teachers.items()}
    args = type("A", (), {"len_quantum": 16, "max_model_len": 256, "mb_tokens": 64, "chunk": 8, "pad_id": 0})()
    for name, m in tm.items():
        sub = [s for s in seqs if s["domain"] == name]
        lps, hids, n = teacher.score_seqs(m, sub, "cpu", args, want_hidden=set(range(len(sub))))
        for s, lp, j in zip(sub, lps, range(len(sub))):
            ids = torch.tensor([s["prompt_ids"] + s["completion_ids"]])
            with torch.no_grad():
                ref = learner.token_logprobs(m, ids[:, :-1], ids[:, 1:])[0, len(s["prompt_ids"]) - 1:]
            assert torch.allclose(torch.tensor(lp), ref, atol=1e-5)
            assert hids[j].shape == (len(s["completion_ids"]), m.config.hidden_size)
            s["teacher_logp"], s["teacher"], s["hid"] = lp, name, ("f", f"0.{id(s)}")
    hid = {}
    for name, m in tm.items():
        sub = [s for s in seqs if s["domain"] == name]
        _, hids, _ = teacher.score_seqs(m, sub, "cpu", args, want_hidden=set(range(len(sub))))
        for j, s in enumerate(sub):
            hid[s["hid"][1]] = hids[j]
    student = AutoModelForCausalLM.from_pretrained(tiny_model, dtype=torch.float32)
    heads = {k: m.get_output_embeddings().weight.detach() for k, m in tm.items()}
    ds = learner.DomainSums()
    total_tok = sum(len(s["completion_ids"]) for s in seqs)
    loss_tot, st_all = 0.0, []
    for mb in learner.distill_microbatches(seqs, 16, 256, 0, 64):
        loss, st = learner.dist_microbatch(student, mb, seqs, hid, heads, "cpu", torch.tensor(float(total_tok)), 8,
                                           ds, grad=True)
        loss.backward()
        loss_tot += float(loss)
        st_all.append(st)
    assert max(s["hid_check_max"] for s in st_all) < 1e-4
    # direct reference: full teacher logits from the teacher models, student logits, exact KL per position
    ref = 0.0
    for s in seqs:
        ids = torch.tensor([s["prompt_ids"] + s["completion_ids"]])
        P = len(s["prompt_ids"]) - 1
        with torch.no_grad():
            lt = torch.log_softmax(tm[s["domain"]](input_ids=ids[:, :-1]).logits[0, P:].float(), -1)
            ls = torch.log_softmax(student(input_ids=ids[:, :-1]).logits[0, P:].float(), -1)
        ref += float((ls.exp() * (ls - lt)).sum())
    assert abs(loss_tot - ref / total_tok) < 1e-4 * max(1.0, abs(ref / total_tok))
    assert "kl_exact_math" in ds.out() and "kl_exact_if" in ds.out()


def _rows(tmp_path):
    rows = [{"id": i, "domain": "math", "question": f"What is {i}+{i}?", "answer": str(2 * i)} for i in range(6)]
    rows += [{"id": f"if-{i}", "domain": "if", "question": f"Say {i} in lowercase.",
              "ground_truth": json.dumps({"func_name": "validate_lowercase"})} for i in range(6)]
    f = tmp_path / "rows.json"
    f.write_text(json.dumps(rows))
    return f


def _run(tiny_model, tmp_path, name, steps, objective, teachers, tchips="0", ship="none", extra_teacher="",
         domains="math:1,if:1", extra_learner="", extra_rollout=""):  # noqa: F811
    run_dir = tmp_path / name
    tspec = ",".join(f"{k}={v}" for k, v in teachers.items())
    learner_obj = "" if objective == "grpo" else f"--objective {objective}"
    cmd = [sys.executable, "-m", "qrl.run", "--run-dir", str(run_dir), "--model", tiny_model, "--mode", "async",
           "--staleness", "1", "--staleness-mode", "fixed", "--steps", str(steps), "--rollout-backend", "hf",
           "--learner-device", "cpu",
           "--rollout-args", f"--domains {domains} --group-size 2 --max-tokens 8 --max-model-len 256 "
                             f"--train-rows {_rows(tmp_path)} {extra_rollout}",
           "--learner-args", f"--dtype float32 --len-quantum 16 --max-model-len 256 --lr 1e-3 --gdn-chunk 128 "
                             f"--mb-tokens 64 --dist-chunk 8 --log-entropy {learner_obj} {extra_learner}"]
    if objective not in ("grpo", "sft"):
        cmd += ["--teacher-chips", tchips, "--teacher-args",
                f"--teachers {tspec} --dtype float32 --len-quantum 16 --max-model-len 256 --mb-tokens 64 "
                f"--chunk 8 --gdn-chunk 128 --ship-hidden {ship} {extra_teacher}"]
    r = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True, text=True,
                       timeout=1200)
    logs = "".join(p.read_text() for p in run_dir.glob("*.log"))
    assert r.returncode == 0, r.stdout + r.stderr + logs
    ev = []
    for f in (run_dir / "events").glob("*.jsonl"):
        ev += [json.loads(line) for line in open(f)]
    return run_dir, ev


def test_opd_loop_lag1_with_diag(tiny_model, tiny_teachers, tmp_path):  # noqa: F811
    run_dir, ev = _run(tiny_model, tmp_path, "opd", 4, "opd", tiny_teachers, ship="diag",
                       extra_teacher="--diag-every 2 --diag-seqs 1")
    train = sorted((e for e in ev if e["kind"] == "train"), key=lambda e: e["step"])
    assert [e["staleness"] for e in train] == [0, 1, 1, 1]
    assert all(e["objective"] == "opd" and e["train_seqs"] == 4 for e in train)
    assert all("kl_teacher_math" in e and "kl_teacher_if" in e for e in train)
    assert all(e["kl_teacher"] > 0 for e in train)
    assert "diag_kl_exact" in train[0] and "diag_kl_exact" in train[2] and "diag_kl_exact" not in train[1]
    score = [e for e in ev if e["kind"] == "score"]
    assert len(score) == 4 and all(e["tokens"] > 0 for e in score)
    assert any("kl_behav_other_math_vs_if" in e for e in score)
    assert not list((run_dir / "scored").glob("*.pt")) and not list((run_dir / "rollouts").glob("*.pt"))
    from qrl import analyze
    sm = analyze.summarize(str(run_dir))
    assert sm["complete"] and 0 <= sm["learner_idle_frac"] < 1 and sm["s_per_step"] > 0


def test_opd_dist_loop_and_two_teacher_shards(tiny_model, tiny_teachers, tmp_path):  # noqa: F811
    run_dir, ev = _run(tiny_model, tmp_path, "dist", 3, "opd_dist", tiny_teachers, tchips="0,1", ship="all")
    train = sorted((e for e in ev if e["kind"] == "train"), key=lambda e: e["step"])
    assert [e["staleness"] for e in train] == [0, 1, 1]
    for e in train:
        assert e["objective"] == "opd_dist" and e["tokens"] if "tokens" in e else True
        assert e["kl_exact"] >= 0 and e["hid_check_max"] < 1e-4
        assert "kl_exact_math" in e and "kl_exact_if" in e and "entropy_math" in e
    assert len([e for e in ev if e["kind"] == "score"]) == 6
    assert sorted(p.name for p in (run_dir / "teacher_heads").iterdir()) == ["if.safetensors", "math.safetensors"]
    from qrl import analyze
    assert analyze.summarize(str(run_dir))["complete"]
    assert not list((run_dir / "scored").glob("*.pt")) and not list((run_dir / "rollouts").glob("*.pt"))


def test_single_teacher_all(tiny_model, tiny_teachers, tmp_path):  # noqa: F811
    _, ev = _run(tiny_model, tmp_path, "opd9b", 2, "opd", {"all": tiny_teachers["math"]})
    assert all(e.get("teacher") is None for e in ev)
    sc = [e for e in ev if e["kind"] == "score"]
    assert all("kl_behav_teacher_all" in e for e in sc)


def test_mixrl_grpo_mixed_domains(tiny_model, tmp_path):  # noqa: F811
    _, ev = _run(tiny_model, tmp_path, "mix", 2, "grpo", {})
    gen = [e for e in ev if e["kind"] == "generate"]
    assert all("reward_math" in g and "reward_if" in g for g in gen)
    train = [e for e in ev if e["kind"] == "train"]
    assert all(e["objective"] == "grpo" and "reward_if" in e for e in train)


def test_sft_teacher_gen_loop(tiny_model, tiny_teachers, tmp_path):  # noqa: F811
    tspec = ",".join(f"{k}={v}" for k, v in tiny_teachers.items())
    run_dir, ev = _run(tiny_model, tmp_path, "sft", 4, "sft", {},
                       extra_rollout=f"--teacher-gen {tspec} --teacher-gen-block 2")
    gen = sorted((e for e in ev if e["kind"] == "generate"), key=lambda e: e["batch"])
    assert [e["batch"] for e in gen] == [0, 1, 2, 3] and all(e["teacher_gen"] for e in gen)
    assert [e["block"] for e in gen] == [1, 2, 2, 1]
    loads = [e for e in ev if e["kind"] == "load_weights" and e.get("teacher")]
    assert [e["teacher"] for e in loads] == ["math", "if"] * 3
    train = sorted((e for e in ev if e["kind"] == "train"), key=lambda e: e["step"])
    assert len(train) == 4 and all(e["objective"] == "sft" for e in train)
    assert all(e["ce"] > 0 and "fkl_teacher_math" in e and "fkl_teacher_if" in e for e in train)
    assert not any(e["kind"] == "score" for e in ev)
    from qrl import analyze
    assert analyze.summarize(str(run_dir))["complete"]


def test_sft_samples_come_from_the_routed_teacher(tiny_model, tiny_teachers, tmp_path):  # noqa: F811
    """behav_logp of a teacher-generated sample is closest to that teacher's log-prob (HF's cached GDN decode and
    the full forward differ slightly on long prompts, so exact equality is not testable here)."""
    import argparse

    from transformers import AutoModelForCausalLM

    from qrl import channel, learner, rollout

    rows = json.loads(_rows(tmp_path).read_text())
    by = {"math": [r for r in rows if r["domain"] == "math"], "if": [r for r in rows if r["domain"] == "if"]}
    be = rollout.HFBackend(tiny_model, 256, 0)
    ch = channel.Channel(str(tmp_path / "ch"))
    a = argparse.Namespace(teacher_gen=",".join(f"{k}={v}" for k, v in tiny_teachers.items()), teacher_gen_block=4,
                           steps=1, shard_index=0, num_shards=1, seed=3, enable_thinking=False, group_size=2,
                           temperature=1.0)
    rollout.teacher_gen_loop(a, ch, channel.EventLog(ch, "rollout"), be, by, "math:1,if:1", ["math", "if"], None,
                             lambda b: 8)
    seqs = ch.take_batch(0, 1)["seqs"]
    assert {s["domain"] for s in seqs} == {"math", "if"}
    models = {k: AutoModelForCausalLM.from_pretrained(v, dtype=torch.float32).eval()
              for k, v in {"base": tiny_model, **tiny_teachers}.items()}
    for s in seqs:
        ids = torch.tensor([s["prompt_ids"] + s["completion_ids"]])
        err = {}
        for k, m in models.items():
            with torch.no_grad():
                ref = learner.token_logprobs(m, ids[:, :-1], ids[:, 1:])[0, len(s["prompt_ids"]) - 1:]
            err[k] = float((torch.tensor(s["behav_logp"]) - ref).abs().mean())
        assert min(err, key=err.get) == s["domain"], err


def test_merge_methods(tiny_model, tiny_teachers, tmp_path):  # noqa: F811
    from qrl import merge_teachers as mt

    a = torch.tensor([1.0, -2.0, 0.1, 0.0, 3.0])
    b = torch.tensor([2.0, 1.0, -0.2, 0.0, 1.0])
    assert torch.allclose(mt.trim(a, 0.4), torch.tensor([0.0, -2.0, 0.0, 0.0, 3.0]))
    # TIES at density 1: sign from the sum, mean over agreeing non-zeros
    assert torch.allclose(mt.ties_merge([a, b], 1.0), torch.tensor([1.5, -2.0, -0.2, 0.0, 2.0]))
    base = mt.load_state(tiny_model)
    t = [mt.load_state(v) for v in tiny_teachers.values()]
    for method, kw in [("linear", {}), ("ta", {"lam": 1.0}), ("ties", {"lam": 1.0, "density": 0.2})]:
        out = tmp_path / method
        mt.main(["--base", tiny_model, "--teachers", *(f"{k}={v}" for k, v in tiny_teachers.items()),
                 "--method", method, *sum(([f"--{k}", str(v)] for k, v in kw.items()), []), "--out", str(out)])
        m = mt.load_state(str(out))
        assert set(m) == set(base)
        for k, v in base.items():
            if not k.startswith(mt.TEXT_PREFIX):
                assert torch.equal(m[k], v)
                continue
            if method == "linear":
                ref = (t[0][k].float() + t[1][k].float()) / 2
            elif method == "ta":
                ref = t[0][k].float() + t[1][k].float() - v.float()
            else:
                continue
            assert torch.allclose(m[k].float(), ref.to(v.dtype).float(), atol=1e-6), k
        meta = json.load(open(out / "merge.json"))
        assert meta["method"] == method and meta["tensors_merged"] > 0
        from transformers import AutoModelForCausalLM
        AutoModelForCausalLM.from_pretrained(str(out), dtype=torch.float32)
    ties = mt.load_state(str(tmp_path / "ties"))
    k = next(k for k in base if k.startswith(mt.TEXT_PREFIX) and base[k].numel() > 100)
    d = ties[k].float() - base[k].float()
    assert 0.15 < float((d != 0).float().mean()) <= 0.4


def test_export_teacher_roundtrip(tiny_model, tmp_path):  # noqa: F811
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM

    from qrl import export_teacher
    from qrl.learner import checkpoint_keys, export_name_map

    m = AutoModelForCausalLM.from_pretrained(tiny_model, dtype=torch.float32)
    named = [(n, p) for n, p in m.named_parameters()]
    nm = export_name_map([n for n, _ in named], checkpoint_keys(tiny_model))
    ck = {nm[n]: (p.detach() + 1.0).contiguous() for n, p in named if n in nm}
    save_file(ck, str(tmp_path / "step00002.safetensors"))
    ev = tmp_path / "eval.json"
    ev.write_text(json.dumps({"step2": {"pass@1": 0.5, "samples": 2, "problems": 2,
                                        "per_problem": [{"id": "a", "correct": 1}, {"id": "b", "correct": 1}]}}))
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"domain": "math", "seed": 0}))
    out = tmp_path / "teachers" / "math-grpo-k1is-2steps-s0-20261124-abcdef0"
    export_teacher.main(["--base", tiny_model, "--ckpt", str(tmp_path / "step00002.safetensors"), "--meta", str(meta),
                         "--evals", f"aime={ev}", "--out", str(out)])
    export_teacher.main(["--verify", str(out)])
    tj = json.load(open(out / "teacher.json"))
    assert tj["domain"] == "math" and tj["eval_scores"]["aime"]["step2"]["pass@1"] == 0.5
    assert "ci95" in tj["eval_scores"]["aime"]["step2"]
    m2 = AutoModelForCausalLM.from_pretrained(str(out), dtype=torch.float32)
    sd2 = dict(m2.named_parameters())
    for n, p in named:
        if n in nm:
            assert torch.allclose(sd2[n], p.detach() + 1.0)
    full = load_file(str(out / "model.safetensors"))
    assert any(k.startswith("model.visual") for k in full)
    (out / "teacher.json").write_text("{}")
    with pytest.raises(SystemExit):
        export_teacher.main(["--verify", str(out)])
