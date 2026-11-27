"""Build a kept teacher directory (docs/mopd-tpu-plan.md, "Keeping the trained teachers").

The learner checkpoint holds only the text model's trained tensors (checkpoint names, model.language_model.*). They
are merged into the base checkpoint's full state dict, so the vision tower and every other tensor stay unchanged and
the directory loads as the base architecture in vLLM, HF transformers and our learner. Config and tokenizer files
are copied from the base snapshot. teacher.json records identity, recipe, run, data, provenance and eval scores;
SHA256SUMS covers every file; COMPLETE is written last, and readers ignore a directory without it.

    python -m qrl.export_teacher --base /dev/shm/models/Qwen3.5-2B --ckpt RUN/ckpt/step00030.safetensors \
        --run-dir RESULTS/RUN --meta meta.json --evals aime=RUN/eval.step30.json ifeval=... --out DIR
    python -m qrl.export_teacher --verify DIR      # checks SHA256SUMS and COMPLETE, then loads DIR in HF transformers
"""
import argparse
import glob
import hashlib
import json
import os
import shutil

KEEP_EVAL_KEYS = ("pass@1", "inst_acc", "truncated_frac", "mean_len", "samples", "problems")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def merged_state(base, ckpt):
    from safetensors import safe_open
    from safetensors.torch import load_file

    trained = load_file(ckpt)
    state = {}
    for f in sorted(glob.glob(os.path.join(base, "*.safetensors"))):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                state[k] = h.get_tensor(k)
    missing = [k for k in trained if k not in state]
    if missing:
        raise KeyError(f"checkpoint tensors not in the base checkpoint: {missing[:5]}")
    for k, v in trained.items():
        if state[k].shape != v.shape:
            raise ValueError(f"{k}: shape {tuple(v.shape)} != base {tuple(state[k].shape)}")
        state[k] = v.to(state[k].dtype).contiguous()
    return state, len(trained)


def eval_summary(path, n_boot=2000):
    """{label: metrics without per_problem, plus a 95% bootstrap CI over problems x samples}."""
    import numpy as np

    out = {}
    for label, r in json.load(open(path)).items():
        m = {k: r[k] for k in KEEP_EVAL_KEYS if k in r}
        per = r.get("per_problem") or []
        if per:
            n = r["samples"]
            p = np.array([q["correct"] / n for q in per])
            rng = np.random.default_rng(0)
            draws = [rng.binomial(n, p[rng.integers(0, len(p), len(p))]).mean() / n for _ in range(n_boot)]
            m["ci95"] = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
        out[label] = m
    return out


def build(a):
    os.makedirs(a.out, exist_ok=False)
    from safetensors.torch import save_file

    state, n_trained = merged_state(a.base, a.ckpt)
    save_file(state, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    size = os.path.getsize(os.path.join(a.out, "model.safetensors"))
    json.dump({"metadata": {"total_size": size}, "weight_map": {k: "model.safetensors" for k in sorted(state)}},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    for f in os.listdir(a.base):
        src = os.path.join(a.base, f)
        if os.path.isfile(src) and not f.endswith(".safetensors") and f != "model.safetensors.index.json" \
                and not f.startswith("."):
            shutil.copy(src, os.path.join(a.out, f))
    os.makedirs(os.path.join(a.out, "evals"))
    evals = {}
    for spec in a.evals:
        name, path = spec.split("=", 1)
        if os.path.exists(path):
            shutil.copy(path, os.path.join(a.out, "evals", f"{name}.json"))
            evals[name] = eval_summary(path)
    if a.run_dir:
        os.makedirs(os.path.join(a.out, "train_events"))
        for f in glob.glob(os.path.join(a.run_dir, "events", "*.jsonl")):
            shutil.copy(f, os.path.join(a.out, "train_events", os.path.basename(f)))
        if os.path.exists(os.path.join(a.run_dir, "config.json")):
            shutil.copy(os.path.join(a.run_dir, "config.json"), os.path.join(a.out, "train_events", "config.json"))
    meta = json.load(open(a.meta)) if a.meta else {}
    meta.update({"name": os.path.basename(a.out.rstrip("/")), "trained_tensors": n_trained,
                 "checkpoint": os.path.basename(a.ckpt), "eval_scores": evals,
                 "format": "HF Qwen3_5ForConditionalGeneration directory: text-model tensors from the learner "
                           "checkpoint merged into the base checkpoint; config/tokenizer copied from the base"})
    json.dump(meta, open(os.path.join(a.out, "teacher.json"), "w"), indent=1)
    files = sorted(os.path.relpath(p, a.out) for p in glob.glob(os.path.join(a.out, "**"), recursive=True)
                   if os.path.isfile(p))
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as f:
        for rel in files:
            f.write(f"{sha256(os.path.join(a.out, rel))}  {rel}\n")
    open(os.path.join(a.out, "COMPLETE"), "w").write(meta["name"] + "\n")
    print(json.dumps({"out": a.out, "files": len(files) + 2, "model_bytes": size, "trained_tensors": n_trained}))


def verify(d, hf=True):
    if not os.path.exists(os.path.join(d, "COMPLETE")):
        raise SystemExit(f"{d}: no COMPLETE marker")
    bad = []
    for line in open(os.path.join(d, "SHA256SUMS")):
        h, rel = line.rstrip("\n").split("  ", 1)
        if sha256(os.path.join(d, rel)) != h:
            bad.append(rel)
    if bad:
        raise SystemExit(f"{d}: checksum mismatch {bad}")
    print(f"{d}: OK")
    if hf:
        print(json.dumps(hf_load(d)))


def hf_load(d):
    """Load DIR as a causal LM in HF transformers (CPU): no missing or unexpected text-model weights, every
    trained tensor equal to the file's, and a finite forward pass."""
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m, info = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, output_loading_info=True)
    missing = [k for k in sorted(info["missing_keys"]) if "visual" not in k and "mtp" not in k]
    files = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    params = dict(m.named_parameters())
    from qrl.learner import export_name_map
    nm = export_name_map(list(params), list(files))
    mism = 0
    with safe_open(os.path.join(d, "model.safetensors"), "pt") as h:
        for n, k in nm.items():
            mism += int(not torch.equal(params[n].detach(), h.get_tensor(k).to(params[n].dtype)))
    tok = AutoTokenizer.from_pretrained(d)
    with torch.no_grad():
        out = m(**tok("The capital of France is", return_tensors="pt")).logits.float()
    res = {"hf_load": d, "missing": missing[:5], "unexpected": sorted(info["unexpected_keys"])[:5], "tensors_checked": len(nm),
           "tensor_mismatch": mism, "logits_finite": bool(torch.isfinite(out).all())}
    if missing or mism or not res["logits_finite"] or not nm:
        raise SystemExit(f"HF load check failed: {res}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", default=None)
    ap.add_argument("--base")
    ap.add_argument("--ckpt")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--meta", default="")
    ap.add_argument("--evals", nargs="*", default=[])
    ap.add_argument("--out")
    ap.add_argument("--no-hf", action="store_true", help="--verify: checksums only")
    a = ap.parse_args(argv)
    if a.verify:
        verify(a.verify, hf=not a.no_hf)
    else:
        build(a)


if __name__ == "__main__":
    main()
