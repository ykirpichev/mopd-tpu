"""Weight-space merges of the stored teachers (eval-only baseline, docs/mopd-paper-plan.md section 3).

Task vectors are tau_i = theta_i - theta_base over the text-model tensors (model.language_model.*, which includes the
tied embedding); every other tensor (vision tower, MTP) is copied from the base. Arithmetic is in fp32 per tensor and
the result is stored in the base dtype.

    linear:  sum_i w_i theta_i                       (w_i = 1/n: plain average)
    ta:      theta_base + lam * sum_i tau_i          (task arithmetic, Ilharco et al.)
    ties:    theta_base + lam * TIES(tau_1..n; k)    (Yadav et al.: keep the top-k fraction of each tau_i by
             magnitude, elect a per-entry sign from the trimmed sum, average the trimmed entries that agree with it)

The output is a complete HF directory (model.safetensors + index, config and tokenizer copied from the base) plus
merge.json and SHA256SUMS, loadable by vLLM and HF like a teacher directory.

    python -m qrl.merge_teachers --base BASE --teachers math=T1 if=T2 --method ties --density 0.2 --lam 1 --out DIR
"""
import argparse
import glob
import json
import os
import shutil
import time

import torch

TEXT_PREFIX = "model.language_model."


def load_state(d):
    from safetensors import safe_open

    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                out[k] = h.get_tensor(k)
    return out


def trim(tau, density):
    """Zero all but the top `density` fraction of |tau| (ties at the threshold are kept)."""
    if density >= 1.0:
        return tau
    flat = tau.abs().flatten()
    k = max(1, int(round(density * flat.numel())))
    if k >= flat.numel():
        return tau
    thr = flat.kthvalue(flat.numel() - k + 1).values
    return torch.where(tau.abs() >= thr, tau, torch.zeros_like(tau))


def ties_merge(taus, density):
    trimmed = [trim(t, density) for t in taus]
    stack = torch.stack(trimmed)
    sign = torch.sign(stack.sum(0))
    agree = (torch.sign(stack) == sign.unsqueeze(0)) & (stack != 0)
    num = (stack * agree).sum(0)
    cnt = agree.sum(0).clamp(min=1)
    return num / cnt


def merge_tensor(method, base, ts, weights, lam, density):
    b = base.float()
    if method == "linear":
        return sum(w * t.float() for w, t in zip(weights, ts))
    taus = [t.float() - b for t in ts]
    if method == "ta":
        return b + lam * sum(taus)
    if method == "ties":
        return b + lam * ties_merge(taus, density)
    raise ValueError(method)


def merge(base_dir, teacher_dirs, method, lam=1.0, density=1.0, weights=None):
    base = load_state(base_dir)
    teachers = [load_state(d) for d in teacher_dirs]
    weights = weights or [1.0 / len(teachers)] * len(teachers)
    out, stats = {}, {"tensors_merged": 0, "tensors_copied": 0, "tau_nonzero_frac": [0.0] * len(teachers),
                      "tau_sign_conflict_frac": 0.0, "numel_merged": 0}
    for k, v in base.items():
        if not k.startswith(TEXT_PREFIX):
            out[k] = v
            stats["tensors_copied"] += 1
            continue
        ts = [t[k] for t in teachers]
        for t in ts:
            if t.shape != v.shape:
                raise ValueError(f"{k}: teacher shape {tuple(t.shape)} != base {tuple(v.shape)}")
        m = merge_tensor(method, v, ts, weights, lam, density)
        out[k] = m.to(v.dtype).contiguous()
        with torch.no_grad():
            taus = [t.float() - v.float() for t in ts]
            for i, tau in enumerate(taus):
                stats["tau_nonzero_frac"][i] += float((tau != 0).sum())
            if len(taus) == 2:
                stats["tau_sign_conflict_frac"] += float(((taus[0] * taus[1]) < 0).sum())
        stats["numel_merged"] += v.numel()
        stats["tensors_merged"] += 1
    n = max(1, stats["numel_merged"])
    stats["tau_nonzero_frac"] = [x / n for x in stats["tau_nonzero_frac"]]
    stats["tau_sign_conflict_frac"] /= n
    return out, stats


def write_dir(out_dir, state, base_dir, meta):
    from safetensors.torch import save_file

    from .export_teacher import sha256

    os.makedirs(out_dir, exist_ok=False)
    save_file(state, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    size = os.path.getsize(os.path.join(out_dir, "model.safetensors"))
    json.dump({"metadata": {"total_size": size}, "weight_map": {k: "model.safetensors" for k in sorted(state)}},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=1)
    for f in os.listdir(base_dir):
        src = os.path.join(base_dir, f)
        if os.path.isfile(src) and not f.endswith(".safetensors") and f != "model.safetensors.index.json" \
                and not f.startswith("."):
            shutil.copy(src, os.path.join(out_dir, f))
    json.dump(meta, open(os.path.join(out_dir, "merge.json"), "w"), indent=1)
    files = sorted(os.path.relpath(p, out_dir) for p in glob.glob(os.path.join(out_dir, "**"), recursive=True)
                   if os.path.isfile(p))
    with open(os.path.join(out_dir, "SHA256SUMS"), "w") as f:
        for rel in files:
            f.write(f"{sha256(os.path.join(out_dir, rel))}  {rel}\n")
    return size


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--teachers", nargs="+", required=True, help="name=DIR ...")
    ap.add_argument("--method", choices=["linear", "ta", "ties"], required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--density", type=float, default=1.0, help="ties: fraction of each task vector kept")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    names, dirs = zip(*(s.split("=", 1) for s in a.teachers))
    state, stats = merge(a.base, list(dirs), a.method, a.lam, a.density)
    meta = {"method": a.method, "lam": a.lam, "density": a.density, "teachers": dict(zip(names, dirs)),
            "base": a.base, **stats, "merge_s": time.time() - t0}
    size = write_dir(a.out, state, a.base, meta)
    print(json.dumps({"out": a.out, "model_bytes": size, **{k: v for k, v in meta.items() if k != "teachers"}}))


if __name__ == "__main__":
    main()
