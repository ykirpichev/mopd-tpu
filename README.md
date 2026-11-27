# mopd-tpu

This is the code for **"When Is On-Policy Distillation Worth It?"** (Yury Kirpichev, 2026; arXiv link to follow).

The study distills a math specialist and an instruction-following specialist into one Qwen3.5-2B student with multi-teacher on-policy distillation (MOPD). It compares that against joint RL (Mix-RL), single-teacher OPD, offline distillation on teacher samples, and weight merging (linear, task arithmetic, TIES). Everything runs on Cloud TPU v6e-4, with a TorchTPU learner and a vLLM generator running one step asynchronous.

**Live page:** https://ykirpichev.github.io/mopd-tpu/. Companion study: [Async vs sync RL on TPU v6e](https://ykirpichev.github.io/async-vs-sync-rl-tpu/).

**Main finding (4 seeds; ±3 pp equivalence; Holm-corrected).** MOPD reaches each teacher in its own domain, and routing is essential: MOPD is +6 pp over a single teacher on IF. With same-origin teachers this close to the student, however, offline sequence-level KD and a task-arithmetic merge are statistically equivalent to MOPD, at half the compute and at zero compute respectively.

## Layout

```
rl/
  qrl/                  training, distillation, merging, evaluation, reports
    run.py              async orchestrator (generator + learner [+ teacher] processes)
    rollout.py          vLLM generator; --teacher-gen samples from the routed teacher (offline-KD baseline)
    learner.py          TorchTPU learner; objectives grpo, opd (sampled-token MOPD), opd_dist (exact full-vocab KL), sft
    teacher.py          teacher scoring service; ships final hidden states for opd_dist
    grpo.py             group-normalized advantages, truncated-IS policy loss
    merge_teachers.py   linear / task-arithmetic / TIES merges into a full HF directory
    export_teacher.py   teacher export + SHA256SUMS + round-trip verify
    ifeval.py, ifv/     IF rewards and IFEval / IFBench checkers (vendored, see rl/qrl/ifv/NOTICE.md)
    report_paper.py     hierarchical bootstrap, TOST, Holm, GLMM, paper figures
    paper_extras.py     compute table, clip analysis, seed spread
  tpu/                  TPU VM orchestration: mopd.sh work loop, drivers, spend guard, plan JSONs
  plan/                 plan helpers used by the VM loop
  tests/                CPU tests
docs/                   the study site (docs/index.html, charts in docs/charts/), served by GitHub Pages from main /docs
```

The plans used in the paper are `rl/tpu/mopdpilot.json` (pilot), `rl/tpu/mopdfull.json` (main study, seeds s0 and s1, plus the teachers) and `rl/tpu/mopdpaper.json` (seeds s2 and s3, SFT, merges and reference re-evals). See `rl/README.md` for the training loop, the CPU tests and how a TPU run is driven.

## Statistics

```bash
cd rl
python -m qrl.report_paper --roots results/mopdfull/raw results/mopdpaper/raw --out paper.json --media figures/
python -m qrl.paper_extras --roots results/mopdfull/raw results/mopdpaper/raw --paper paper.json --out extras.json
```

The raw per-run results are not part of this repository yet. They are event logs and per-problem correctness counts; they contain no problem text.

## Teachers

The four GRPO teachers (T-math and T-IF, 30 and 60 steps, fine-tuned from Qwen3.5-2B) will be published on Hugging Face under Apache-2.0. The link will be added here.

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The vendored verifiers in `rl/qrl/ifv/` are Apache-2.0 upstream and keep their notices.

Data and benchmarks are not included. They are used under their own licenses:

- AIME 1983–2024 (`gneubig/aime-1983-2024`, CC0-1.0).
- RLVR-IFeval (ODC-BY-1.0).
- IFEval (Apache-2.0).
- IFBench (ODC-BY-1.0).
- MathArena AIME / HMMT / BRUMO (CC BY-NC-SA 4.0).
- AIME 2024 (`HuggingFaceH4/aime_2024`; no license stated, problems © MAA).
- IFEval eval set (`google/IFEval`, Apache-2.0).
