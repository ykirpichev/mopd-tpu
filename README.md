# Multi-teacher on-policy distillation (MOPD) on TPU v6e

Distilling a math teacher and an instruction-following teacher into one Qwen3.5-2B student, on Google TPU v6e-4 with TorchTPU + vLLM.

**Live page:** https://ykirpichev.github.io/mopd-tpu/

Companion study: [Async vs sync RL on TPU v6e](https://ykirpichev.github.io/async-vs-sync-rl-tpu/).

## TL;DR

- MOPD merges both specialists into one 2B student; Mix-RL (GRPO on the same mixed prompts) does not. At step 60, MOPD matches the math teacher on AIME (−0.4 pp) and is within 1.5 pp of the IF teacher on IFEval and IFBench.
- MOPD beats Mix-RL on all three benchmarks and closes 95% / 78% / 76% of the base-to-teacher gap (Mix-RL: 54% / 56% / 22%).
- Routing is what buys the IF gains: distilling from the math teacher alone learns nothing on IF.
- The exact full-vocabulary loss (MOPD-dist) ties with sampled-token MOPD within ±3 pp, for about 3.5% more step time.
- Total cost about $295 (pilot ≈ $78, full ≈ $217).

## Layout

The site is the static page `docs/index.html` with charts in `docs/charts/`, served by GitHub Pages from `main` `/docs`.

Code and raw results are on branch `cursor/qwen38-async-vs-sync-rl-bc4a` under `rl/`.
