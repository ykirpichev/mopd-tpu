# Vendored instruction-following verifiers

All files in this directory come from Apache-2.0 projects and stay under the Apache License, Version 2.0.

| path | upstream |
|---|---|
| `if_functions.py` | [allenai/open-instruct](https://github.com/allenai/open-instruct), `open_instruct/if_functions.py`. It provides the RLVR-IFeval ground-truth rewards. |
| `ifeval/` | [google-research/google-research](https://github.com/google-research/google-research), `instruction_following_eval/` (IFEval) |
| `ifbench/` | [allenai/IFBench](https://github.com/allenai/IFBench), `ifbench/` (IFBench_test) |

**Changes from upstream:**

- Imports were rewritten to relative imports so the packages load from `qrl.ifv`.
- The NLTK punkt loader also accepts `punkt_tab` (nltk ≥ 3.9).

There are no other changes.
