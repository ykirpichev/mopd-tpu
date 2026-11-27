"""Instruction following: RLVR-IFeval training prompts and reward, IFEval / IFBench eval sets and scoring.

Training reward: 1.0 iff open-instruct's verifier for the prompt's single constraint accepts the answer (the
`IFEvalVerifierOld` path: `ground_truth` is JSON with `func_name` plus its arguments). Eval: prompt-level strict
accuracy, every listed instruction followed, with the IFEval (google-research) and IFBench (allenai) checkers.
The verifiers are vendored in qrl/ifv (Apache-2.0).
"""
import json
import os

TRAIN_DATASET = "allenai/RLVR-IFeval"
EVAL_SETS = {"ifeval": ("google/IFEval", "train"), "ifbench": ("allenai/IFBench_test", "train")}
NLTK_PACKAGES = ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger_eng")


def setup_nltk(download=True):
    """One NLTK data dir for every process (IFBench would otherwise download into its package dir)."""
    d = os.environ.setdefault("NLTK_DATA", os.path.expanduser("~/nltk_data"))
    import nltk

    if d not in nltk.data.path:
        nltk.data.path.insert(0, d)
    if download:
        for p in NLTK_PACKAGES:
            sub = {"punkt": "tokenizers/punkt", "punkt_tab": "tokenizers/punkt_tab", "stopwords": "corpora/stopwords",
                   "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng"}[p]
            try:
                nltk.data.find(sub)
            except LookupError:
                nltk.download(p, download_dir=d, quiet=True)
    return d


def load_train():
    from datasets import load_dataset

    rows = []
    for i, r in enumerate(load_dataset(TRAIN_DATASET, split="train")):
        msgs = r["messages"]
        if len(msgs) != 1 or msgs[0]["role"] != "user":
            continue
        rows.append({"id": f"if-{i}", "domain": "if", "question": msgs[0]["content"], "ground_truth": r["ground_truth"]})
    return rows


def strip_thinking(text):
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


def reward(completion, ground_truth):
    from .ifv.if_functions import IF_FUNCTIONS_MAP

    answer = strip_thinking(completion)
    if not answer:
        return 0.0
    c = json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)
    fn = IF_FUNCTIONS_MAP.get(c.pop("func_name", None))
    if fn is None:
        return 0.0
    try:
        return float(bool(fn(answer, **{k: v for k, v in c.items() if v is not None})))
    except Exception:  # noqa: BLE001  a verifier crash on odd text is a failed constraint, not a dead rollout
        return 0.0


def load_eval(name):
    from datasets import load_dataset

    ds_id, split = EVAL_SETS[name]
    rows = []
    for r in load_dataset(ds_id, split=split):
        kw = [{k: v for k, v in (a or {}).items() if v is not None} for a in r["kwargs"]]
        rows.append({"id": f"{name}-{r['key']}", "domain": "if", "question": r["prompt"],
                     "instruction_id_list": list(r["instruction_id_list"]), "kwargs": kw})
    return rows


def _registry(name):
    if name == "ifeval":
        from .ifv.ifeval import instructions_registry
    else:
        setup_nltk(download=False)
        from .ifv.ifbench import instructions_registry
    return instructions_registry.INSTRUCTION_DICT


def follow_list(row, response, name):
    """Strict check of every instruction of an eval row: list of bools (IFEval/IFBench evaluation_lib)."""
    reg = _registry(name)
    out = []
    resp = strip_thinking(response)
    for iid, kw in zip(row["instruction_id_list"], row["kwargs"]):
        try:
            inst = reg[iid](iid)
            inst.build_description(**kw)
            args = inst.get_instruction_args()
            if args and "prompt" in args:
                inst.build_description(prompt=row["question"])
            out.append(bool(resp.strip()) and bool(inst.check_following(resp)))
        except Exception:  # noqa: BLE001
            out.append(False)
    return out
