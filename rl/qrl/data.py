"""AIME splits, prompt formatting, and the exact-match reward."""
import random
import re

TRAIN_DATASET = "gneubig/aime-1983-2024"
EVAL_DATASETS = ("HuggingFaceH4/aime_2024", "MathArena/aime_2025")
TRAIN_MAX_YEAR = 2023

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def load_train():
    from datasets import load_dataset

    ds = load_dataset(TRAIN_DATASET, split="train")
    rows = [
        {"id": r["ID"], "domain": "math", "question": r["Question"], "answer": normalize(str(r["Answer"]))}
        for r in ds
        if int(r["Year"]) <= TRAIN_MAX_YEAR
    ]
    return rows


# v2 adds AIME 2026 and the integer-answer problems of three 2025-26 competitions (exact match stays
# reliable). Every problem post-dates the training cutoff (AIME <= 2023).
EVAL_V2_EXTRA = (("MathArena/aime_2026", "aime2026", False), ("MathArena/hmmt_feb_2025", "hmmt2025", True),
                 ("MathArena/hmmt_feb_2026", "hmmt2026", True), ("MathArena/brumo_2025", "brumo2025", True))


def load_eval(version="v1"):
    from datasets import load_dataset

    rows = []
    for r in load_dataset(EVAL_DATASETS[0], split="train"):
        rows.append({"id": f"2024-{r['id']}", "question": r["problem"], "answer": normalize(str(r["answer"]))})
    for r in load_dataset(EVAL_DATASETS[1], split="train"):
        rows.append({"id": f"2025-{r['problem_idx']}", "question": r["problem"], "answer": normalize(str(r["answer"]))})
    if version == "v2":
        for name, tag, integer_only in EVAL_V2_EXTRA:
            for r in load_dataset(name, split="train"):
                ans = str(r["answer"]).strip()
                if integer_only and not re.fullmatch(r"-?\d+", ans):
                    continue
                rows.append({"id": f"{tag}-{r['problem_idx']}", "question": r["problem"], "answer": normalize(ans)})
    return rows


def filter_by_profile(rows, profile_path, keep_zero_frac=0.2, seed=12345):
    """Difficulty filter from a base-model profile ({id: {"correct": c, "n": n}}): keep prompts the base
    model solves sometimes but not always, plus a fixed random share of never-solved ones so the
    curriculum does not collapse onto what the base model can already do. Independent of the run seed,
    so every run trains on the same subset."""
    import json

    prof = json.load(open(profile_path))
    prof = prof.get("prompts", prof)
    rng = random.Random(seed)
    keep = []
    for r in rows:
        p = prof.get(str(r["id"]))
        if p is None:
            continue
        c, n = p["correct"], p["n"]
        if 0 < c < n or (c == 0 and rng.random() < keep_zero_frac):
            keep.append(r)
    return keep


def prompt_order(n, seed):
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


def batch_prompts(rows, batch_idx, prompts_per_batch, seed):
    """Prompts for training batch `batch_idx`; identical across modes for a given seed."""
    order = prompt_order(len(rows), seed)
    start = batch_idx * prompts_per_batch
    return [rows[order[(start + i) % len(rows)]] for i in range(prompts_per_batch)]


def parse_domains(spec):
    """'math:8,if:8' -> [('math', 8), ('if', 8)]: prompts per batch from each domain, in batch order."""
    out = []
    for part in spec.split(","):
        d, n = part.split(":")
        if d not in ("math", "if"):
            raise ValueError(f"unknown domain {d}")
        out.append((d, int(n)))
    return out


def domain_of(row):
    return row.get("domain", "math")


def mixed_batch(rows_by_domain, spec, batch_idx, seed):
    """Batch `batch_idx`: each domain's prompts in its own seeded order, domains concatenated in spec order.
    A single 'math:16' spec gives exactly batch_prompts(math_rows, batch_idx, 16, seed)."""
    out = []
    for d, n in parse_domains(spec):
        out += batch_prompts(rows_by_domain[d], batch_idx, n, seed)
    return out


def row_prompt(tokenizer, row, enable_thinking=False):
    """Math rows get the boxed-answer instruction; IF prompts are used verbatim (their constraint is the task)."""
    return chat_prompt(tokenizer, row["question"], enable_thinking, suffix=INSTRUCTION if domain_of(row) == "math" else None)


def row_reward(text, row):
    if domain_of(row) == "if":
        from . import ifeval

        return ifeval.reward(text, row["ground_truth"])
    return reward(text, row["answer"])


def chat_prompt(tokenizer, question, enable_thinking=False, suffix=INSTRUCTION):
    messages = [{"role": "user", "content": f"{question}\n{suffix}" if suffix else question}]
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    except Exception:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def normalize(ans):
    if ans is None:
        return None
    s = ans.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("$", "").replace(",", "").replace("\\!", "").replace(" ", "")
    s = re.sub(r"\^\{?\\circ\}?", "", s)
    s = s.rstrip(".")
    if re.fullmatch(r"[+-]?\d+(\.0+)?", s):
        return str(int(float(s)))
    return s


def reward(completion, answer):
    pred = normalize(last_boxed(completion))
    return 1.0 if pred is not None and pred == answer else 0.0
