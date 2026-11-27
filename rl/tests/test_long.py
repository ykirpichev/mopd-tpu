"""CPU tests for plan65k1long and plan55sync60: intermediate checkpoints are kept for every eval step, and the pair
script's eval/ckpt plumbing (CKPT_STEPS, EVAL_SPECS, EVAL_SPLIT) produces one eval per spec on each run's own chips."""
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tests.test_cpu import tiny_model  # noqa: E402,F401
from tests.test_v2 import _loop  # noqa: E402


@pytest.mark.parametrize("behavior", ["use", "none"])
def test_intermediate_checkpoints_kept(tiny_model, tmp_path, behavior):  # noqa: F811
    ck = tmp_path / "ckpt"
    _, ev = _loop(tiny_model, tmp_path, "async", 1, 6, extra_run=("--staleness-mode", "fixed"),
                  extra_learner=f"--behavior-logprobs {behavior} --loss tis --ckpt-dir {ck} --ckpt-steps 2,4,6")
    assert sorted(os.listdir(ck)) == ["step00002.safetensors", "step00004.safetensors", "step00006.safetensors"]
    train = sorted((e for e in ev if e["kind"] == "train"), key=lambda e: e["step"])
    assert [e["staleness"] for e in train] == [0, 1, 1, 1, 1, 1]


def test_sync_intermediate_checkpoints_kept(tiny_model, tmp_path):  # noqa: F811
    ck = tmp_path / "ckpt"
    _, ev = _loop(tiny_model, tmp_path, "sync", 0, 6,
                  extra_learner=f"--behavior-logprobs use --loss tis --ckpt-dir {ck} --ckpt-steps 2,4,6")
    assert sorted(os.listdir(ck)) == ["step00002.safetensors", "step00004.safetensors", "step00006.safetensors"]
    assert all(e["staleness"] == 0 for e in ev if e["kind"] == "train")


PAIR_HARNESS = r"""
set -u
RES=$1; MODEL=m; CAP=6144; STEPS=60; LEARNER_FLAGS='--gdn-chunk 256 --mb-tokens 8192'; TRAIN_FILTER=1
CKPT_STEPS=20,40,60; EVAL_SPECS='step20 16;step40 16;step60 16'; EVAL_SPLIT=1
log() { echo "LOG $*"; }
X() { echo "X $*"; }
in_chip() { echo "CHIP $1 PORT $2 $*"; }
waitjobs() { wait; }
final_sync() { :; }
stage() { :; }
sudo() { if [ "$1" = docker ]; then echo "DOCKER $*"; else "$@"; fi; }
mkdir -p $RES/pilot; echo {} > $RES/pilot/profile.json
eval "$(sed -n '/^evaluate_run() {/,/^}/p; /^train() {/,/^}/p; /^run_pair() {/,/^}/p' $2 \
       | sed '/first-grant watcher/,/done ) &/d')"
RUN_A=$3; RUN_B=$4
run_pair
"""


def _pair(tmp_path, run_a, run_b):
    out = subprocess.run(["bash", "-c", PAIR_HARNESS, "x", str(tmp_path), os.path.join(ROOT, "tpu", "remote_v2.sh"),
                          run_a, run_b], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_sync_pair_script(tmp_path):
    o = _pair(tmp_path, "lsync-s0 sync bounded 0 use 0", "lsync-s1 sync bounded 0 use 1")
    runs = [l for l in o.splitlines() if l.startswith("DOCKER") and "qrl.run" in l]
    assert len(runs) == 2 and all("--mode sync --staleness 0 --staleness-mode bounded --steps 60" in l for l in runs)
    assert all("--ckpt-steps 20,40,60" in l and "--behavior-logprobs use" in l and "--mb-tokens 8192" in l for l in runs)
    assert {re.search(r"--seed (\d)", l).group(1) for l in runs} == {"0", "1"}
    evals = [l for l in o.splitlines() if l.startswith("CHIP") and "qrl.evaluate" in l]
    for name, chips in (("lsync-s0", {"0", "1"}), ("lsync-s1", {"2", "3"})):
        mine = [l for l in evals if f"/results/{name}/" in l]
        assert {re.search(r"CHIP (\d)", l).group(1) for l in mine} == chips and len(mine) == 6


def test_pair_script_evals_three_checkpoints_split(tmp_path):
    o = _pair(tmp_path, "lk1is-s0 async fixed 1 use 0", "lk1nois-s0 async fixed 1 none 0")
    runs = [l for l in o.splitlines() if l.startswith("DOCKER") and "qrl.run" in l]
    assert len(runs) == 2 and all("--steps 60" in l and "--ckpt-steps 20,40,60" in l for l in runs)
    assert all("--staleness 1 --staleness-mode fixed" in l and "--gdn-chunk 256 --mb-tokens 8192" in l for l in runs)
    assert any("--behavior-logprobs use" in l for l in runs) and any("--behavior-logprobs none" in l for l in runs)
    evals = [l for l in o.splitlines() if l.startswith("CHIP") and "qrl.evaluate" in l]
    for name, chips in (("lk1is-s0", {"0", "1"}), ("lk1nois-s0", {"2", "3"})):
        mine = [l for l in evals if f"/results/{name}/" in l]
        assert {re.search(r"CHIP (\d)", l).group(1) for l in mine} == chips
        for s in (20, 40, 60):
            shard = [l for l in mine if f"step{s}=/results/{name}/ckpt/step{s:05d}.safetensors" in l]
            assert len(shard) == 2 and all("--samples 16 --max-tokens 6144" in l and "--eval-set v2" in l for l in shard)
        merges = [l for l in o.splitlines() if l.startswith("X") and "--merge" in l and f"/results/{name}/" in l]
        assert sorted(re.search(r"--out /results/\S+/eval\.(step\d+)\.json", l).group(1) for l in merges) == \
            ["step20", "step40", "step60"]
    assert "eval A on chips 0,1" in o and "eval A on chips 0-3" not in o
