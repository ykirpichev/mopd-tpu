"""Launch one run (rollout process + learner process) and wait for both.

TPU chip pinning: each process sees exactly one chip through TPU_VISIBLE_CHIPS
and its own TPU_PROCESS_PORT, so two runs can share a 4-chip v6e host
(run A on chips 0/1, run B on chips 2/3).
"""
import argparse
import json
import os
import subprocess
import sys
import time


def tpu_env(chip, port):
    return {
        "TPU_VISIBLE_CHIPS": str(chip),
        "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1",
        "TPU_PROCESS_BOUNDS": "1,1,1",
        "TPU_PROCESS_PORT": str(port),
        "TPU_PROCESS_ADDRESSES": f"localhost:{port}",
        "CLOUD_TPU_TASK_ID": "0",
        "TORCH_TPU_TOPOLOGY": "1,1,1",
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["sync", "async"], required=True)
    ap.add_argument("--staleness", type=int, default=2, help="k for async (sync forces 0)")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout-backend", default="vllm")
    ap.add_argument("--learner-device", default="tpu")
    ap.add_argument("--rollout-chip", type=int, default=0)
    ap.add_argument("--rollout-chips", default="", help="comma list; >1 chip = one rollout engine per chip, "
                    "each generating a disjoint shard of every batch")
    ap.add_argument("--learner-chip", type=int, default=1)
    ap.add_argument("--port-base", type=int, default=8476)
    ap.add_argument("--staleness-mode", choices=["bounded", "fixed"], default="bounded")
    ap.add_argument("--rollout-args", default="", help="extra args for qrl.rollout")
    ap.add_argument("--learner-args", default="", help="extra args for qrl.learner")
    ap.add_argument("--teacher-chips", default="", help="comma list: one teacher prefill service per chip "
                    "(qrl.teacher; distillation objectives). Each scores a disjoint share of every batch")
    ap.add_argument("--teacher-args", default="", help="extra args for qrl.teacher (--teachers is required)")
    args = ap.parse_args(argv)

    k = 0 if args.mode == "sync" else args.staleness
    chips = [int(c) for c in args.rollout_chips.split(",") if c] or [args.rollout_chip]
    n = len(chips)
    os.makedirs(args.run_dir, exist_ok=True)
    stop = os.path.join(args.run_dir, "STOP")
    if os.path.exists(stop):
        os.remove(stop)
    json.dump({**vars(args), "k": k, "t_launch": time.time()}, open(os.path.join(args.run_dir, "config.json"), "w"), indent=1)
    common = ["--run-dir", args.run_dir, "--model", args.model, "--steps", str(args.steps), "--seed", str(args.seed)]
    r_cmds = [[sys.executable, "-m", "qrl.rollout", *common, "--staleness", str(k), "--backend", args.rollout_backend,
               "--staleness-mode", args.staleness_mode, "--shard-index", str(i), "--num-shards", str(n),
               *args.rollout_args.split()] for i in range(n)]
    tchips = [int(c) for c in args.teacher_chips.split(",") if c]
    nt = len(tchips)
    l_cmd = [sys.executable, "-m", "qrl.learner", *common, "--device", args.learner_device, "--num-shards", str(n),
             "--keep-weights", str(max(2, k + 2)), *(["--teacher-shards", str(nt)] if nt else []),
             *args.learner_args.split()]
    t_cmds = [[sys.executable, "-m", "qrl.teacher", "--run-dir", args.run_dir, "--model", args.model, "--steps",
               str(args.steps), "--seed", str(args.seed), "--device", args.learner_device,
               "--rollout-shards", str(n), "--shard-index", str(i), "--num-shards", str(nt),
               *args.teacher_args.split()] for i in range(nt)]
    base = dict(os.environ)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    envs = []
    for i, chip in enumerate(chips + [None]):
        e = dict(base)
        if chip is not None and args.rollout_backend == "vllm":
            e.update(tpu_env(chip, args.port_base + 2 * i))
        if chip is None and args.learner_device == "tpu":
            e.update(tpu_env(args.learner_chip, args.port_base + 1))
        e["PYTHONPATH"] = here + os.pathsep + e.get("PYTHONPATH", "")
        envs.append(e)
    for i, chip in enumerate(tchips):
        e = dict(base)
        if args.learner_device == "tpu":
            e.update(tpu_env(chip, args.port_base + 2 * n + 1 + 2 * i))
        e["PYTHONPATH"] = here + os.pathsep + e.get("PYTHONPATH", "")
        envs.append(e)
    names = (["rollout" if n == 1 else f"rollout.s{i}" for i in range(n)] + ["learner"]
             + ["teacher" if nt == 1 else f"teacher.s{i}" for i in range(nt)])
    logs = [open(os.path.join(args.run_dir, f"{nm}.log"), "w") for nm in names]
    procs = [subprocess.Popen(c, env=e, stdout=lg, stderr=subprocess.STDOUT)
             for c, e, lg in zip(r_cmds + [l_cmd] + t_cmds, envs, logs)]
    rc = 0
    try:
        while any(p.poll() is None for p in procs):
            for p in procs:
                if p.poll() not in (None, 0):
                    rc = p.returncode
                    open(stop, "w").write(f"child exited {rc}")
            time.sleep(1)
        rc = rc or max(p.returncode for p in procs)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
    print(json.dumps({"run_dir": args.run_dir, "mode": args.mode, "k": k, "returncodes": [p.returncode for p in procs]}))
    sys.exit(rc)


if __name__ == "__main__":
    main()
