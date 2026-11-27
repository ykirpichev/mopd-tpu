"""Filesystem channel between the rollout process and the learner process.

Layout under run_dir (put it on /dev/shm so a weight push is a memory copy):
  weights/v00012.safetensors   bf16 policy weights for version 12 (atomic rename)
  weights/LATEST               "12"
  rollouts/b000012.pt          rollout batch 12 (consumed and deleted by the learner, or by the teacher service)
  scored/b000012.pt            batch 12 with teacher log-probs (distillation: teacher service -> learner)
  scored/b000012.hid.pt        the teacher's final hidden states for it (full-distribution arm; written first)
  teacher_heads/<name>.safetensors  each teacher's output layer, written once by the teacher service
  events/<proc>.jsonl          timestamped busy/idle intervals and metrics
  STOP                         either side asks the other to exit
"""
import json
import os
import time

import torch


class Channel:
    def __init__(self, run_dir, keep_weights=2):
        self.root = run_dir
        self.wdir = os.path.join(run_dir, "weights")
        self.rdir = os.path.join(run_dir, "rollouts")
        self.edir = os.path.join(run_dir, "events")
        self.sdir = os.path.join(run_dir, "scored")
        self.hdir = os.path.join(run_dir, "teacher_heads")
        self.keep_weights = keep_weights
        for d in (self.wdir, self.rdir, self.edir, self.sdir, self.hdir):
            os.makedirs(d, exist_ok=True)

    def stop_requested(self):
        return os.path.exists(os.path.join(self.root, "STOP"))

    def request_stop(self, reason=""):
        with open(os.path.join(self.root, "STOP"), "w") as f:
            f.write(reason)

    def latest_version(self):
        try:
            with open(os.path.join(self.wdir, "LATEST")) as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError):
            return 0

    def weights_path(self, version):
        return os.path.join(self.wdir, f"v{version:05d}.safetensors")

    def publish_weights(self, version, state_dict):
        from safetensors.torch import save_file

        path = self.weights_path(version)
        save_file(state_dict, path + ".tmp")
        os.replace(path + ".tmp", path)
        tmp = os.path.join(self.wdir, "LATEST.tmp")
        with open(tmp, "w") as f:
            f.write(str(version))
        os.replace(tmp, os.path.join(self.wdir, "LATEST"))
        stale = version - self.keep_weights
        if stale > 0 and os.path.exists(self.weights_path(stale)):
            os.remove(self.weights_path(stale))

    def _batch_path(self, idx, shard=None, stage="rollouts"):
        d = self.rdir if stage == "rollouts" else self.sdir
        return os.path.join(d, f"b{idx:06d}.pt" if shard is None else f"b{idx:06d}.s{shard}.pt")

    def put_batch(self, idx, batch, shard=None, stage="rollouts"):
        path = self._batch_path(idx, shard, stage)
        torch.save(batch, path + ".tmp")
        os.replace(path + ".tmp", path)

    def batch_ready(self, idx, num_shards=1, stage="rollouts"):
        if num_shards == 1:
            return os.path.exists(self._batch_path(idx, None, stage))
        return all(os.path.exists(self._batch_path(idx, s, stage)) for s in range(num_shards))

    def take_batch(self, idx, num_shards=1, stage="rollouts", keep=False):
        """Load (and, unless keep, delete) batch idx. Shards are concatenated in shard order; the merged batch's
        version is the oldest shard's, so staleness is reported for the worst shard."""
        paths = ([self._batch_path(idx, None, stage)] if num_shards == 1
                 else [self._batch_path(idx, s, stage) for s in range(num_shards)])
        parts = [torch.load(p, weights_only=False) for p in paths]
        for p in paths if not keep else ():
            os.remove(p)
        if num_shards == 1:
            return parts[0]
        merged = dict(parts[0])
        merged["seqs"] = [s for p in parts for s in p["seqs"]]
        merged["version"] = min(p["version"] for p in parts)
        merged["shard_versions"] = [p["version"] for p in parts]
        merged["gen_start"] = min(p["gen_start"] for p in parts)
        merged["gen_end"] = max(p["gen_end"] for p in parts)
        return merged

    def remove_batch(self, idx, num_shards=1, stage="rollouts"):
        for s in [None] if num_shards == 1 else range(num_shards):
            try:
                os.remove(self._batch_path(idx, s, stage))
            except FileNotFoundError:
                pass

    def hidden_path(self, idx, shard=None):
        return os.path.join(self.sdir, f"b{idx:06d}.hid.pt" if shard is None else f"b{idx:06d}.s{shard}.hid.pt")

    def put_hidden(self, idx, hidden, shard=None):
        """{key: [completion tokens, hidden] bf16}; written before the scored batch, whose sequences carry
        s["hid"] = (file name, key), so shards merge without renumbering."""
        path = self.hidden_path(idx, shard)
        torch.save(hidden, path + ".tmp")
        os.replace(path + ".tmp", path)
        return os.path.basename(path)

    def take_hidden(self, name):
        path = os.path.join(self.sdir, name)
        h = torch.load(path, weights_only=True)
        os.remove(path)
        return h

    def head_path(self, name):
        return os.path.join(self.hdir, f"{name}.safetensors")

    def put_head(self, name, weight):
        from safetensors.torch import save_file

        path = self.head_path(name)
        save_file({"weight": weight}, path + ".tmp")
        os.replace(path + ".tmp", path)


class EventLog:
    def __init__(self, channel, proc):
        self.f = open(os.path.join(channel.edir, f"{proc}.jsonl"), "a", buffering=1)
        self.proc = proc

    def interval(self, kind, start, end, **kw):
        self.f.write(json.dumps({"proc": self.proc, "kind": kind, "start": start, "end": end, **kw}) + "\n")

    def point(self, kind, **kw):
        self.f.write(json.dumps({"proc": self.proc, "kind": kind, "t": time.time(), **kw}) + "\n")


def wait_until(pred, channel, poll=0.05, timeout=None):
    t0 = time.time()
    while not pred():
        if channel.stop_requested():
            raise SystemExit("stop requested")
        if timeout is not None and time.time() - t0 > timeout:
            raise TimeoutError("wait_until timed out")
        time.sleep(poll)
    return time.time() - t0
