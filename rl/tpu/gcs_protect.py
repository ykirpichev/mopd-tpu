"""Deletes in the qrl-bc4a bucket that never touch the kept prefixes.

cache/ (the vLLM compile cache) and teachers/ (trained teacher checkpoints, docs/mopd-tpu-plan.md) outlive every
plan. A plan's teardown deletes its own prefix (plan90/, plan25k1/, ...) and nothing else, through this script:
it refuses the bucket root, wildcards at the top level, any kept prefix, and anything containing one, and never
deletes buckets.

    python rl/tpu/gcs_protect.py check gs://BUCKET/plan90/            # exit 0 if allowed, 2 if refused
    python rl/tpu/gcs_protect.py rm -r gs://BUCKET/plan90/            # gcloud storage rm, after the check
    python rl/tpu/gcs_protect.py hold gs://BUCKET/teachers/NAME/      # temporary hold on every object under it
"""
import subprocess
import sys

BUCKET = "qrl-bc4a-compile-cache-tpu-builders-509006"
PROTECTED = ("cache/", "teachers/")


class Refused(ValueError):
    pass


def check(url, protected=PROTECTED):
    """Raise Refused unless url names objects strictly inside one non-protected top-level prefix."""
    if not url.startswith("gs://"):
        raise Refused(f"not a gs:// url: {url}")
    bucket, _, path = url[len("gs://"):].partition("/")
    if not bucket:
        raise Refused(f"no bucket: {url}")
    path = path.lstrip("/")
    top, sep, _ = path.partition("/")
    if ".." in path.split("/"):
        raise Refused(f"refusing '..' in {url}")
    if not path or not sep or not top or any(c in top for c in "*?[]{}") or top == ".":
        raise Refused(f"refusing a bucket-level or top-level wildcard delete: {url}")
    for p in protected:
        if (top + "/") == p:
            raise Refused(f"{url} is under kept prefix {p}")
    return bucket, path


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if not a or a[0] not in ("check", "rm", "hold"):
        print(__doc__)
        return 1
    cmd, rest = a[0], a[1:]
    urls = [x for x in rest if x.startswith("gs://")]
    flags = [x for x in rest if not x.startswith("gs://")]
    if cmd == "hold":
        for u in urls:
            if not any(u[len("gs://"):].partition("/")[2].lstrip("/").startswith(p) for p in PROTECTED):
                print(f"hold is only for kept prefixes {PROTECTED}: {u}", file=sys.stderr)
                return 2
            target = u.rstrip("/") + "/**"
            r = subprocess.run(["gcloud", "storage", "objects", "update", target, "--temporary-hold"])
            if r.returncode:
                return r.returncode
        return 0
    if not urls:
        print("no gs:// url given", file=sys.stderr)
        return 1
    try:
        for u in urls:
            check(u)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    if cmd == "check":
        return 0
    if any(f not in ("-r", "--recursive", "--quiet", "-q") for f in flags):
        print(f"unsupported flags {flags}", file=sys.stderr)
        return 1
    return subprocess.run(["gcloud", "storage", "rm", *flags, *urls]).returncode


if __name__ == "__main__":
    sys.exit(main())
