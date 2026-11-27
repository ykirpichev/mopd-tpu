"""GKE side of plan90: cluster qrl-bc4a-gke (zonal, us-east1-d) with autoscaled 0->1 TPU node pools (Spot and
flex-start ct6e-standard-4t). The "request" is the Job qrl-bc4a-work: its pending pod makes the autoscaler ask
for a TPU node, and deleting it (plus resizing the TPU pools to 0) cancels the request. The pod runs the same
vm_startup.sh (guard90 --self, vm_hbgate.sh, auto90 work loop) as a Compute VM, with GKE=1.

Talks to the Kubernetes API over HTTPS with a gcloud OAuth access token (no kubectl needed), so it runs the same
on the agent machine, in the Cloud Run requester and inside the pod. Every gcloud call has a timeout.

    python3 rl/tpu/gke90.py --plan plan.json status          # JSON: cluster, job, tpu nodes
    python3 rl/tpu/gke90.py --plan plan.json ensure-job      # create the Job if missing (replace a finished one)
    python3 rl/tpu/gke90.py --plan plan.json delete-job
    python3 rl/tpu/gke90.py --plan plan.json scale0 [POOL]   # resize TPU pools (or one) to 0 nodes
    python3 rl/tpu/gke90.py --plan plan.json kill-node NODE  # delete-job + resize that node's pool to 0
    python3 rl/tpu/gke90.py --plan plan.json delete-cluster
    python3 rl/tpu/gke90.py --plan plan.json job-yaml        # print the Job manifest (JSON)
    python3 rl/tpu/gke90.py --plan plan.json dry-job         # CPU dry run of the pod path (QRL_DRY=1, plan90-dry/)
    python3 rl/tpu/gke90.py --plan plan.json delete-dry-job
"""
import argparse
import base64
import json
import os
import shutil
import ssl
import subprocess
import sys
import urllib.error
import urllib.request

PROJECT = "tpu-builders-509006"
GCLOUD = shutil.which("gcloud") or os.path.expanduser("~/google-cloud-sdk/bin/gcloud")
JOB = "qrl-bc4a-work"
DRY_JOB = "qrl-bc4a-dry"
B = "gs://qrl-bc4a-compile-cache-tpu-builders-509006/plan90"
GRANTED = ("PROVISIONING", "STAGING", "RUNNING")

# The pod's entrypoint: install gcloud (the vLLM image has none), then run the same bootstrap as a Compute VM.
BOOT = r"""set -u
export HOME=/root GKE=1
mkdir -p /opt
if ! command -v gcloud >/dev/null; then
  for i in 1 2 3 4 5; do
    python3 -c "import urllib.request as u; u.urlretrieve('https://dl.google.com/dl/cloudsdk/channels/rapid/google-cloud-sdk.tar.gz', '/opt/gcs.tgz')" \
      && tar xzf /opt/gcs.tgz -C /opt && break
    sleep 20
  done
fi
export PATH=/opt/google-cloud-sdk/bin:$PATH
gcloud config set core/disable_usage_reporting true >/dev/null 2>&1
for i in $(seq 1 20); do
  gcloud storage cp %s/vm_startup.sh /root/vm_startup.sh --quiet && exec bash /root/vm_startup.sh
  sleep 30
done
echo "could not fetch vm_startup.sh; exiting"; exit 1
""" % B


def gc(args, timeout=120):
    try:
        return subprocess.run([GCLOUD, *args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", f"timeout after {timeout}s")


class Gke:
    def __init__(self, pl):
        g = pl["gke"]
        self.pl, self.name, self.loc = pl, g["cluster"], g["location"]
        self._cl = None

    def cluster(self):
        """Cluster dict, or None when it does not exist; raises when gcloud can't tell."""
        if self._cl is None:
            out = gc(["container", "clusters", "list", f"--project={PROJECT}", f"--filter=name={self.name}",
                      "--format=json"])
            if out.returncode:
                raise RuntimeError(out.stderr[-300:])
            cl = json.loads(out.stdout or "[]")
            self._cl = cl[0] if cl else {}
        return self._cl or None

    def api(self, method, path, body=None, ok404=False):
        cl = self.cluster()
        if not cl or cl.get("status") not in ("RUNNING", "RECONCILING"):
            raise RuntimeError(f"cluster {self.name} not running ({cl.get('status') if cl else 'absent'})")
        tok = gc(["auth", "print-access-token"], 60).stdout.strip()
        ctx = ssl.create_default_context(cadata=base64.b64decode(cl["masterAuth"]["clusterCaCertificate"]).decode())
        req = urllib.request.Request(f"https://{cl['endpoint']}{path}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            if e.code == 404 and ok404:
                return None
            raise RuntimeError(f"{method} {path}: HTTP {e.code} {e.read()[:300]!r}") from None

    def tpu_pools(self):
        return [p["name"] for p in (self.cluster() or {}).get("nodePools", []) if p["name"].startswith("tpu-")]

    def job(self):
        return self.api("GET", f"/apis/batch/v1/namespaces/default/jobs/{JOB}", ok404=True)

    def job_state(self, j):
        if not j:
            return "none"
        st = j.get("status", {})
        for c in st.get("conditions", []):
            if c.get("status") == "True" and c.get("type") in ("Complete", "Failed"):
                return c["type"].lower()
        return "running" if st.get("active") else "pending"

    def pods(self):
        r = self.api("GET", f"/api/v1/namespaces/default/pods?labelSelector=job-name%3D{JOB}") or {}
        return [{"name": p["metadata"]["name"], "phase": p["status"].get("phase"), "node": p["spec"].get("nodeName")}
                for p in r.get("items", [])]

    def manifest(self):
        g = self.pl["gke"]
        return {
            "apiVersion": "batch/v1", "kind": "Job",
            "metadata": {"name": JOB, "labels": {"owner": "qrl-bc4a"}},
            "spec": {
                "backoffLimit": 100,  # a preempted Spot node fails the pod; the Job re-creates it (claims resume)
                "template": {
                    "metadata": {"labels": {"owner": "qrl-bc4a"}},
                    "spec": {
                        "restartPolicy": "Never",
                        "activeDeadlineSeconds": int(g.get("pod_max_run_s", 43200)),
                        "terminationGracePeriodSeconds": 25,
                        "nodeSelector": {"cloud.google.com/gke-tpu-accelerator": "tpu-v6e-slice",
                                         "cloud.google.com/gke-tpu-topology": "2x2"},
                        "tolerations": [
                            {"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"},
                            {"key": "qrl-bc4a", "operator": "Exists", "effect": "NoSchedule"},
                            {"key": "cloud.google.com/gke-spot", "operator": "Exists", "effect": "NoSchedule"},
                            {"key": "cloud.google.com/gke-flex-start", "operator": "Exists", "effect": "NoSchedule"},
                        ],
                        "containers": [{
                            "name": "qrl", "image": g.get("image", "vllm/vllm-torchtpu:v0.32.0"),
                            "command": ["bash", "-c", BOOT],
                            "securityContext": {"privileged": True},
                            "resources": {"requests": {"google.com/tpu": 4}, "limits": {"google.com/tpu": 4}},
                            "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
                        }],
                        "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "200Gi"}}],
                    },
                },
            },
        }

    def dry_manifest(self):
        """The work Job on the CPU default pool: same image, BOOT and privileges; no TPU, disk-backed /dev/shm."""
        m = json.loads(json.dumps(self.manifest()))
        m["metadata"]["name"] = DRY_JOB
        m["spec"]["backoffLimit"] = 0
        sp = m["spec"]["template"]["spec"]
        sp["activeDeadlineSeconds"] = 3600
        sp["nodeSelector"] = {"cloud.google.com/gke-nodepool": "default-pool"}
        c = sp["containers"][0]
        c.pop("resources")
        c["env"] = [{"name": "QRL_DRY", "value": "1"}]
        sp["volumes"] = [{"name": "dshm", "emptyDir": {}}]
        return m

    def delete_job(self):
        self.api("DELETE", f"/apis/batch/v1/namespaces/default/jobs/{JOB}",
                 {"kind": "DeleteOptions", "apiVersion": "v1", "propagationPolicy": "Background",
                  "gracePeriodSeconds": 0}, ok404=True)

    def ensure_job(self):
        st = self.job_state(self.job())
        if st in ("complete", "failed"):
            self.delete_job()
            return f"replaced {st} job (re-created next pass)"
        if st == "none":
            self.api("POST", "/apis/batch/v1/namespaces/default/jobs", self.manifest())
            return "created"
        return st

    def resize0(self, pool):
        out = gc(["container", "clusters", "resize", self.name, f"--node-pool={pool}", "--num-nodes=0",
                  f"--location={self.loc}", f"--project={PROJECT}", "--quiet", "--async"], 180)
        return out.returncode == 0 or out.stderr[-200:]

    def tpu_nodes(self):
        out = gc(["compute", "instances", "list", f"--project={PROJECT}", f"--filter=name~^gke-{self.name}-tpu-",
                  "--format=json"])
        if out.returncode:
            raise RuntimeError(out.stderr[-300:])
        return [{"name": v["name"], "zone": v["zone"].rsplit("/", 1)[-1], "status": v["status"],
                 "start": v.get("lastStartTimestamp"),
                 "model": v.get("scheduling", {}).get("provisioningModel", "STANDARD"),
                 "pool": v.get("labels", {}).get("goog-k8s-node-pool-name", "")} for v in json.loads(out.stdout)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("cmd")
    ap.add_argument("arg", nargs="?")
    a = ap.parse_args()
    pl = json.load(open(a.plan))
    g = Gke(pl)
    if a.cmd == "job-yaml":
        print(json.dumps(g.manifest(), indent=1))
        return
    cl = g.cluster()
    if a.cmd == "status":
        st = {"cluster": cl.get("status") if cl else "absent", "pools": g.tpu_pools(), "nodes": g.tpu_nodes()}
        if cl and cl.get("status") == "RUNNING":
            j = g.job()
            st.update(job=g.job_state(j), pods=g.pods())
        print(json.dumps(st))
    elif a.cmd == "ensure-job":
        print(g.ensure_job())
    elif a.cmd == "delete-job":
        if cl:
            g.delete_job()
        print("deleted")
    elif a.cmd == "scale0":
        for p in [a.arg] if a.arg else g.tpu_pools():
            print(p, g.resize0(p))
    elif a.cmd == "kill-node":
        nodes = {n["name"]: n for n in g.tpu_nodes()}
        try:
            g.delete_job()
        except RuntimeError as e:
            print(f"delete-job failed: {e}", file=sys.stderr)
        pool = nodes.get(a.arg, {}).get("pool") or next((p for p in g.tpu_pools() if p in (a.arg or "")), None)
        for p in [pool] if pool else g.tpu_pools():
            print(p, g.resize0(p))
    elif a.cmd == "dry-job":
        g.api("POST", "/apis/batch/v1/namespaces/default/jobs", g.dry_manifest())
        print("created", DRY_JOB)
    elif a.cmd == "delete-dry-job":
        g.api("DELETE", f"/apis/batch/v1/namespaces/default/jobs/{DRY_JOB}",
              {"kind": "DeleteOptions", "apiVersion": "v1", "propagationPolicy": "Background", "gracePeriodSeconds": 0},
              ok404=True)
        print("deleted", DRY_JOB)
    elif a.cmd == "delete-cluster":
        if cl:
            out = gc(["container", "clusters", "delete", g.name, f"--location={g.loc}", f"--project={PROJECT}",
                      "--quiet", "--async"], 180)
            print("delete requested" if out.returncode == 0 else out.stderr[-300:])
        else:
            print("absent")
    else:
        raise SystemExit(f"unknown command {a.cmd}")


if __name__ == "__main__":
    main()
