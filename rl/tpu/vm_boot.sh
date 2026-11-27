#!/bin/bash
# Metadata startup-script for plan90 requests: fetch the current vm_startup.sh from GCS and run it, so fixes
# reach requests that were created earlier. Deletes the VM if it can't, rather than idle unguarded.
# The plan's GCS prefix comes from instance metadata qrl-q (driver90 sets it; plan90 when absent).
export PATH=$PATH:/snap/bin:/usr/lib/google-cloud-sdk/bin:/usr/local/bin
Q=$(curl -sf -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/attributes/qrl-q)
Q=${Q:-gs://qrl-bc4a-compile-cache-tpu-builders-509006/plan90}
for i in $(seq 1 20); do
  timeout 120 gcloud storage cp $Q/vm_startup.sh /root/vm_startup.sh --quiet \
    && exec bash /root/vm_startup.sh
  sleep 30
done
MD=http://metadata.google.internal/computeMetadata/v1/instance
gcloud compute instances delete "$(curl -s -H Metadata-Flavor:Google $MD/name)" \
  --zone "$(curl -s -H Metadata-Flavor:Google $MD/zone | awk -F/ '{print $NF}')" --quiet
