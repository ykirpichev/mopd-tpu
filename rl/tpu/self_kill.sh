#!/bin/bash
# Remove this plan90 machine: a Compute VM deletes itself; a GKE pod (GKE=1) deletes its Job and resizes its TPU
# node pool to 0 (deleting a GKE node directly would just make its instance group re-create it).
#   bash self_kill.sh NAME ZONE
NAME=$1 ZONE=$2
if [ "${GKE:-0}" = 1 ] && [ "${QRL_DRY:-0}" = 1 ]; then  # a dry pod borrows a node it doesn't own: never release it
  echo "dry pod: not releasing $NAME"; exit 0
fi
if [ "${GKE:-0}" = 1 ]; then
  timeout 450 python3 /root/rl/tpu/gke90.py --plan /root/plan.json kill-node "$NAME"
else
  timeout 300 gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet
fi
