#!/bin/bash
# Source this file, then call gcp_auth. Activates the service account in GCP_SA_KEY_JSON
# (raw JSON or base64 of it) without echoing, logging, or keeping the key: it is written to a
# 0600 temp file, used once, and deleted (also on exit via gcp_auth_cleanup).
GCP_PROJECT=${GCP_PROJECT:-tpu-builders-509006}

gcp_auth_cleanup() {
  if [ -n "${_GCP_KEY_FILE:-}" ]; then
    rm -f "$_GCP_KEY_FILE"
    _GCP_KEY_FILE=
  fi
}

gcp_auth() {
  case $- in *x*) echo "gcp_auth: refusing to run with xtrace on" >&2; return 1 ;; esac
  if [ -z "${GCP_SA_KEY_JSON:-}" ]; then
    echo "gcp_auth: GCP_SA_KEY_JSON is not set (add it as a Cloud Agents secret)" >&2
    return 1
  fi
  command -v gcloud >/dev/null || { echo "gcp_auth: gcloud not found; run rl/tpu/install_gcloud.sh" >&2; return 1; }
  local old_umask rc=0
  old_umask=$(umask)
  umask 077
  _GCP_KEY_FILE=$(mktemp "${TMPDIR:-/tmp}/gcp-sa.XXXXXXXX")
  umask "$old_umask"
  if printf '%s' "$GCP_SA_KEY_JSON" | head -c 1 | grep -q '{'; then
    printf '%s' "$GCP_SA_KEY_JSON" > "$_GCP_KEY_FILE"
  else
    printf '%s' "$GCP_SA_KEY_JSON" | base64 -d > "$_GCP_KEY_FILE" 2>/dev/null || rc=1
  fi
  if [ $rc -eq 0 ]; then
    gcloud auth activate-service-account --key-file="$_GCP_KEY_FILE" --quiet >/dev/null 2>&1 || rc=1
  fi
  gcp_auth_cleanup
  if [ $rc -ne 0 ]; then
    echo "gcp_auth: service-account activation failed (key not printed)" >&2
    return 1
  fi
  gcloud config set project "$GCP_PROJECT" --quiet >/dev/null 2>&1 || return 1
  echo "gcp_auth: active account $(gcloud config get-value account 2>/dev/null), project $GCP_PROJECT" >&2
}
