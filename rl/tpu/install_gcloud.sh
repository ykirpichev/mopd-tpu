#!/bin/bash
# Installs the Google Cloud CLI under $HOME/google-cloud-sdk if gcloud is absent. Makes no GCP API calls.
# Afterwards: export PATH="$HOME/google-cloud-sdk/bin:$PATH"
set -euo pipefail
if command -v gcloud >/dev/null; then
  gcloud --version | head -1
  exit 0
fi
SDK=$HOME/google-cloud-sdk
if [ ! -x "$SDK/bin/gcloud" ]; then
  tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/sdk.tgz" https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
  tar xzf "$tmp/sdk.tgz" -C "$HOME"
  rm -rf "$tmp"
  "$SDK/install.sh" --quiet --usage-reporting=false --path-update=false --command-completion=false >/dev/null
fi
"$SDK/bin/gcloud" config set core/disable_usage_reporting true --quiet >/dev/null 2>&1 || true
"$SDK/bin/gcloud" --version | head -1
echo "export PATH=\"$SDK/bin:\$PATH\""
