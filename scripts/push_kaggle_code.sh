#!/usr/bin/env bash
# Copy the code needed by the Kaggle notebook and push it as a dataset version.
# Usage: scripts/push_kaggle_code.sh "message"
set -euo pipefail
cd "$(dirname "$0")/.."

dest=outputs/kaggle_code
mkdir -p "$dest"
cp src/asr/normalize.py src/asr/metrics.py scripts/train_asr.py "$dest"/

if [ ! -f "$dest/dataset-metadata.json" ]; then
  cat > "$dest/dataset-metadata.json" <<'META'
{
  "title": "Zalo ASR Code",
  "id": "vandat2601/zalo-asr-code",
  "licenses": [{"name": "other"}]
}
META
  kaggle datasets create -p "$dest"
else
  kaggle datasets version -p "$dest" -m "${1:-update code}"
fi
