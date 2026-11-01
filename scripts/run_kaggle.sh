#!/usr/bin/env bash
# Push code and an experiment config to Kaggle, then start the training kernel.
# Usage: scripts/run_kaggle.sh configs/<experiments>.json ["message"]
set -euo pipefail
cd "$(dirname "$0")/.."

config=${1:?usage: run_kaggle.sh configs/<file>.json [message]}
msg=${2:-"run $(basename "$config")"}
owner=vandat2601
code_dir=outputs/kaggle_code
kernel_dir=outputs/kaggle_kernel
mkdir -p "$code_dir" "$kernel_dir"

cp src/asr/normalize.py src/asr/metrics.py scripts/train_asr.py "$code_dir"/
cp "$config" "$code_dir"/experiments.json
cp scripts/kaggle_run.py "$kernel_dir"/kaggle_run.py

cat > "$kernel_dir"/kernel-metadata.json <<META
{
  "id": "$owner/zalo-asr-train",
  "title": "Zalo ASR Train",
  "code_file": "kaggle_run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "dataset_sources": ["$owner/zalo-lifemate-asr", "$owner/zalo-asr-code"]
}
META

kaggle datasets version -p "$code_dir" -m "$msg"
sleep 20
for _ in $(seq 1 30); do
  status=$(kaggle datasets status "$owner/zalo-asr-code" 2>&1 || true)
  echo "code dataset status: $status"
  case "$status" in *ready*) break ;; esac
  sleep 10
done

kaggle kernels push -p "$kernel_dir"
echo "started. Check: kaggle kernels status $owner/zalo-asr-train"
