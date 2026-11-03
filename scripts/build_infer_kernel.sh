#!/usr/bin/env bash
# Concatenate the production backend and the Kaggle driver into one kernel file.
set -euo pipefail
cat src/asr/local.py scripts/infer_variants_driver.py > outputs/kaggle_infer/infer_asr_kaggle.py
echo "built outputs/kaggle_infer/infer_asr_kaggle.py"
