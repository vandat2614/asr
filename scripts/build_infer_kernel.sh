#!/usr/bin/env bash
# Concatenate the production backend and a Kaggle driver into one kernel file.
# Usage: scripts/build_infer_kernel.sh [driver name without .py, default infer_variants_driver]
set -euo pipefail
driver="${1:-infer_variants_driver}"
cat src/asr/local.py "scripts/${driver}.py" > outputs/kaggle_infer/infer_asr_kaggle.py
echo "built outputs/kaggle_infer/infer_asr_kaggle.py from ${driver}"
