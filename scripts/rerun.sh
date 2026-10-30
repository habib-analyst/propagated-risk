#!/bin/bash
# Determinism check: regenerate head-to-head into a temp file and compare
# byte-for-byte with the committed results/head2head.json, then print SHA-256.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/run_experiment.py --train-seeds 2101 2102 2103 2104 2105 \
  --test-seeds 3101 3102 3103 3104 3105 \
  --n-train 1200 --n-test 600 --out results/head2head_rerun.json --boot-seed 777 --n-boot 2000
cmp results/head2head.json results/head2head_rerun.json && echo "BYTE-IDENTICAL"
sha256sum results/head2head.json
rm results/head2head_rerun.json
