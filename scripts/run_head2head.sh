#!/bin/bash
# Preregistered head-to-head (seeds fixed in PREREG.md). Run from build dir.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/run_experiment.py --train-seeds 2101 2102 2103 2104 2105 \
  --test-seeds 3101 3102 3103 3104 3105 \
  --n-train 1200 --n-test 600 --out results/head2head.json --boot-seed 777 --n-boot 2000
