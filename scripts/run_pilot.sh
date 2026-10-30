#!/bin/bash
# Pilot gate run (seeds fixed in PILOT.md). Must be run from the build dir.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/run_experiment.py --train-seeds 11 12 13 --test-seeds 101 102 103 \
  --n-train 1200 --n-test 600 --out results/pilot.json --boot-seed 777 --n-boot 2000
