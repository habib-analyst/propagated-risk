# propagated-risk — Interaction-Effect Propagation on scripted traces

A CPU-only, $0, deterministic re-implementation of the *formal
structure* of IE-Prop from arXiv 2610.33282 ("Estimation of Propagated
Risk in LLM-based Multi-Agent Systems", Ganguly et al., Iowa State,
23 Oct 2026), evaluated on scripted multi-agent traces — NOT on LLM
traces. Read the scope warning first: numbers here describe this
scripted analog only and are never reproductions of the paper's
figures (+~0.05 AUROC over baseline; +0.03 from multi-step tracking,
both on Qwen3 LLM traces over GSM8K/MedQA/WebQA).

## What the paper claims (from the live abstract, 2026-10-30)
Post-hoc, agent-level error-risk estimation from completed
multi-agent execution traces. For each sender→receiver interaction,
IE-Prop learns separate signed effects for a correct vs an incorrect
sender, mixes them with the sender's estimated error risk (Eq. 6),
adds the result to receiver-local evidence, and propagates recursively
through the execution graph (Eq. 7). Trained end-to-end by BCE over
node correctness labels (Eq. 8). Baselines: Receiver-Only (local
uncertainty), Receiver-Text (receiver text, no sender info), Free
Learner (unconstrained interaction adjustment), PropUQ-MAS (monotone
noisy-OR propagation), plus a Recursion Control (parent risk replaced
by parent local risk).

## What this repo does
Three scripted agents solve `a + b × c` problems in a chain.
Each non-root agent visibly adopts, recomputes, or overrides its
sender's answer; the interaction type is recovered from the observable
texts alone (verb + answer relation), never from labels. Five
estimators are fitted by deterministic full-batch Adam on BCE:
Receiver-Only, Receiver-Text, Free Learner, Recursion Control,
IE-Prop. Final-node AUROC is the headline metric, with paired
bootstrap CIs (2000 resamples) for IE-Prop minus each baseline.

## Results
See SUMMARY.md for the preregistered head-to-head numbers and
KILLGATE.md for per-gate verdicts. Headline (head-to-head, fresh
seeds 2101–2105/3101–3105): IE-Prop final-node AUROC 0.8597 vs
Receiver-Only 0.6596, Receiver-Text 0.6597, Free Learner 0.8125,
Recursion Control 0.8231 — all differences positive on 5/5 seeds
with paired bootstrap 95% CIs excluding 0. Scripted-analog numbers
only; see SUMMARY.md for the scope warning.

## Reproduce
```bash
python3 -m pytest tests/ -q
bash scripts/run_head2head.sh   # regenerates results/head2head.json
bash scripts/rerun.sh           # determinism check: byte-identical?
```
Requirements: Python 3.12 + numpy only. No network, no GPU, no paid APIs.

## Determinism
`results/head2head.json` SHA-256 digest is recorded in SUMMARY.md,
KILLGATE.md and PAPER.md. Re-running `run_head2head.sh` must
reproduce byte-identical JSON (same Python/numpy); the rerun script
verifies the digest.

## Files
- `src/propagated_risk.py` — generator, models, AUROC, paired bootstrap
- `scripts/run_experiment.py` — train/evaluate driver
- `scripts/run_pilot.sh`, `scripts/run_head2head.sh`, `scripts/rerun.sh`
- `PILOT.md` — pilot gate criteria (fixed before pilot numbers)
- `PREREG.md` — preregistered head-to-head bars and fresh seeds
- `KILLGATE.md` — per-gate verdicts with exact numbers
- `SUMMARY.md` — final numbers
- `results/` — pilot.json, head2head.json
