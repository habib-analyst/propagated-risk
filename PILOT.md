# PILOT GATE — propagated-risk (written BEFORE any pilot run)

Paper: arXiv 2610.33282, "Estimation of Propagated Risk in LLM-based
Multi-Agent Systems", Ganguly, Chattopadhyay, Liu, Sarkar (Iowa State),
submitted 23 Oct 2026. Claims (abstract, read live 2026-10-30):
IE-Prop, an interaction-conditioned model learning separate effects for
correct vs incorrect upstream agents, combined with the sender's
estimated error risk and propagated recursively through the execution
graph; +~0.05 final-node AUROC over the established baseline on average
across 3 architectures x 3 benchmarks; +0.03 AUROC from multi-step risk
tracking in sequential systems. Evaluation uses real LLM traces
(Qwen3 0.6B/1.7B/4B, GSM8K/MedQA/WebQA, 4x A100).

Watch cautions carried in: effects are SMALL (+0.05, +0.03) and the
work needs LLM execution traces. This pilot must therefore prove the
central effect is exhibitable and measurable with real separation at
CPU/scripted scale. If not, KILL on scope — no bar-moving, no
seed/subset cherry-picking, no shrinking bars.

## Pilot design (fixed here, before running)
- Scripted 3-agent chain analog (see src/propagated_risk.py docstring).
- Pilot seeds: 11, 12, 13. Train 1200 traces / test 600 traces per seed.
- Metric: final-node AUROC, mean over the 3 pilot seeds.
- Paired bootstrap CI (2000 resamples, seed 777) on the pooled
  per-seed IE-Prop minus Receiver-Only difference is reported; the
  kill/pass decision uses the mean difference and the per-seed sign.

## Pilot PASS criteria (all must hold; otherwise KILL on scope)
- P1: mean IE-Prop AUROC − Receiver-Only AUROC >= 0.04
- P2: mean IE-Prop AUROC − Receiver-Text AUROC >= 0.02
- P3: IE-Prop beats Receiver-Only on ALL 3 pilot seeds (no sign flip)
- P4: paired bootstrap 95% CI for IE-Prop − Receiver-Only (per-seed,
  reported for each seed) excludes 0 on at least 2 of 3 seeds

Recursion gain (IE-Prop − Recursion Control) is REPORTED at pilot
but is not a pilot kill criterion; it becomes a preregistered G3 bar
for the head-to-head. Pilot seeds (11–13) are never reused for the
head-to-head; head-to-head seeds are preregistered in PREREG.md.
