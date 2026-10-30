# PREREGISTRATION — propagated-risk head-to-head
# Locked AFTER the pilot (results/pilot.json) and BEFORE any head-to-head run.
# No bar, seed, or analysis choice below may change after head-to-head
# numbers are seen. Pilot seeds (11-13 / 101-103) are never reused here.

## Design (unchanged from pilot)
- Scripted 3-agent chain analog, src/propagated_risk.py (frozen).
- Train 1200 traces / test 600 traces per seed; final-node AUROC.
- Head-to-head seeds: train 2101, 2102, 2103, 2104, 2105;
  test 3101, 3102, 3103, 3104, 3105. Bootstrap seed 777, 2000 resamples.
- Models: receiver_only, receiver_text, free_learner,
  recursion_ctrl, ieprop (definitions in src docstring).

## Pilot numbers that motivate (not guarantee) these bars
Pilot means: receiver_only 0.6673, receiver_text 0.6651,
free_learner 0.8251, recursion_ctrl 0.8358, ieprop 0.8640.
Mean differences: IE−RO +0.1967, IE−RT +0.1989, IE−Free +0.0389,
IE−RecCtrl +0.0282. Bars below are set WELL below pilot means so a
pass requires a real, repeatable effect, not a pilot-matched fluke —
and well above the paper's small-effect scale (+0.05) for the headline
comparison, because this analog's generative effects are stronger by
construction (stated scope limit, not a claim about LLM traces).

## G2 — disease bars (ALL must hold on the mean over 5 seeds)
- D1: mean IE-Prop AUROC − Receiver-Only AUROC >= 0.04
- D2: mean IE-Prop AUROC − Receiver-Text AUROC >= 0.02
- D3: mean IE-Prop AUROC − Free Learner AUROC >= 0.01
- D4: each difference positive on ALL 5 seeds (no sign flip)
- D5: paired bootstrap 95% CI for IE-Prop − Receiver-Only excludes 0
  on at least 4 of 5 seeds

## G3 — load-bearing mechanism + benign-utility bound (ALL must hold)
- M1 (state conditioning is load-bearing): D3 already requires
  beating the Free Learner; additionally the fitted IE-Prop deltas
  must show opposite-signed or clearly separated d0/d1 for at least
  one interaction type (reported from the fitted params, not tuned).
- M2 (recursion is load-bearing): mean IE-Prop − Recursion Control
  AUROC >= 0.01, positive on all 5 seeds, and paired bootstrap 95%
  CI excludes 0 on at least 3 of 5 seeds.
- B1 (benign bound): mean predicted IE-Prop risk on all-correct
  chains <= 0.40 on every seed (no risk inflation on clean traces).
- B2 (deferral utility): deferring the top-20% riskiest final nodes
  must raise retained accuracy vs random deferral by >= 0.03 on the
  5-seed mean (retained accuracy = fraction correct among kept nodes).

## Decision rule
- ALL G2 + G3 bars pass → PASS (staged payload, PAPER.md, DRAFT ONLY
  LinkedIn draft, REPO NEEDED report).
- ANY bar fails → KILL on that bar (honest KILLED.md + negative
  PAPER.md, no repo, no draft). No post-hoc bar changes, no
  seed dropping, no subset selection.
