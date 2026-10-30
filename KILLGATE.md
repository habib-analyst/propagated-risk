# KILLGATE — propagated-risk (project #96, arXiv 2610.33282)

Verdict: **PASS — all gates pass.** Staged payload only; no repo
created, no push (REPO NEEDED: habib-analyst/propagated-risk, for the
main agent via the browser flow).

## G0 — harness
Deterministic, CPU-only, $0, no paid APIs, no real systems/accounts/
credentials. Scripted 3-agent arithmetic-chain analog only; scope
stated honestly in README.md, PAPER.md, and the code docstring.
**PASS.**

## G1 — author-code recheck (2026-10-30, live)
- arXiv abs page (2610.33282v1, submitted 23 Oct 2026): no code link
  — only generic CatalyzeX/DagsHub/alphaXiv toggles, no repository.
- Full HTML text: zero occurrences of "github".
- Web searches: exact title, author names (Ganguly, Chattopadhyay,
  Liu, Sarkar), "IE-Prop", arXiv ID — returned only arXiv/paperz
  mirrors, no repository. Paper is 7 days old; no author code found.
- #95 lesson applied: the paper's own code link was checked FIRST
  (abs page + full text), not just a web search.
**PASS (no author code; build proceeds).**

## Pilot gate (PILOT.md, fixed before running; seeds 11-13/101-103)
Mean AUROC: IE-Prop 0.8640, Receiver-Only 0.6673, Receiver-Text
0.6651, Free Learner 0.8251, Recursion Control 0.8358.
- P1: IE−RO +0.1967 >= 0.04 ✓
- P2: IE−RT +0.1989 >= 0.02 ✓
- P3: IE beats RO on 3/3 seeds ✓
- P4: IE−RO paired bootstrap 95% CI excludes 0 on 3/3 seeds ✓
**PASS.** (One pre-pilot bug fix, disclosed: the deferral diagnostic
kept the riskiest 80% instead of dropping the top 20%; fixed before
the full pilot run; AUROC numbers unaffected.)

## G2 — disease (PREREG.md, fresh seeds 2101-2105/3101-3105)
Mean final-node AUROC over 5 seeds (600 test traces each):
Receiver-Only 0.6596, Receiver-Text 0.6597, Free Learner 0.8125,
Recursion Control 0.8231, IE-Prop 0.8597.
- D1: IE−RO +0.2001 >= 0.04 ✓
- D2: IE−RT +0.2000 >= 0.02 ✓
- D3: IE−Free +0.0472 >= 0.01 ✓
- D4: every difference positive on all 5 seeds ✓
- D5: IE−RO paired bootstrap 95% CI excludes 0 on 5/5 seeds ✓
  (per-seed IE−RO CIs: [0.1615,0.2476], [0.1388,0.2238],
  [0.1474,0.2316], [0.1665,0.2552], [0.1705,0.2630])
**PASS.**

## G3 — load-bearing mechanism + benign-utility bound
- M1: fitted IE-Prop deltas (results/fitted_params.json, all 5 seeds):
  adopt d0 ≈ −1.48…−1.76 vs d1 ≈ +2.91…+3.32 — opposite-signed,
  separated by ~4.7 log-odds. Copying a correct sender lowers
  receiver error log-odds; copying an incorrect sender raises it.
  Recompute d0≈d1≈−0.9 (sender-independent, as designed); override
  both ≈ +2. State conditioning is load-bearing, not decorative. ✓
- M2: IE−Recursion Control +0.0366 >= 0.01, positive on 5/5 seeds,
  paired bootstrap 95% CI excludes 0 on 5/5 seeds (per-seed:
  [0.0243,0.0653], [0.0178,0.0519], [0.0204,0.0505], [0.0220,0.0661],
  [0.0044,0.0446]). Recursive propagation adds signal beyond local
  parent risk + immediate interaction. ✓
- B1: mean predicted risk on all-correct chains 0.2124–0.2797 per
  seed, all <= 0.40. No risk inflation on clean traces. ✓
- B2: deferring top-20% riskiest final nodes: retained accuracy
  beats random deferral by +0.1171 on the 5-seed mean (>= 0.03). ✓
**PASS.**

## Tests + determinism
- pytest: 6 passed (tests use seeds 5/6 only, disjoint from pilot
  and head-to-head seeds).
- Determinism: `bash scripts/rerun.sh` regenerates
  results/head2head.json byte-identical.
- SHA-256(results/head2head.json):
  d8e9e7e61727f489162b37c781d2aaf6f635fbf44c87e25a2666c527e967274a

## Scope limit (must travel with every number)
All effect sizes above are properties of the scripted analog, whose
generative effects are deliberately stronger than the paper's LLM
setting. They do NOT reproduce or confirm the paper's +0.05 / +0.03
AUROC figures, and must never be quoted as doing so.
