# SUMMARY — propagated-risk (project #96) — PASS

**Paper:** arXiv 2610.33282, "Estimation of Propagated Risk in
LLM-based Multi-Agent Systems" (Ganguly, Chattopadhyay, Liu, Sarkar,
Iowa State, 23 Oct 2026). **Repo needed:** habib-analyst/propagated-risk
(not created by this builder; no push made).

**What was built:** a CPU-only, $0, deterministic implementation of
IE-Prop's formal structure (local score + sender-state-mixed
interaction effects + recursive propagation, trained by BCE) on
scripted 3-agent arithmetic traces. Interaction types (adopt /
recompute / override) are recovered from observable trace texts only.

**Preregistered head-to-head (PREREG.md; train seeds 2101–2105,
test seeds 3101–3105; 1200 train / 600 test traces per seed;
paired bootstrap CIs, 2000 resamples, seed 777). Final-node AUROC,
mean over 5 seeds:**

| Model | AUROC |
|---|---|
| Receiver-Only | 0.6596 |
| Receiver-Text | 0.6597 |
| Free Learner | 0.8125 |
| Recursion Control | 0.8231 |
| **IE-Prop** | **0.8597** |

**Differences (IE-Prop minus baseline, mean; all positive on 5/5
seeds; paired 95% CI excludes 0 on 5/5 seeds for every comparison):**
IE−Receiver-Only +0.2001; IE−Receiver-Text +0.2000;
IE−Free Learner +0.0472; IE−Recursion Control +0.0366.

**Mechanism:** fitted adopt effects are opposite-signed on all
seeds (d0 ≈ −1.6 when the sender is correct, d1 ≈ +3.1 when wrong):
copying a wrong answer propagates risk, copying a right answer
protects — the paper's propagation-vs-correction structure, learned
from traces, not hard-coded.

**Benign utility:** mean predicted risk on all-correct chains
0.21–0.28 (bound 0.40); deferring the top-20% riskiest final nodes
raises retained accuracy by +0.1171 over random deferral.

**Checks:** pytest 6 passed; determinism rerun byte-identical;
SHA-256(results/head2head.json) =
d8e9e7e61727f489162b37c781d2aaf6f635fbf44c87e25a2666c527e967274a.

**Scope honesty:** these are scripted-analog effect sizes, larger
than the paper's LLM-trace effects by construction. They show the
IE-Prop mechanism is exhibitable and measurable at CPU/scripted
scale; they are NOT a reproduction of the paper's +0.05/+0.03 AUROC
claims on LLM traces. Compute spent: $0, CPU only (~35 min total
across pilot, head-to-head, params, tests, rerun).
