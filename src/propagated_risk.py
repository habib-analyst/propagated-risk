"""Propagated Risk (IE-Prop analog) — scripted, CPU-only, $0.

Scope (stated honestly everywhere): this is NOT the paper's LLM system.
The paper (arXiv 2610.33282) estimates error risk from real LLM
multi-agent execution traces (Qwen3 0.6B/1.7B/4B, GSM8K/MedQA/WebQA,
4x A100). Here, agents are deterministic scripted arithmetic solvers
whose visible outputs are short text traces. The analog reproduces the
paper's *formal structure* exactly — Eq. (1) local score, Eq. (6)
sender-state mixing, Eq. (7) recursive propagation, Eq. (8) BCE over
node labels — on traces where the interaction type (adopt / recompute /
override) is recoverable from the observable texts alone, never from
ground-truth labels. Effect sizes measured here are properties of this
scripted analog; they are NOT claims about LLM traces and must never
be quoted as reproductions of the paper's +0.05 / +0.03 AUROC figures.

Generative process (fixed before any pilot run; see PILOT.md):
  chain of 3 agents solving a 3-operand arithmetic problem, gold = a + b*c
  (a, b, c drawn per problem). Root: correct w.p. 0.70, else a small slip.
  Non-root behaviour (sampled per edge, observable in the texts):
    adopt     (0.40): copy the sender's answer verbatim
    recompute (0.35): independent recomputation, correct w.p. 0.85
    override  (0.25): replace with a new answer, correct w.p. 0.20
  Local signal: a "confidence" scalar, correct ~ N(0.62, 0.18),
  incorrect ~ N(0.52, 0.18), clipped to [0.05, 0.95]. Deliberately weak
  (Receiver-Only AUROC ~ 0.65) so interaction signal has headroom.
  Receiver-Text features (receiver-only, no sender info): local logit
  plus problem magnitude (log1p|gold|), which is independent of
  correctness by construction.

Models (all trained by deterministic full-batch Adam on BCE):
  receiver_only:   sigmoid(w . [1, local_logit])
  receiver_text:   sigmoid(w . [1, local_logit, magnitude])
  free_learner:    sigmoid(b_v + delta[type]) — one unconstrained
                   adjustment per interaction type, no sender-state
                   mixing, no parent-risk input (paper's Free Learner).
  recursion_ctrl:  IE-Prop form but parent risk replaced by the
                   parent's LOCAL risk sigma(b_p) (paper's Recursion
                   Control), so no recursively accumulated risk flows.
  ieprop:          sigmoid(b_v + sum_p (1-r_p) d0[type] + r_p d1[type])
                   with r_p the recursively estimated parent risk
                   (paper Eq. 6-7), trained end-to-end by backprop
                   through the recursion (manual reverse-mode below).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TYPES = ("adopt", "recompute", "override")
TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}
TYPE_PROBS = np.array([0.40, 0.35, 0.25])
ROOT_P_CORRECT = 0.70
RECOMPUTE_P_CORRECT = 0.85
OVERRIDE_P_CORRECT = 0.20


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


@dataclass
class Node:
    text: str
    answer: int
    correct: bool  # 1 = error indicator E_v in the paper (we store error)
    local_logit: float
    magnitude: float

    @property
    def error(self) -> int:
        return 0 if self.correct else 1


@dataclass
class Trace:
    problem: str
    gold: int
    nodes: list  # list[Node], chain order
    types: list  # list[str], types[i] is edge into node i (None for root)


def generate_traces(seed: int, n: int) -> list:
    """Deterministic scripted traces. Interaction type is encoded ONLY
    in the observable texts (verb + answer relation), never in labels."""
    rng = np.random.default_rng(seed)
    traces = []
    for _ in range(n):
        a = int(rng.integers(2, 30))
        b = int(rng.integers(2, 30))
        c = int(rng.integers(2, 30))
        gold = a + b * c
        problem = f"What is {a} + {b} x {c}?"
        nodes, types = [], [None]
        prev_answer = None
        for i in range(3):
            if i == 0:
                correct = bool(rng.random() < ROOT_P_CORRECT)
                ans = gold if correct else gold + int(rng.choice([-3, -2, -1, 1, 2, 3, 10]))
                text = f"My answer: {ans}."
                typ = None
            else:
                k = int(rng.choice(3, p=TYPE_PROBS))
                typ = TYPES[k]
                if typ == "adopt":
                    ans = prev_answer
                    correct = ans == gold
                    text = f"I agree with the previous agent: {ans}."
                elif typ == "recompute":
                    correct = bool(rng.random() < RECOMPUTE_P_CORRECT)
                    ans = gold if correct else gold + int(rng.choice([-3, -2, -1, 1, 2, 3, 10]))
                    text = f"Recomputing independently: {ans}."
                else:  # override
                    correct = bool(rng.random() < OVERRIDE_P_CORRECT)
                    ans = gold if correct else gold + int(rng.choice([-3, -2, -1, 1, 2, 3, 10]))
                    text = f"I disagree; instead: {ans}."
                types.append(typ)
            mu = 0.62 if correct else 0.52
            conf = float(np.clip(rng.normal(mu, 0.18), 0.05, 0.95))
            nodes.append(Node(text=text, answer=ans, correct=correct,
                              local_logit=_logit(conf),
                              magnitude=math.log1p(abs(gold))))
            prev_answer = ans
        traces.append(Trace(problem=problem, gold=gold, nodes=nodes, types=types))
    return traces


def extract_type(sender_text: str, receiver_text: str) -> str:
    """Recover the interaction type from observable texts only.
    Uses the receiver's verb; asserts the answer relation is consistent
    (adopt copies, recompute/override need not copy). No label access."""
    if receiver_text.startswith("I agree"):
        return "adopt"
    if receiver_text.startswith("Recomputing"):
        return "recompute"
    if receiver_text.startswith("I disagree"):
        return "override"
    raise ValueError(f"unrecognised receiver text: {receiver_text!r}")


def auroc(scores, labels) -> float:
    """Rank-based AUROC (Mann-Whitney). scores: risk (higher = more error)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank with ties averaged
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    s_sorted = scores[order]
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def _adam(params, grads, state, lr=0.05, epochs=1):
    # full-batch Adam, deterministic (no randomness anywhere)
    for _ in range(epochs):
        for k in params:
            g = grads[k]
            state[k]["m"] = 0.9 * state[k]["m"] + 0.1 * g
            state[k]["v"] = 0.999 * state[k]["v"] + 0.001 * (g * g)
            state[k]["t"] += 1
            m = state[k]["m"] / (1 - 0.9 ** state[k]["t"])
            v = state[k]["v"] / (1 - 0.999 ** state[k]["t"])
            params[k] = params[k] - lr * m / (np.sqrt(v) + 1e-8)
    return params


def _fit_logistic(X, y, epochs=400, lr=0.05):
    w = np.zeros(X.shape[1])
    state = {"w": {"m": np.zeros_like(w), "v": np.zeros_like(w), "t": 0}}
    for _ in range(epochs):
        p = _sigmoid(X @ w)
        grad = X.T @ (p - y) / len(y) + 1e-4 * w
        w = _adam({"w": w}, {"w": grad}, state, lr=lr, epochs=1)["w"]
    return w


def _node_features(traces, use_magnitude: bool):
    """Per-trace arrays: local features for every node, final-node rows."""
    rows, ys, mags = [], [], []
    for tr in traces:
        for nd in tr.nodes:
            feat = [1.0, nd.local_logit]
            if use_magnitude:
                feat.append(nd.magnitude)
            rows.append(feat)
            ys.append(nd.error)
    return np.array(rows), np.array(ys)


def _standardise(train_X, test_X):
    mu = train_X[:, 1:].mean(axis=0)
    sd = train_X[:, 1:].std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    tr = train_X.copy(); te = test_X.copy()
    tr[:, 1:] = (train_X[:, 1:] - mu) / sd
    te[:, 1:] = (test_X[:, 1:] - mu) / sd
    return tr, te


def fit_models(train_traces):
    """Fit all five models. Returns dict kind -> params + standardiser."""
    X_loc, y = _node_features(train_traces, use_magnitude=False)
    X_txt, _ = _node_features(train_traces, use_magnitude=True)
    X_loc_s, _ = _standardise(X_loc, X_loc)
    # standardiser from local features (columns 1..)
    mu_loc = X_loc[:, 1:].mean(axis=0); sd_loc = X_loc[:, 1:].std(axis=0)
    mu_txt = X_txt[:, 1:].mean(axis=0); sd_txt = X_txt[:, 1:].std(axis=0)
    sd_loc = np.where(sd_loc == 0, 1.0, sd_loc)
    sd_txt = np.where(sd_txt == 0, 1.0, sd_txt)

    def std(X, mu, sd):
        Z = X.copy(); Z[:, 1:] = (X[:, 1:] - mu) / sd; return Z

    w_ro = _fit_logistic(std(X_loc, mu_loc, sd_loc), y)
    w_rt = _fit_logistic(std(X_txt, mu_txt, sd_txt), y)

    # Edge models: joint training over all nodes with manual reverse-mode
    # through the chain recursion. Parameters: w (local, [1, local_logit]
    # standardised), d0/d1 per type (ieprop, recursion_ctrl) or delta per
    # type (free). Root nodes have no edge term.
    def train_edge(mode):
        w = np.zeros(2)
        d0 = np.zeros(3); d1 = np.zeros(3); delta = np.zeros(3)
        params = {"w": w, "d0": d0, "d1": d1, "delta": delta}
        state = {k: {"m": np.zeros_like(v), "v": np.zeros_like(v), "t": 0}
                 for k, v in params.items()}
        data = []
        for tr in train_traces:
            feats = []
            for nd in tr.nodes:
                feats.append(np.array([1.0, (nd.local_logit - mu_loc[0]) / sd_loc[0]]))
            data.append((tr, feats))
        for _ in range(400):
            grads = {k: np.zeros_like(v) for k, v in params.items()}
            for tr, feats in data:
                b = [float(f @ params["w"]) for f in feats]
                r, eta = [], []
                for i, nd in enumerate(tr.nodes):
                    if i == 0:
                        e = b[0]
                    else:
                        t = TYPE_INDEX[tr.types[i]]
                        if mode == "free":
                            e = b[i] + params["delta"][t]
                        elif mode == "recursion_ctrl":
                            rp = _sigmoid(b[i - 1])
                            e = b[i] + (1 - rp) * params["d0"][t] + rp * params["d1"][t]
                        else:  # ieprop
                            rp = r[i - 1]
                            e = b[i] + (1 - rp) * params["d0"][t] + rp * params["d1"][t]
                    eta.append(e); r.append(float(_sigmoid(e)))
                # reverse-mode: g_eta[i] = dL/d eta_i
                g_eta = [0.0] * 3
                for i in (2, 1, 0):
                    g = r[i] - tr.nodes[i].error
                    if i < 2:
                        t = TYPE_INDEX[tr.types[i + 1]]
                        if mode == "free":
                            pass  # no parent-risk path
                        elif mode == "recursion_ctrl":
                            pass  # parent LOCAL risk: no path through r[i]
                        else:
                            g += g_eta[i + 1] * (params["d1"][t] - params["d0"][t]) \
                                * r[i] * (1 - r[i])
                    g_eta[i] = g
                for i, nd in enumerate(tr.nodes):
                    grads["w"] += g_eta[i] * feats[i]
                    if i > 0:
                        t = TYPE_INDEX[tr.types[i]]
                        if mode == "free":
                            grads["delta"][t] += g_eta[i]
                        elif mode == "recursion_ctrl":
                            # Parent LOCAL risk sigma(b_{i-1}) depends on w
                            # (through b, not through eta), so its gradient
                            # is a direct w-term, not a g_eta accumulation.
                            rp = _sigmoid(b[i - 1])
                            grads["d0"][t] += g_eta[i] * (1 - rp)
                            grads["d1"][t] += g_eta[i] * rp
                            grads["w"] += (g_eta[i] * (params["d1"][t] - params["d0"][t])
                                           * rp * (1 - rp)) * feats[i - 1]
                        else:
                            rp = r[i - 1]
                            grads["d0"][t] += g_eta[i] * (1 - rp)
                            grads["d1"][t] += g_eta[i] * rp
            n = sum(len(tr.nodes) for tr, _ in data)
            for k in grads:
                grads[k] = grads[k] / n + 1e-4 * params[k]
            params = _adam(params, grads, state, lr=0.05, epochs=1)
        return params

    edge = {m: train_edge(m) for m in ("ieprop", "free", "recursion_ctrl")}
    return {"w_ro": w_ro, "w_rt": w_rt,
            "mu_loc": mu_loc, "sd_loc": sd_loc,
            "mu_txt": mu_txt, "sd_txt": sd_txt, "edge": edge}


def predict_final(models, traces):
    """Final-node (node 2) predicted risks per model + labels."""
    out = {k: [] for k in ("receiver_only", "receiver_text", "free_learner",
                           "recursion_ctrl", "ieprop")}
    labels, all_correct = [], []
    for tr in traces:
        labels.append(tr.nodes[2].error)
        all_correct.append(all(nd.correct for nd in tr.nodes))
        f_loc = np.array([1.0, (tr.nodes[2].local_logit - models["mu_loc"][0]) / models["sd_loc"][0]])
        f_txt = np.array([1.0,
                          (tr.nodes[2].local_logit - models["mu_txt"][0]) / models["sd_txt"][0],
                          (tr.nodes[2].magnitude - models["mu_txt"][1]) / models["sd_txt"][1]])
        out["receiver_only"].append(float(_sigmoid(f_loc @ models["w_ro"])))
        out["receiver_text"].append(float(_sigmoid(f_txt @ models["w_rt"])))
        # recursive predictions for edge models
        for kind, key in (("ieprop", "ieprop"), ("free", "free_learner"),
                          ("recursion_ctrl", "recursion_ctrl")):
            p = models["edge"][kind]
            b = [float(np.array([1.0, (nd.local_logit - models["mu_loc"][0]) / models["sd_loc"][0]]) @ p["w"])
                 for nd in tr.nodes]
            r = []
            for i, nd in enumerate(tr.nodes):
                if i == 0:
                    r.append(float(_sigmoid(b[0])))
                else:
                    t = TYPE_INDEX[tr.types[i]]
                    if kind == "free":
                        r.append(float(_sigmoid(b[i] + p["delta"][t])))
                    elif kind == "recursion_ctrl":
                        rp = float(_sigmoid(b[i - 1]))
                        r.append(float(_sigmoid(b[i] + (1 - rp) * p["d0"][t] + rp * p["d1"][t])))
                    else:
                        rp = r[i - 1]
                        r.append(float(_sigmoid(b[i] + (1 - rp) * p["d0"][t] + rp * p["d1"][t])))
            out[key].append(r[2])
    return out, np.array(labels), np.array(all_correct)


def paired_bootstrap_ci(a_scores, b_scores, labels, seed, n_boot=2000):
    """Paired bootstrap CI for AUROC(a) - AUROC(b), resampling indices."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a_scores); b = np.asarray(b_scores); y = np.asarray(labels)
    base = auroc(a, y) - auroc(b, y)
    diffs = []
    idx_all = np.arange(len(y))
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(auroc(a[idx], y[idx]) - auroc(b[idx], y[idx]))
    diffs = np.array(diffs)
    return base, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
