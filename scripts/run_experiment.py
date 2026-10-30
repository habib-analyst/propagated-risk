"""Run one experiment: train on train-seed traces, evaluate on test-seed traces.
Usage: python3 scripts/run_experiment.py --train-seeds 11 12 13 --test-seeds 101 102 103
       --n-train 1200 --n-test 600 --out results/pilot.json --boot-seed 777
Deterministic: all randomness flows from the seed arguments only.
"""
import argparse, json, sys
sys.path.insert(0, "src")
import numpy as np
from propagated_risk import (generate_traces, fit_models, predict_final,
                             auroc, paired_bootstrap_ci)

MODELS = ["receiver_only", "receiver_text", "free_learner",
          "recursion_ctrl", "ieprop"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seeds", type=int, nargs="+", required=True)
    ap.add_argument("--test-seeds", type=int, nargs="+", required=True)
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot-seed", type=int, default=777)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    assert len(args.train_seeds) == len(args.test_seeds)
    per_seed, all_scores, all_labels = [], {}, []
    for ts, es in zip(args.train_seeds, args.test_seeds):
        train = generate_traces(ts, args.n_train)
        test = generate_traces(es, args.n_test)
        models = fit_models(train)
        preds, labels, all_correct = predict_final(models, test)
        row = {"train_seed": ts, "test_seed": es}
        for k in MODELS:
            row[k] = auroc(preds[k], labels)
        # benign-utility diagnostics (final node)
        ie = np.asarray(preds["ieprop"])
        row["benign_mean_risk_all_correct"] = float(
            ie[all_correct].mean()) if all_correct.any() else float("nan")
        row["n_all_correct"] = int(all_correct.sum())
        # deferral: drop top-20% riskiest final nodes, retained accuracy
        order = np.argsort(-ie)  # descending risk
        keep = np.sort(order[int(0.2 * len(ie)):])  # drop top-20% riskiest
        acc_retained = float((labels[keep] == 0).mean())
        rng = np.random.default_rng(4242)
        rand_keep = np.sort(rng.choice(len(ie), size=len(keep), replace=False))
        acc_random = float((labels[rand_keep] == 0).mean())
        row["deferral_retained_acc"] = acc_retained
        row["deferral_random_acc"] = acc_random
        row["final_error_rate"] = float(labels.mean())
        row["paired_ci_ieprop_minus"] = {}
        for base in ("receiver_only", "receiver_text", "free_learner",
                     "recursion_ctrl"):
            d, lo, hi = paired_bootstrap_ci(
                preds["ieprop"], preds[base], labels,
                seed=args.boot_seed + es, n_boot=args.n_boot)
            row["paired_ci_ieprop_minus"][base] = [d, lo, hi]
        per_seed.append(row)
        for k in MODELS:
            all_scores.setdefault(k, []).append(preds[k])
        all_labels.append(labels)
    summary = {}
    for k in MODELS:
        summary[k] = float(np.mean([r[k] for r in per_seed]))
    out = {"per_seed": per_seed, "mean_auroc": summary,
           "train_seeds": args.train_seeds, "test_seeds": args.test_seeds,
           "n_train": args.n_train, "n_test": args.n_test}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
