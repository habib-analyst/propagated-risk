"""Tests for the propagated-risk analog. Test seeds (5, 6) are disjoint
from pilot seeds (11-13/101-103) and preregistered head-to-head seeds
(2101-2105/3101-3105); no reported metric uses test seeds."""
import sys
sys.path.insert(0, "src")
import numpy as np
from propagated_risk import (generate_traces, extract_type, auroc,
                             fit_models, predict_final)


def test_determinism():
    a = generate_traces(5, 50)
    b = generate_traces(5, 50)
    assert [(t.problem, [n.text for n in t.nodes]) for t in a] == \
           [(t.problem, [n.text for n in t.nodes]) for t in b]


def test_extract_type_matches_generated():
    for tr in generate_traces(5, 200):
        for i in range(1, 3):
            assert extract_type(tr.nodes[i - 1].text, tr.nodes[i].text) == tr.types[i]


def test_adopt_copies_sender_answer():
    for tr in generate_traces(6, 300):
        for i in range(1, 3):
            if tr.types[i] == "adopt":
                assert tr.nodes[i].answer == tr.nodes[i - 1].answer
                assert tr.nodes[i].correct == tr.nodes[i - 1].correct


def test_auroc_known_values():
    assert auroc([0.1, 0.9], [0, 1]) == 1.0
    assert auroc([0.9, 0.1], [0, 1]) == 0.0
    assert auroc([0.5, 0.5], [0, 1]) == 0.5


def test_fit_and_predict_shapes_and_range():
    train = generate_traces(5, 120)
    test = generate_traces(6, 60)
    models = fit_models(train)
    preds, labels, all_correct = predict_final(models, test)
    assert len(labels) == 60
    for k, v in preds.items():
        assert len(v) == 60
        assert all(0.0 <= float(x) <= 1.0 for x in v)
    # interaction signal must beat chance on this (generative) data
    assert auroc(preds["ieprop"], labels) > 0.60


def test_ieprop_beats_receiver_only_small():
    train = generate_traces(5, 300)
    test = generate_traces(6, 200)
    models = fit_models(train)
    preds, labels, _ = predict_final(models, test)
    assert auroc(preds["ieprop"], labels) > auroc(preds["receiver_only"], labels)
