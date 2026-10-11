"""Accuracy and macro-F1, recorded once per experiment.

Every experiment reports through `evaluate`, so the metric definitions, the
class order and the confusion-matrix layout are identical across the six
comparisons in the report.
"""

import json

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .config import LABELS, RESULTS_DIR


def evaluate(name, y_true, y_pred, meta=None):
    """Score one experiment on the test set and persist the numbers.

    `meta` records the experimental settings (representation, hyper-parameters,
    vocabulary size, ...) alongside the scores, so the report's tables can be
    rebuilt from `results/` alone.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)

    per_class = f1_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
    record = {
        "experiment": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
        "per_class_f1": {lab: float(v) for lab, v in zip(LABELS, per_class)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "labels": LABELS,
        "n_test": len(y_true),
        "meta": meta or {},
    }

    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"[{name}] acc={record['accuracy']:.4f}  macro-F1={record['macro_f1']:.4f}")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=4, zero_division=0))
    return record
