"""Logistic Regression with the regularisation strength chosen on validation.

The classifier is held fixed across every comparison in the report - only the
document representation changes.  `C` is the one degree of freedom, and it is
selected on the validation split (never on test) so that all six experiments
get an equally fair shot at their representation.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = (0.1, 1.0, 10.0, 100.0)


def fit_lr(X_train, y_train, X_val, y_val, labels, scale=False, class_weight=None):
    """Fit LR for each C in the grid, keep the best on validation macro-F1.

    `scale` standardises the features first, which the dense 100-d document
    vectors need for L-BFGS to converge well; the sparse BoW matrices are
    left alone.

    Returns (best_estimator, val_macro_f1, chosen_C).
    """
    best = None
    for C in C_GRID:
        steps = [StandardScaler()] if scale else []
        steps.append(LogisticRegression(C=C, max_iter=3000, solver="lbfgs", class_weight=class_weight))
        model = make_pipeline(*steps)
        model.fit(X_train, y_train)

        score = f1_score(y_val, model.predict(X_val), average="macro", labels=labels, zero_division=0)
        print(f"    C={C:<6g} val macro-F1={score:.4f}")
        if best is None or score > best[1]:
            best = (model, score, C)

    return best
