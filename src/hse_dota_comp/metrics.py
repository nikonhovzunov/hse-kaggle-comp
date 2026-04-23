"""Competition metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return the normalized Gini used by the competition."""
    return 2.0 * roc_auc_score(y_true, y_score) - 1.0
