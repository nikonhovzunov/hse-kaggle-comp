"""Submission helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_submission(
    matches_test: pd.DataFrame,
    probabilities: np.ndarray,
    output_path: str | Path,
) -> pd.DataFrame:
    """Create and save a Kaggle submission with columns ID and Value."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission = matches_test[["match_id"]].copy()
    submission["Value"] = probabilities
    submission = submission.rename(columns={"match_id": "ID"})
    submission = submission[["ID", "Value"]]
    submission.to_csv(output_path, index=False)
    return submission
