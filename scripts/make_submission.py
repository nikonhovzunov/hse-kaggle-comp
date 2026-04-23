"""Generate the Kaggle submission for the best Dota 2 pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory with raw Kaggle CSV files.",
    )
    parser.add_argument(
        "--submission-path",
        type=Path,
        default=PROJECT_ROOT / "submissions" / "best_submission.csv",
        help="Where to write the Kaggle submission.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "logreg_best.joblib",
        help="Where to save the trained LogisticRegression model.",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Do not save the trained LogisticRegression model.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable feature-shape logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from hse_dota_comp.pipeline import run_best_submission_pipeline

    model_path = None if args.no_save_model else args.model_path
    result = run_best_submission_pipeline(
        data_dir=args.data_dir,
        submission_path=args.submission_path,
        model_path=model_path,
        verbose=not args.quiet,
    )

    print(f"Submission saved to: {result.submission_path}")
    print(f"Train matrix shape:  {result.train_shape}")
    print(f"Test matrix shape:   {result.test_shape}")
    if result.model_path is not None:
        print(f"Model saved to:      {result.model_path}")


if __name__ == "__main__":
    main()
