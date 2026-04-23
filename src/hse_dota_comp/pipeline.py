"""End-to-end pipeline for the best Dota 2 Kaggle submission."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import nltk
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler

from hse_dota_comp.config import BEST_FEATURE_BUILD_KWARGS, BEST_MODEL_PARAMS
from hse_dota_comp.data import CompetitionData, load_competition_data
from hse_dota_comp.features.feature_sets import OVERKILL_INTERACTION_FEATURES
from hse_dota_comp.features.notebook_features import DatasetBuilder, MultiFeatureBuilder
from hse_dota_comp.submission import make_submission


@dataclass
class FeatureMatrices:
    X_train: sparse.csr_matrix
    y_train: np.ndarray
    X_test: sparse.csr_matrix
    base_feature_count: int
    interaction_feature_count: int
    chat_feature_count: int


@dataclass
class PipelineRunResult:
    submission_path: Path
    model_path: Path | None
    train_shape: tuple[int, int]
    test_shape: tuple[int, int]
    base_feature_count: int
    interaction_feature_count: int
    chat_feature_count: int


def ensure_nltk_resources() -> None:
    """Download the only NLTK resource required by the original notebook."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def _validate_interaction_features(columns: list[str]) -> None:
    missing = [col for col in OVERKILL_INTERACTION_FEATURES if col not in columns]
    if missing:
        formatted = "\n".join(f"- {col}" for col in missing)
        raise ValueError(
            "The feature builder did not produce all interaction columns used "
            f"by the best notebook pipeline:\n{formatted}"
        )


def build_best_submission_features(
    data: CompetitionData,
    verbose: bool = True,
) -> FeatureMatrices:
    """Build the same structured + chat sparse matrix used in the best notebook."""
    ensure_nltk_resources()

    builder = DatasetBuilder(
        matches_train=data.matches_train,
        matches_test=data.matches_test,
        players_info=data.players,
        heroes_info=data.heroes,
        advantages_info=data.advantages,
        game_chat=data.chat,
    )

    X_train_df, y_train, X_test_df, X_train_chat, X_test_chat = builder.build_train_test(
        **BEST_FEATURE_BUILD_KWARGS
    )

    _validate_interaction_features(X_train_df.columns.to_list())

    interaction_cols = OVERKILL_INTERACTION_FEATURES
    base_cols = [col for col in X_train_df.columns if col not in set(interaction_cols)]

    X_train_base = X_train_df[base_cols].astype(np.float32)
    X_test_base = X_test_df[base_cols].astype(np.float32)

    interaction_builder = MultiFeatureBuilder(
        interaction_cols=interaction_cols,
        other_cols=[],
        use_inter_scaler=True,
        use_poly=True,
        degree=2,
        interaction_only=True,
        include_bias=False,
        use_svd=False,
        verbose=verbose,
    )

    X_train_interactions = interaction_builder.fit_transform(X_train_df[interaction_cols])
    X_test_interactions = interaction_builder.transform(X_test_df[interaction_cols])

    X_train_base_sp = sparse.csr_matrix(X_train_base.to_numpy(dtype=np.float32))
    X_test_base_sp = sparse.csr_matrix(X_test_base.to_numpy(dtype=np.float32))

    X_train_struct = sparse.hstack(
        [X_train_base_sp, X_train_interactions],
        format="csr",
    )
    X_test_struct = sparse.hstack(
        [X_test_base_sp, X_test_interactions],
        format="csr",
    )

    final_scaler = MaxAbsScaler()
    X_train_struct = final_scaler.fit_transform(X_train_struct)
    X_test_struct = final_scaler.transform(X_test_struct)

    X_train_final = sparse.hstack([X_train_struct, X_train_chat], format="csr")
    X_test_final = sparse.hstack([X_test_struct, X_test_chat], format="csr")

    if verbose:
        print(f"Structured train matrix: {X_train_struct.shape}")
        print(f"Chat train matrix:       {X_train_chat.shape}")
        print(f"Final train matrix:      {X_train_final.shape}")
        print(f"Final test matrix:       {X_test_final.shape}")

    return FeatureMatrices(
        X_train=X_train_final,
        y_train=y_train,
        X_test=X_test_final,
        base_feature_count=X_train_base_sp.shape[1],
        interaction_feature_count=X_train_interactions.shape[1],
        chat_feature_count=X_train_chat.shape[1],
    )


def train_best_model(features: FeatureMatrices) -> LogisticRegression:
    """Train the final LogisticRegression model from the best notebook."""
    model = LogisticRegression(**BEST_MODEL_PARAMS)
    model.fit(features.X_train, features.y_train)
    return model


def run_best_submission_pipeline(
    data_dir: str | Path,
    submission_path: str | Path,
    model_path: str | Path | None = None,
    verbose: bool = True,
) -> PipelineRunResult:
    """Load data, build features, train model, and write Kaggle submission."""
    data_dir = Path(data_dir)
    submission_path = Path(submission_path)
    model_path = Path(model_path) if model_path is not None else None

    data = load_competition_data(data_dir)
    features = build_best_submission_features(data, verbose=verbose)
    model = train_best_model(features)
    probabilities = model.predict_proba(features.X_test)[:, 1]

    make_submission(data.matches_test, probabilities, submission_path)

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    return PipelineRunResult(
        submission_path=submission_path,
        model_path=model_path,
        train_shape=features.X_train.shape,
        test_shape=features.X_test.shape,
        base_feature_count=features.base_feature_count,
        interaction_feature_count=features.interaction_feature_count,
        chat_feature_count=features.chat_feature_count,
    )
