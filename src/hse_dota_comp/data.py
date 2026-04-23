"""Data loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hse_dota_comp.config import REQUIRED_DATA_FILES


@dataclass(frozen=True)
class CompetitionData:
    matches_train: pd.DataFrame
    matches_test: pd.DataFrame
    players: pd.DataFrame
    heroes: pd.DataFrame
    advantages: pd.DataFrame
    chat: pd.DataFrame


def expected_file_table() -> str:
    lines = ["Expected files:"]
    for name, filename in REQUIRED_DATA_FILES.items():
        lines.append(f"- {name}: {filename}")
    return "\n".join(lines)


def validate_data_dir(data_dir: Path) -> None:
    missing = [
        filename
        for filename in REQUIRED_DATA_FILES.values()
        if not (data_dir / filename).exists()
    ]
    if missing:
        formatted = "\n".join(f"- {name}" for name in missing)
        raise FileNotFoundError(
            f"Missing competition files in {data_dir}:\n{formatted}\n\n"
            f"{expected_file_table()}"
        )


def load_competition_data(data_dir: str | Path) -> CompetitionData:
    data_dir = Path(data_dir)
    validate_data_dir(data_dir)

    return CompetitionData(
        matches_train=pd.read_csv(data_dir / REQUIRED_DATA_FILES["matches_train"]),
        matches_test=pd.read_csv(data_dir / REQUIRED_DATA_FILES["matches_test"]),
        players=pd.read_csv(data_dir / REQUIRED_DATA_FILES["players"]),
        heroes=pd.read_csv(data_dir / REQUIRED_DATA_FILES["heroes"]),
        advantages=pd.read_csv(data_dir / REQUIRED_DATA_FILES["advantages"]),
        chat=pd.read_csv(data_dir / REQUIRED_DATA_FILES["chat"]),
    )
