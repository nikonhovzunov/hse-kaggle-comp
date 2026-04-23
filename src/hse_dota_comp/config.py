"""Project configuration for the best public Kaggle submission."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompetitionResult:
    competition_name: str = "HSE ML Course Competition 2026 - Dota 2"
    kaggle_username: str = "NikonHV"
    rank: str = "8th place out of 444"
    score: float = 0.41971
    metric: str = "Gini = 2 * ROC-AUC - 1"


REQUIRED_DATA_FILES = {
    "matches_train": "matches_df_train.csv",
    "matches_test": "matches_df_test.csv",
    "players": "player_df.csv",
    "heroes": "Constants.Heroes.csv",
    "advantages": "dota_adv.csv",
    "chat": "game_chat.csv",
}


BEST_FEATURE_BUILD_KWARGS = {
    "use_base": True,
    "use_game_mode": True,
    "use_time": True,
    "use_region": True,
    "use_heroes": True,
    "use_team_comp": True,
    "use_hero_stat_std": True,
    "use_hero_stat_mean": True,
    "use_advantages_agg": True,
    "use_hero_match_stat": True,
    "use_advantages_trend": True,
    "use_hero_winrate": True,
    "adv_trend_method": "ols",
    "use_players": False,
    "use_players_diff": False,
    "use_adv_bins": False,
    "adv_bin_cols": None,
    "adv_n_bins": 3,
    "adv_bin_strategy": "uniform",
    "adv_bins_keep_original": True,
    "use_chat_stats": True,
    "use_chat_slang": True,
    "return_chat_sparse": True,
    "chat_max_features": 60000,
    "chat_min_df": 5,
    "chat_ngram_range": (1, 2),
    "use_chat_char": False,
    "chat_char_max_features": 3000,
    "chat_char_min_df": 30,
    "chat_char_ngram_range": (2, 5),
    "use_hero_pair_synergy": True,
    "target_col": "radiant_win",
}


BEST_MODEL_PARAMS = {
    "max_iter": 12000,
    "random_state": 42,
}
