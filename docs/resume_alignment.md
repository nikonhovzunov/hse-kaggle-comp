# Resume Alignment

This file records the exact claims from the current resume PDF and where they are represented in the project.

## Resume Claim

**HSE ML Course Competition 2026 (Dota 2)** — 8 место из 444  
Kaggle | предсказание исхода матча

Competition URL: https://www.kaggle.com/competitions/dota-2-hse-ml-1-course-competition-2026

- Построил модель для предсказания исхода матча по составам команд, динамике золота и опыта и сигналам из игровых чатов.
- При ограничении на линейные модели вынес нелинейность в признаки: добавил полиномиальные взаимодействия и объединил игровые и текстовые сигналы в единое признаковое пространство.

## Project Coverage

- Rank/platform: `README.md`, `docs/project_card.md`, `docs/resume_ru.md`.
- Match outcome target: `src/hse_dota_comp/config.py`, `src/hse_dota_comp/submission.py`.
- Team composition and hero features: `src/hse_dota_comp/features/notebook_features.py`.
- Gold/xp advantage features: `src/hse_dota_comp/features/notebook_features.py`.
- Game chat features: `src/hse_dota_comp/features/notebook_features.py`.
- Polynomial interactions: `src/hse_dota_comp/features/notebook_features.py`, `src/hse_dota_comp/features/feature_sets.py`.
- Unified sparse feature matrix: `src/hse_dota_comp/pipeline.py`.
- Linear model: `src/hse_dota_comp/pipeline.py`, `src/hse_dota_comp/config.py`.

Note: the repository also records the confirmed score `0.41971`, which is not shown in the current resume line but is consistent with the project evidence and earlier confirmed competition result.
