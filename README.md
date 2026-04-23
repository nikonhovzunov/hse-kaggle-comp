# HSE Dota 2 Competition Codebase

Code-only version of the HSE ML Course Competition 2026 Dota 2 solution.

The repository keeps the reusable training and submission pipeline without raw data, generated models, submissions, or personal notes.

## Repository Layout

```text
hse-kaggle-comp/
  README.md
  requirements.txt
  scripts/
    train.py
    make_submission.py
  src/
    hse_dota_comp/
      config.py
      data.py
      metrics.py
      pipeline.py
      submission.py
      features/
        feature_sets.py
        notebook_features.py
  data/
    raw/
  models/
  submissions/
```

## Data

Place Kaggle files into `data/raw/`:

```text
data/raw/
  matches_df_train.csv
  matches_df_test.csv
  player_df.csv
  Constants.Heroes.csv
  dota_adv.csv
  game_chat.csv
```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run

Train:

```powershell
python scripts/train.py
```

Generate submission:

```powershell
python scripts/make_submission.py `
  --data-dir data/raw `
  --submission-path submissions/best_submission.csv `
  --model-path models/logreg_best.joblib
```

Tracked repository contents are limited to source code and launcher scripts. Raw data, models, submissions, caches, and auxiliary docs are ignored.
