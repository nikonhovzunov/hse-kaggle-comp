# HSE ML Course Competition 2026 - Dota 2

GitHub-ready version of my best Kaggle solution for the **HSE ML Course Competition 2026 - Dota 2**.

The original solution was developed in a notebook. This repository keeps the core feature logic close to the best submission and wraps it into a cleaner Python project with reusable modules, a command-line submission script, and no raw data committed to Git.

## Project Card

| Field | Value |
| --- | --- |
| Project | HSE ML Course Competition 2026 - Dota 2 |
| Platform | Kaggle |
| Kaggle nickname | NikonHV |
| Result | 8th place out of 444 |
| Public/private score | 0.41971 |
| Target | `radiant_win` |
| Metric | Gini = `2 * ROC-AUC - 1` |
| Main model | Logistic Regression |
| Main signal | Dota-specific tabular features + chat TF-IDF |

## Task

Predict whether the Radiant team wins a Dota 2 match.

The submission format is:

```text
ID,Value
...
```

where `ID` is `match_id` and `Value` is the predicted probability of Radiant victory.

## Solution

The final pipeline combines structured match features, hero-based features, historical advantage features, and game chat text features.

Feature groups:

- match metadata: lobby/game mode, region and time features;
- hero picks and team composition features;
- hero roles, attributes and aggregated hero statistics;
- historical gold/xp advantage aggregates and trend features;
- hero win-rate and hero pair synergy features;
- chat statistics and TF-IDF features for Radiant/Dire chat;
- second-order interactions for a selected high-signal feature block.

Final estimator:

```python
LogisticRegression(max_iter=12000, random_state=42)
```

The model itself is simple; most of the performance comes from feature engineering and building a large sparse feature matrix.

## Repository Structure

```text
hse-kaggle-comp/
  README.md
  requirements.txt
  data/
    README.md
    raw/
  scripts/
    make_submission.py
    train.py
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
  models/
  submissions/
  docs/
    project_card.md
    resume_ru.md
```

## Data

Raw Kaggle data is not included.

Place the files into `data/raw/`:

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

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The pipeline uses NLTK stopwords. They are downloaded automatically on the first run if missing.

## Generate Submission

PowerShell:

```powershell
python scripts/make_submission.py `
  --data-dir data/raw `
  --submission-path submissions/best_submission.csv `
  --model-path models/logreg_best.joblib
```

To skip saving the model, add `--no-save-model`.

## Reproducibility Notes

The implementation follows the best confirmed notebook:

```text
best_submit_ever_0_41971.ipynb
```

The notebook-derived feature engineering code is kept in:

```text
src/hse_dota_comp/features/notebook_features.py
```

The cleaner orchestration code is kept in:

```text
src/hse_dota_comp/pipeline.py
```

## Resume

Russian resume wording is available in:

```text
docs/resume_ru.md
```
