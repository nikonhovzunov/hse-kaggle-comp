# Project Card

| Field | Value |
| --- | --- |
| Project | HSE ML Course Competition 2026 - Dota 2 |
| Platform | Kaggle |
| Competition URL | https://www.kaggle.com/competitions/dota-2-hse-ml-1-course-competition-2026 |
| Kaggle nickname | NikonHV |
| Result | 8th place out of 444 |
| Public/private score | 0.41971 |
| Target | `radiant_win` |
| Metric | Gini = `2 * ROC-AUC - 1` |
| Main model | Logistic Regression |
| Key techniques | Dota-specific feature engineering, chat TF-IDF, sparse matrix pipeline |

## Short Summary

Predicting Dota 2 match outcome from match metadata, hero information, gold/xp advantage history and in-game chat. The final pipeline uses a large sparse feature matrix and a simple Logistic Regression model.

## Reproducible Entry Point

```powershell
python scripts/make_submission.py `
  --data-dir data/raw `
  --submission-path submissions/best_submission.csv `
  --model-path models/logreg_best.joblib
```
