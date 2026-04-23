# Формулировка для резюме

**HSE ML Course Competition 2026 (Dota 2)** — 8 место из 444, Kaggle, предсказание исхода матча.

- Построил модель для предсказания исхода матча по составам команд, динамике золота и опыта и сигналам из игровых чатов.
- При ограничении на линейные модели вынес нелинейность в признаки: добавил полиномиальные взаимодействия и объединил игровые и текстовые сигналы в единое признаковое пространство.

Расширенная версия для GitHub / интервью:

- Реализовал feature engineering: признаки пиков героев, team composition, hero stats, hero win-rate, hero pair synergy, агрегаты и тренды по gold/xp advantage, chat statistics и TF-IDF по Radiant/Dire chat.
- Собрал итоговую sparse matrix из структурных и текстовых признаков; обучил Logistic Regression (`max_iter=12000`) и получил подтвержденный submission score `0.41971`.

Короткая версия:

- HSE ML Course Competition 2026 (Dota 2): 8/444; sparse ML pipeline с Dota-specific feature engineering, chat TF-IDF и Logistic Regression для предсказания `radiant_win`.
