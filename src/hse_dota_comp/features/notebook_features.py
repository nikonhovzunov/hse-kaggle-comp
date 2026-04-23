"""Feature engineering copied from the best Kaggle notebook.

This module preserves the feature-generation logic used in
`best_submit_ever_0_41971.ipynb` and is intentionally kept close to
the original solution. The cleaner orchestration lives in `pipeline.py`.
"""

from __future__ import annotations

import ast
import gc
import inspect
import re
from collections import namedtuple
from itertools import combinations
from pathlib import Path

import category_encoders as ce
import nltk
import numpy as np
import pandas as pd
import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MaxAbsScaler,
    MultiLabelBinarizer,
    PolynomialFeatures,
    StandardScaler,
)

if not hasattr(inspect, 'getargspec'):
    ArgSpec = namedtuple('ArgSpec', ['args', 'varargs', 'keywords', 'defaults'])

    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec

class MultiFeatureBuilder:
    def __init__(
        self,
        interaction_cols: list[str],
        other_cols: list[str] | None = None,
        use_inter_scaler: bool = True,
        use_poly: bool = True,
        degree: int = 2,
        interaction_only: bool = True,
        include_bias: bool = False,
        use_svd: bool = False,
        n_components: int = 32,
        svd_algorithm: str = 'randomized',
        svd_n_iter: int = 3,
        scale_svd: bool = False,
        random_state: int = 42,
        verbose: bool = False
    ):
        self.interaction_cols = list(interaction_cols)
        self.other_cols = other_cols

        self.use_inter_scaler = use_inter_scaler
        self.use_poly = use_poly
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

        self.use_svd = use_svd
        self.n_components = n_components
        self.svd_algorithm = svd_algorithm
        self.svd_n_iter = svd_n_iter
        self.scale_svd = scale_svd
        self.random_state = random_state
        self.verbose = verbose

        self.inter_scaler = MaxAbsScaler() if self.use_inter_scaler else None

        self.poly = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        ) if self.use_poly else None

        self.svd = TruncatedSVD(
            n_components=self.n_components,
            algorithm=self.svd_algorithm,
            n_iter=self.svd_n_iter,
            random_state=self.random_state
        ) if self.use_svd else None

        self.svd_scaler = StandardScaler() if self.scale_svd else None

        self.other_cols_ = None

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _validate_columns(self, X: pd.DataFrame):
        missing_interaction_cols = [col for col in self.interaction_cols if col not in X.columns]

        if self.other_cols is not None:
            missing_other_cols = [col for col in self.other_cols if col not in X.columns]
           

    def _get_other_cols(self, X: pd.DataFrame):
        if self.other_cols is not None:
            return list(self.other_cols)

        return [col for col in X.columns if col not in self.interaction_cols]

    def _split_feature_blocks(self, X: pd.DataFrame):
        X_inter = X[self.interaction_cols].astype(np.float32)
        X_other = X[self.other_cols_].astype(np.float32) if len(self.other_cols_) > 0 else pd.DataFrame(index=X.index)
        
        return X_inter, X_other

    def _prepare_inter_block_for_fit(self, X_inter: pd.DataFrame):
        X_inter = sparse.csr_matrix(X_inter.to_numpy(dtype=np.float32))
        self._log(f'Interaction block initial shape: {X_inter.shape}')

        if self.inter_scaler is not None:
            X_inter = self.inter_scaler.fit_transform(X_inter)
            self._log(f'After MaxAbsScaler: {X_inter.shape}')

        if self.poly is not None:
            X_inter = self.poly.fit_transform(X_inter)
            self._log(f'After PolynomialFeatures: {X_inter.shape}')

        if self.svd is not None:
            X_inter = self.svd.fit_transform(X_inter)
            X_inter = np.asarray(X_inter, dtype=np.float32)
            self._log(f'After TruncatedSVD: {X_inter.shape}')
            self._log(f'Explained variance sum: {self.svd.explained_variance_ratio_.sum():.6f}')

            if self.svd_scaler is not None:
                X_inter = self.svd_scaler.fit_transform(X_inter)
                X_inter = np.asarray(X_inter, dtype=np.float32)
                self._log(f'After StandardScaler on SVD block: {X_inter.shape}')

        return X_inter

    def _prepare_inter_block_for_transform(self, X_inter: pd.DataFrame):
        X_inter = sparse.csr_matrix(X_inter.to_numpy(dtype=np.float32))
        self._log(f'Interaction block initial shape: {X_inter.shape}')

        if self.inter_scaler is not None:
            X_inter = self.inter_scaler.transform(X_inter)
            self._log(f'After MaxAbsScaler: {X_inter.shape}')

        if self.poly is not None:
            X_inter = self.poly.transform(X_inter)
            self._log(f'After PolynomialFeatures: {X_inter.shape}')

        if self.svd is not None:
            X_inter = self.svd.transform(X_inter)
            X_inter = np.asarray(X_inter, dtype=np.float32)
            self._log(f'After TruncatedSVD: {X_inter.shape}')

            if self.svd_scaler is not None:
                X_inter = self.svd_scaler.transform(X_inter)
                X_inter = np.asarray(X_inter, dtype=np.float32)
                self._log(f'After StandardScaler on SVD block: {X_inter.shape}')

        return X_inter

    def _merge_feature_blocks(self, X_other: pd.DataFrame, X_inter):
        if X_other.shape[1] == 0:
            X_final = X_inter
        else:
            X_other_np = X_other.to_numpy(dtype=np.float32)
            if sparse.issparse(X_inter):
                X_other_sp = sparse.csr_matrix(X_other_np)
                X_final = sparse.hstack([X_other_sp, X_inter], format='csr')
            else:
                X_final = np.hstack([X_other_np, np.asarray(X_inter, dtype=np.float32)])
        self._log(f'Final shape: {X_final.shape}')
        return X_final

    def fit_transform(self, X: pd.DataFrame):
        self._validate_columns(X)
        self.other_cols_ = self._get_other_cols(X)

        X_inter, X_other = self._split_feature_blocks(X)
        X_inter = self._prepare_inter_block_for_fit(X_inter)
        X_final = self._merge_feature_blocks(X_other, X_inter)

        return X_final

    def transform(self, X: pd.DataFrame):

        self._validate_columns(X)

        X_inter = X[self.interaction_cols].astype(np.float32)
        X_other = X[self.other_cols_].astype(np.float32)

        X_inter = self._prepare_inter_block_for_transform(X_inter)
        X_final = self._merge_feature_blocks(X_other, X_inter)

        return X_final

    def build_train_test(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        X_train_final = self.fit_transform(X_train)
        X_test_final = self.transform(X_test)

        return X_train_final, X_test_final

class DatasetBuilder:
    def __init__(
        self,
        matches_train: pd.DataFrame,
        matches_test: pd.DataFrame,
        players_info: pd.DataFrame,
        heroes_info: pd.DataFrame,
        advantages_info: pd.DataFrame,
        game_chat: pd.DataFrame | None = None

    ):
        self.matches_train = self._preprocess_matches(matches_train)
        self.matches_test = self._preprocess_matches(matches_test)
        self.radiant_slots = [i for i in range(0, 5)]
        self.dire_slots = [i for i in range(128, 133)]

        players_info = self._preprocess_players_info(players_info)
        self.players_info = players_info

        self.heroes_info = self._preprocess_heroes(heroes_info)
        self.advantages_info = self._preprocess_advantages_info(advantages_info)

        self.adv_bin_cols_default = [
            'gold_last_clean',
            'xp_last_clean',
            'gold_last3_mean_clean',
            'xp_last3_mean_clean',
            'gold_mean_clean',
            'xp_mean_clean',
            'gold_ahead_frac',
            'xp_ahead_frac',
            'gold_last3_minus_first3',
            'xp_last3_minus_first3',
            'gold_trend_slope_ols',
            'xp_trend_slope_ols'
        ]
        
        self.stop_words = set(stopwords.words('russian')) | set(stopwords.words('english'))
        self.morph = pymorphy2.MorphAnalyzer()

        self.chat_slang_dict = {
            'gg': ['gg'],
            'gl': ['gl'],
            'hf': ['hf'],
            'wp': ['wp'],
            'ggwp': ['ggwp'],
            'glhf': ['glhf'],
        
            'afk': ['afk'],
            'ez': ['ez', 'изи'],
            'noob': ['noob', 'nub', 'нуб'],
            'report': ['report', 'репорт'],
        
            'feed': ['feed', 'feeding', 'feeder', 'фид', 'фижу'],
            'gank': ['gank', 'ganking', 'gang', 'ганг'],
            'ward': ['ward', 'wards', 'sentry', 'sentries', 'warding', 'вард', 'варды'],
            'smoke': ['smoke', 'smoked', 'смок'],
            'rosh': ['rosh', 'roshan', 'рош', 'рошан'],
            'tp': ['tp', 'teleport', 'тп'],
        
            'push': ['push', 'pushing', 'пуш'],
            'split_push': ['split push', 'splitpush', 'rat'],
            'def': ['def', 'defend', 'деф'],
            'backdoor': ['backdoor', 'bd'],
        
            'stack': ['stack', 'stacking', 'стак'],
            'pull': ['pull', 'pulling', 'пул'],
            'farm': ['farm', 'farming', 'фарм'],
        
            'carry': ['carry', 'керри'],
            'support': ['support', 'supp', 'саппорт', 'сап'],
            'mid': ['mid', 'мид'],
            'bot': ['bot', 'бот'],
            'top': ['top', 'топ']
        }
        
        self.game_chat = self._preprocess_game_chat(game_chat) if game_chat is not None else None

    def _preprocessing_text(self, text: str):
        text = str(text).lower()
        text = re.sub(r'(.)\1{2,}', r'\1', text)
    
        tokens = wordpunct_tokenize(text)
        tokens = [token for token in tokens if re.fullmatch(r'[a-zа-яё]+', token)]
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.morph.parse(token)[0].normal_form for token in tokens]
    
        return ' '.join(tokens)

    def _normalize_chat_raw(self, text: str):
        text = str(text).lower()
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _build_chat_team_stats(self, s: pd.Series, prefix: str):
        s = s.fillna('').astype(str)
    
        out = pd.DataFrame(index=s.index)
    
        out[f'{prefix}_has_chat'] = (s.str.len() > 0).astype(np.float32)
        out[f'{prefix}_char_cnt'] = s.str.len().astype(np.float32)
        out[f'{prefix}_word_cnt'] = s.str.count(r'(?u)\b\w+\b').astype(np.float32)
    
        return out
    
    def _preprocess_game_chat(self, df):
        df = df.copy()
    
        all_match_ids = np.union1d(
            self.matches_train['match_id'].unique(),
            self.matches_test['match_id'].unique()
        )
    
        df = df[df['match_id'].isin(all_match_ids)].copy()
        df = df.drop_duplicates(subset='match_id').reset_index(drop=True)
    
        df['radiant_chat_raw'] = df['radiant_chat'].fillna('').apply(self._normalize_chat_raw)
        df['dire_chat_raw'] = df['dire_chat'].fillna('').apply(self._normalize_chat_raw)
    
        df['radiant_chat'] = df['radiant_chat_raw'].apply(self._preprocessing_text)
        df['dire_chat'] = df['dire_chat_raw'].apply(self._preprocessing_text)
    
        return df[['match_id', 'radiant_chat', 'dire_chat', 'radiant_chat_raw', 'dire_chat_raw']]
    
    def _make_chat_sparse_features(
        self,
        chat_max_features=None,
        chat_min_df=5,
        chat_ngram_range=(1, 2),
        use_chat_char=True,
        chat_char_max_features=30000,
        chat_char_min_df=10,
        chat_char_ngram_range=(3, 5)
    ):
        if self.game_chat is None:
            raise ValueError('game_chat is None')
    
        train_chat = self.matches_train[['match_id']].merge(self.game_chat, on='match_id', how='left')
        test_chat = self.matches_test[['match_id']].merge(self.game_chat, on='match_id', how='left')
    
        train_chat['radiant_chat_raw'] = train_chat['radiant_chat_raw'].fillna('')
        train_chat['dire_chat_raw'] = train_chat['dire_chat_raw'].fillna('')
        test_chat['radiant_chat_raw'] = test_chat['radiant_chat_raw'].fillna('')
        test_chat['dire_chat_raw'] = test_chat['dire_chat_raw'].fillna('')
    
        word_vectorizer = TfidfVectorizer(
            max_features=chat_max_features,
            min_df=chat_min_df,
            ngram_range=chat_ngram_range,
            analyzer='word',
            lowercase=False,
            sublinear_tf=True,
            dtype=np.float32
        )
    
        corpus = pd.concat([train_chat['radiant_chat_raw'], train_chat['dire_chat_raw']], ignore_index=True)
    
        word_vectorizer.fit(corpus)
    
        X_train_word_rad = word_vectorizer.transform(train_chat['radiant_chat_raw'])
        X_train_word_dire = word_vectorizer.transform(train_chat['dire_chat_raw'])
        X_test_word_rad = word_vectorizer.transform(test_chat['radiant_chat_raw'])
        X_test_word_dire = word_vectorizer.transform(test_chat['dire_chat_raw'])
    
        blocks_train = [X_train_word_rad, X_train_word_dire]
        blocks_test = [X_test_word_rad, X_test_word_dire]
    
        if use_chat_char:
            char_vectorizer = TfidfVectorizer(
                max_features=chat_char_max_features,
                min_df=chat_char_min_df,
                ngram_range=chat_char_ngram_range,
                analyzer='char_wb',
                lowercase=False,
                sublinear_tf=True,
                dtype=np.float32
            )
    
            char_vectorizer.fit(corpus)
    
            X_train_char_rad = char_vectorizer.transform(train_chat['radiant_chat_raw'])
            X_train_char_dire = char_vectorizer.transform(train_chat['dire_chat_raw'])
            X_test_char_rad = char_vectorizer.transform(test_chat['radiant_chat_raw'])
            X_test_char_dire = char_vectorizer.transform(test_chat['dire_chat_raw'])
    
            blocks_train.extend([X_train_char_rad, X_train_char_dire])
            blocks_test.extend([X_test_char_rad, X_test_char_dire])
    
        X_train_chat = sparse.hstack(blocks_train, format='csr', dtype=np.float32)
        X_test_chat = sparse.hstack(blocks_test, format='csr', dtype=np.float32)
    
        return X_train_chat, X_test_chat

    def _build_chat_slang_team_stats(self, s: pd.Series, team: str):
        s = s.fillna('').astype(str)
    
        out = pd.DataFrame(index=s.index)
    
        for feature_name, variants in self.chat_slang_dict.items():
            cnt = pd.Series(0, index=s.index, dtype=np.float32)
    
            for variant in variants:
                pattern = rf'(?<!\w){re.escape(variant)}(?!\w)'
                cnt = cnt + s.str.count(pattern).astype(np.float32)
    
            out[f'{feature_name}_{team}_cnt'] = cnt
    
        return out


    def _make_chat_slang_features(self):
        if self.game_chat is None:
            raise ValueError('game_chat is None')
    
        chat = self.game_chat.copy()
    
        rad = self._build_chat_slang_team_stats(chat['radiant_chat_raw'], 'radiant')
        dire = self._build_chat_slang_team_stats(chat['dire_chat_raw'], 'dire')
    
        stats = pd.concat([rad, dire], axis=1)
    
        for feature_name in self.chat_slang_dict.keys():
            stats[f'{feature_name}_diff'] = stats[f'{feature_name}_radiant_cnt'] - stats[f'{feature_name}_dire_cnt']
    
        res = pd.concat([chat[['match_id']].reset_index(drop=True), stats.reset_index(drop=True)], axis=1)
    
        train_block = self.matches_train[['match_id']].merge(res, on='match_id', how='left')
        test_block = self.matches_test[['match_id']].merge(res, on='match_id', how='left')
    
        return train_block, test_block

    def _make_chat_stats_features(self):
        if self.game_chat is None:
            raise ValueError('game_chat is None')
    
        chat = self.game_chat.copy()
    
        rad = self._build_chat_team_stats(chat['radiant_chat_raw'], 'chat_rad')
        dire = self._build_chat_team_stats(chat['dire_chat_raw'], 'chat_dire')
    
        stats = pd.concat([rad, dire], axis=1)
    
        stats['chat_has_chat_diff'] = stats['chat_rad_has_chat'] - stats['chat_dire_has_chat']
        stats['chat_char_cnt_diff'] = stats['chat_rad_char_cnt'] - stats['chat_dire_char_cnt']
        stats['chat_word_cnt_diff'] = stats['chat_rad_word_cnt'] - stats['chat_dire_word_cnt']
    
        res = pd.concat([chat[['match_id']].reset_index(drop=True), stats.reset_index(drop=True)], axis=1)
    
        train_block = self.matches_train[['match_id']].merge(res, on='match_id', how='left')
        test_block = self.matches_test[['match_id']].merge(res, on='match_id', how='left')
    
        return train_block, test_block

       
    def _preprocess_players_info(self, df):

        df = df.copy()
        
        # 1) Удаление матчей, включающих в себя игроков с NaN и которые НЕ оказались в тестевой выборке  
        #test_match_ids = self.matches_test['match_id'].unique()
        #temp = df[~df['match_id'].isin(test_match_ids)]
        #match_id_to_drop_not_test = temp[temp['kills'].isna()]['match_id'].unique()
        #df = df[~df['match_id'].isin(match_id_to_drop_not_test)]

        # 2) Удаление матчей с повторными героями
        heroes = df.groupby('match_id')['hero_id'].apply(lambda x: len(set(x))).reset_index()
        match_id_to_drop_heroes = heroes[heroes['hero_id'] != 10]['match_id'].unique()
        df = df[~df['match_id'].isin(match_id_to_drop_heroes)]

        # 3) Удаление hero_id=0
        match_id_to_drop_hero_0 = df[df['hero_id'] == 0]['match_id'].unique()
        df = df[~df['match_id'].isin(match_id_to_drop_hero_0)]

        # 4) Пофильтруем player_df оставив только матчи которые есть в matches_df_test и matches_df_test
        train_match_ids = self.matches_train['match_id'].unique()
        test_match_ids = self.matches_test['match_id'].unique()
        all_match_ids = np.union1d(train_match_ids, test_match_ids)
        df = df[df['match_id'].isin(all_match_ids)]

        # 5) Удаления матча в котором напутаны слоты
        df_radiant = df[df['player_slot'].isin(self.radiant_slots)].groupby('match_id')['hero_id'].count().reset_index()
        match_id_to_drop_broken_slots = df_radiant[df_radiant['hero_id'] != 5]['match_id'].unique()
        df = df[~df['match_id'].isin(match_id_to_drop_broken_slots)]
        
        return df

    def _preprocess_matches(self, df):
        df = df.copy()
    
        # 1) Добавление флага mmr_missing для пропущенных значений признака avg_mmr
        df['mmr_missing'] = df['avg_mmr'].isna().astype(int)
        
        # 2) Заполнение пропущенных значений признака avg_mmr медианой по режимам
        df['avg_mmr'] = df['avg_mmr'].fillna(df.groupby('game_mode')['avg_mmr'].transform('median'))

        # 3) Cкейлинг avg_mmr квадратным корнем
        df['avg_mmr'] = np.sqrt(df['avg_mmr'])

        # 4) Приведение к дате
        df['date'] = pd.to_datetime(df['date'])

        # 5) Приведение game_mode str
        df['game_mode'] = df['game_mode'].astype(str)
        return df

    def _preprocess_heroes(self, df):
        df = df.copy()
        
        # Удаление мусорных признаков
        df = df.drop(['Unnamed: 0', 'name', 'img', 'icon', 'legs', 'localized_name', 'turn_rate'], axis=1)
        
        # OHE ролей
        df['roles'] = df['roles'].apply(ast.literal_eval)
        unique_classes = df['roles'].explode().unique()
        mlb_roles = MultiLabelBinarizer(classes=unique_classes)
        hero_roles = pd.DataFrame(mlb_roles.fit_transform(df['roles']), columns=mlb_roles.classes_)
        df = pd.concat([df, hero_roles], axis=1).drop('roles', axis=1)
        
        # OHE primary_attr
        primary_attr_encoder = ce.OneHotEncoder(use_cat_names=True)
        ohe_primary_attr = primary_attr_encoder.fit_transform(df[['primary_attr']])
        df = pd.concat([df, ohe_primary_attr], axis=1).drop('primary_attr', axis=1)
        
        # OHE attack_type
        attack_type_encoder = ce.OneHotEncoder(use_cat_names=True)
        ohe_attack_type = attack_type_encoder.fit_transform(df[['attack_type']])
        df = pd.concat([df, ohe_attack_type], axis=1).drop('attack_type', axis=1)
        
        # Ручное заполнение base_health_regen = NaN у id=85 (Undying)
        df.loc[df['id'] == 85, 'base_health_regen'] = -0.25
        
        # Переименования для merge
        df = df.rename(columns={'id':'hero_id'})
        
        # Сбор count признаков
        count_features = list(mlb_roles.classes_) + list(ohe_primary_attr.columns) + list(ohe_attack_type.columns)
        self.heroes_count_features = count_features

        return df

    def _parse_adv_array_16(self, x):
        vals = np.fromstring(str(x).strip()[1:-1], sep=' ', dtype=np.float32)
    
        arr = np.zeros(16, dtype=np.float32)
        if vals.size == 0:
            return arr, 1
    
        m = min(16, vals.size)
        arr[:m] = vals[:m]
    
        return arr, 0

    def _safe_divide(self, num, den):
        return np.divide(num, den, out=np.zeros_like(num, dtype=np.float32), where=den > 0).astype(np.float32)

    def _longest_true_streak(self, mask):
        cur = np.zeros(mask.shape[0], dtype=np.int16)
        best = np.zeros(mask.shape[0], dtype=np.int16)
        for t in range(mask.shape[1]):
            cur = np.where(mask[:, t], cur + 1, 0)
            best = np.maximum(best, cur)
        return best
    
    def _first_true_idx(self, mask):
        any_true = mask.any(axis=1)
        idx = mask.argmax(axis=1).astype(np.int16)
        idx = np.where(any_true, idx, -1)
        return idx.astype(np.int16)
    
    def _last_true_idx(self, mask):
        any_true = mask.any(axis=1)
        rev_idx = mask[:, ::-1].argmax(axis=1)
        idx = mask.shape[1] - 1 - rev_idx
        idx = np.where(any_true, idx, -1)
        return idx.astype(np.int16)
    
    def _ols_slope_from_clean(self, clean):
        mask = ~np.isnan(clean)
        n_valid = mask.sum(axis=1)
    
        t = np.arange(clean.shape[1], dtype=np.float32)
        t_row = np.broadcast_to(t, clean.shape)
    
        t_sum = np.sum(np.where(mask, t_row, 0.0), axis=1)
        y_sum = np.nansum(clean, axis=1)
    
        t_mean = self._safe_divide(t_sum, n_valid)
        y_mean = self._safe_divide(y_sum, n_valid)
    
        t_centered = t_row - t_mean[:, None]
        y_centered = clean - y_mean[:, None]
    
        cov_ty = np.nansum(np.where(mask, t_centered * y_centered, np.nan), axis=1)
        var_t = np.nansum(np.where(mask, t_centered ** 2, np.nan), axis=1)
    
        slope = np.divide(cov_ty, var_t, out=np.zeros_like(cov_ty, dtype=np.float32), where=(n_valid >= 2) & (var_t > 0))
        slope = np.nan_to_num(slope, nan=0.0)
    
        return slope.astype(np.float32)
    
    def _weighted_nanmean(self, clean, weights):
        weights = weights.astype(np.float32)
        mask = ~np.isnan(clean)
        weighted_sum = np.nansum(clean * weights[None, :], axis=1)
        weight_sum = np.sum(np.where(mask, weights[None, :], 0.0), axis=1)
        return self._safe_divide(weighted_sum, weight_sum)
    
    def _make_window_feature_block(self, clean, prefix, windows=(2, 3, 4, 5, 6, 8, 10, 12, 16)):
        n = clean.shape[1]
        res = {}
    
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
    
            for w in windows:
                w = int(min(max(1, w), n))
    
                first = clean[:, :w]
                last = clean[:, -w:]
    
                first_mean = np.nanmean(first, axis=1)
                first_std = np.nanstd(first, axis=1)
                first_min = np.nanmin(first, axis=1)
                first_max = np.nanmax(first, axis=1)
                first_abs_mean = np.nanmean(np.abs(first), axis=1)
                first_sum = np.nansum(first, axis=1)
                first_q25 = np.nanquantile(first, 0.25, axis=1)
                first_q75 = np.nanquantile(first, 0.75, axis=1)
                first_slope = self._ols_slope_from_clean(first)
    
                last_mean = np.nanmean(last, axis=1)
                last_std = np.nanstd(last, axis=1)
                last_min = np.nanmin(last, axis=1)
                last_max = np.nanmax(last, axis=1)
                last_abs_mean = np.nanmean(np.abs(last), axis=1)
                last_sum = np.nansum(last, axis=1)
                last_q25 = np.nanquantile(last, 0.25, axis=1)
                last_q75 = np.nanquantile(last, 0.75, axis=1)
                last_slope = self._ols_slope_from_clean(last)
    
                first_valid = (~np.isnan(first)).sum(axis=1)
                last_valid = (~np.isnan(last)).sum(axis=1)
    
                first_pos_frac = self._safe_divide((first > 0).sum(axis=1), first_valid)
                first_neg_frac = self._safe_divide((first < 0).sum(axis=1), first_valid)
                last_pos_frac = self._safe_divide((last > 0).sum(axis=1), last_valid)
                last_neg_frac = self._safe_divide((last < 0).sum(axis=1), last_valid)
    
                res[f'{prefix}_first{w}_mean'] = np.nan_to_num(first_mean, nan=0.0).astype(np.float32)
                res[f'{prefix}_first{w}_std'] = np.nan_to_num(first_std, nan=0.0).astype(np.float32)
                res[f'{prefix}_first{w}_min'] = np.nan_to_num(first_min, nan=0.0).astype(np.float32)
                res[f'{prefix}_first{w}_max'] = np.nan_to_num(first_max, nan=0.0).astype(np.float32)
                res[f'{prefix}_first{w}_abs_mean'] = np.nan_to_num(first_abs_mean, nan=0.0).astype(np.float32)
                res[f'{prefix}_first{w}_sum'] = np.nan_to_num(first_sum, nan=0.0).astype(np.float32)
                res[f'{prefix}_first{w}_q25'] = np.nan_to_num(first_q25, nan=0.0).astype(np.float32)
                res[f'{prefix}_first{w}_q75'] = np.nan_to_num(first_q75, nan=0.0).astype(np.float32)
                res[f'{prefix}_first{w}_slope'] = np.nan_to_num(first_slope, nan=0.0).astype(np.float32)
                res[f'{prefix}_first{w}_pos_frac'] = np.nan_to_num(first_pos_frac, nan=0.0).astype(np.float32)
                res[f'{prefix}_first{w}_neg_frac'] = np.nan_to_num(first_neg_frac, nan=0.0).astype(np.float32)
    
                res[f'{prefix}_last{w}_mean'] = np.nan_to_num(last_mean, nan=0.0).astype(np.float32)
                res[f'{prefix}_last{w}_std'] = np.nan_to_num(last_std, nan=0.0).astype(np.float32)
                res[f'{prefix}_last{w}_min'] = np.nan_to_num(last_min, nan=0.0).astype(np.float32)
                res[f'{prefix}_last{w}_max'] = np.nan_to_num(last_max, nan=0.0).astype(np.float32)
                res[f'{prefix}_last{w}_abs_mean'] = np.nan_to_num(last_abs_mean, nan=0.0).astype(np.float32)
                res[f'{prefix}_last{w}_sum'] = np.nan_to_num(last_sum, nan=0.0).astype(np.float32)
                res[f'{prefix}_last{w}_q25'] = np.nan_to_num(last_q25, nan=0.0).astype(np.float32)
                res[f'{prefix}_last{w}_q75'] = np.nan_to_num(last_q75, nan=0.0).astype(np.float32)
                res[f'{prefix}_last{w}_slope'] = np.nan_to_num(last_slope, nan=0.0).astype(np.float32)
                res[f'{prefix}_last{w}_pos_frac'] = np.nan_to_num(last_pos_frac, nan=0.0).astype(np.float32)
                res[f'{prefix}_last{w}_neg_frac'] = np.nan_to_num(last_neg_frac, nan=0.0).astype(np.float32)
    
                res[f'{prefix}_last{w}_minus_first{w}_mean'] = np.nan_to_num(last_mean - first_mean, nan=0.0).astype(np.float32)
                res[f'{prefix}_last{w}_minus_first{w}_sum'] = np.nan_to_num(last_sum - first_sum, nan=0.0).astype(np.float32)
                res[f'{prefix}_last{w}_minus_first{w}_slope'] = np.nan_to_num(last_slope - first_slope, nan=0.0).astype(np.float32)
    
        return pd.DataFrame(res)


    def _preprocess_advantages_info(self, df):
        df = df.copy()
    
        all_matches = np.union1d(
            self.matches_train['match_id'].unique(),
            self.matches_test['match_id'].unique()
        )
        df = df[df['match_id'].isin(all_matches)].copy()
        df = df.drop_duplicates(subset='match_id').reset_index(drop=True)
    
        gold_rows = []
        xp_rows = []
        missing_rows = []
    
        for gold_str, xp_str in zip(df['radiant_gold_adv'], df['radiant_exp_adv']):
            gold_arr, gold_missing = self._parse_adv_array_16(gold_str)
            xp_arr, xp_missing = self._parse_adv_array_16(xp_str)
    
            gold_rows.append(gold_arr)
            xp_rows.append(xp_arr)
            missing_rows.append(max(gold_missing, xp_missing))
    
        gold_matrix = np.vstack(gold_rows).astype(np.float32)
        xp_matrix = np.vstack(xp_rows).astype(np.float32)
    
        return {
            'match_id': df['match_id'].to_numpy(),
            'history_adv_missing': np.asarray(missing_rows, dtype=np.int8),
            'gold_matrix': gold_matrix,
            'xp_matrix': xp_matrix
        }

    def _make_adv_features_from_matrix(self, arr, kind: str):
        if kind == 'gold':
            broken_threshold = 100_000
            prefix = 'gold'
            thresholds = [500, 1000, 2000, 4000, 8000]
        elif kind == 'xp':
            broken_threshold = 50_000
            prefix = 'xp'
            thresholds = [500, 1000, 2000, 3000, 5000]
        else:
            raise ValueError('kind must be either \'gold\' or \'xp\'')
    
        broken_mask = np.abs(arr) > broken_threshold
        clean = arr.copy()
        clean[broken_mask] = np.nan
    
        mask = ~np.isnan(clean)
        valid_count = mask.sum(axis=1).astype(np.int16)
    
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
    
            mean_clean = np.nanmean(clean, axis=1)
            std_clean = np.nanstd(clean, axis=1)
            q10_clean = np.nanquantile(clean, 0.10, axis=1)
            q25_clean = np.nanquantile(clean, 0.25, axis=1)
            median_clean = np.nanmedian(clean, axis=1)
            q75_clean = np.nanquantile(clean, 0.75, axis=1)
            q90_clean = np.nanquantile(clean, 0.90, axis=1)
            min_clean = np.nanmin(clean, axis=1)
            max_clean = np.nanmax(clean, axis=1)
            abs_mean_clean = np.nanmean(np.abs(clean), axis=1)
            rms_clean = np.sqrt(np.nanmean(clean ** 2, axis=1))
    
        range_clean = max_clean - min_clean
        iqr_clean = q75_clean - q25_clean
        p90_p10_range_clean = q90_clean - q10_clean
    
        centered = np.abs(clean - median_clean[:, None])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mad_clean = np.nanmedian(centered, axis=1)
    
        rows = np.arange(clean.shape[0])
    
        has_valid = valid_count > 0
        first_valid_idx = mask.argmax(axis=1)
        rev_valid = mask[:, ::-1]
        last_valid_idx = clean.shape[1] - 1 - rev_valid.argmax(axis=1)
    
        first_clean = np.zeros(clean.shape[0], dtype=np.float32)
        last_clean = np.zeros(clean.shape[0], dtype=np.float32)
        first_clean[has_valid] = clean[rows[has_valid], first_valid_idx[has_valid]]
        last_clean[has_valid] = clean[rows[has_valid], last_valid_idx[has_valid]]
    
        pos_clean = np.where(clean > 0, clean, np.nan)
        neg_clean = np.where(clean < 0, clean, np.nan)
    
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            pos_mean = np.nanmean(pos_clean, axis=1)
            pos_max = np.nanmax(pos_clean, axis=1)
            neg_mean = np.nanmean(neg_clean, axis=1)
            neg_min = np.nanmin(neg_clean, axis=1)
    
        pos_sum = np.nansum(np.where(clean > 0, clean, 0.0), axis=1)
        neg_sum = np.nansum(np.where(clean < 0, clean, 0.0), axis=1)
        neg_sum_abs = np.abs(neg_sum)
    
        ahead_mask = clean > 0
        behind_mask = clean < 0
        zero_mask = clean == 0
    
        minutes_ahead = ahead_mask.sum(axis=1).astype(np.int16)
        minutes_behind = behind_mask.sum(axis=1).astype(np.int16)
        minutes_zero = zero_mask.sum(axis=1).astype(np.int16)
    
        ahead_frac = self._safe_divide(minutes_ahead, valid_count)
        behind_frac = self._safe_divide(minutes_behind, valid_count)
        zero_frac = self._safe_divide(minutes_zero, valid_count)
    
        longest_ahead_streak = self._longest_true_streak(ahead_mask)
        longest_behind_streak = self._longest_true_streak(behind_mask)
    
        first_ahead_idx = self._first_true_idx(ahead_mask)
        last_ahead_idx = self._last_true_idx(ahead_mask)
        first_behind_idx = self._first_true_idx(behind_mask)
        last_behind_idx = self._last_true_idx(behind_mask)
    
        sign_state = np.where(clean > 0, 1, np.where(clean < 0, -1, 0)).astype(np.int8)
        valid_transitions = (sign_state[:, 1:] != 0) & (sign_state[:, :-1] != 0)
        sign_changes = ((sign_state[:, 1:] != sign_state[:, :-1]) & valid_transitions).sum(axis=1).astype(np.int16)
    
        diff = np.diff(clean, axis=1)
        diff_valid = ~np.isnan(clean[:, 1:]) & ~np.isnan(clean[:, :-1])
        diff = np.where(diff_valid, diff, np.nan)
    
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            diff_mean = np.nanmean(diff, axis=1)
            diff_std = np.nanstd(diff, axis=1)
            diff_abs_mean = np.nanmean(np.abs(diff), axis=1)
    
        up_moves = np.nansum(diff > 0, axis=1).astype(np.int16)
        down_moves = np.nansum(diff < 0, axis=1).astype(np.int16)
    
        signed_area = np.nansum(clean, axis=1)
        abs_area = np.nansum(np.abs(clean), axis=1)
        pos_area_frac = self._safe_divide(pos_sum, abs_area)
        neg_area_frac = self._safe_divide(neg_sum_abs, abs_area)
    
        filled_neg_inf = np.where(np.isnan(clean), -np.inf, clean)
        filled_pos_inf = np.where(np.isnan(clean), np.inf, clean)
    
        peak_idx = np.argmax(filled_neg_inf, axis=1).astype(np.int16)
        trough_idx = np.argmin(filled_pos_inf, axis=1).astype(np.int16)
    
        peak_val = np.max(np.where(np.isnan(clean), -np.inf, clean), axis=1)
        trough_val = np.min(np.where(np.isnan(clean), np.inf, clean), axis=1)
    
        peak_val = np.where(np.isfinite(peak_val), peak_val, 0.0).astype(np.float32)
        trough_val = np.where(np.isfinite(trough_val), trough_val, 0.0).astype(np.float32)
    
        running_max = np.maximum.accumulate(filled_neg_inf, axis=1)
        drawdown = running_max - filled_neg_inf
        drawdown[~np.isfinite(drawdown)] = 0.0
        max_drawdown = drawdown.max(axis=1).astype(np.float32)
    
        running_min = np.minimum.accumulate(filled_pos_inf, axis=1)
        rebound = filled_neg_inf - running_min
        rebound[~np.isfinite(rebound)] = 0.0
        max_rebound = rebound.max(axis=1).astype(np.float32)
    
        lin_weights = np.arange(1, clean.shape[1] + 1, dtype=np.float32)
        exp_weights = np.power(1.35, np.arange(clean.shape[1], dtype=np.float32))
    
        weighted_mean_lin = self._weighted_nanmean(clean, lin_weights)
        weighted_mean_exp = self._weighted_nanmean(clean, exp_weights)
        weighted_abs_mean_lin = self._weighted_nanmean(np.abs(clean), lin_weights)
        weighted_abs_mean_exp = self._weighted_nanmean(np.abs(clean), exp_weights)
    
        non_broken = ~broken_mask
        first_non_broken = non_broken.argmax(axis=1)
        all_broken = ~non_broken.any(axis=1)
        broken_prefix_len = np.where(all_broken, arr.shape[1], first_non_broken).astype(np.int16)
    
        rev_non_broken = non_broken[:, ::-1]
        last_non_broken_from_end = rev_non_broken.argmax(axis=1)
        broken_suffix_len = np.where(all_broken, arr.shape[1], last_non_broken_from_end).astype(np.int16)
    
        res = {
            f'{prefix}_valid_count': valid_count.astype(np.float32),
            f'{prefix}_first_clean': first_clean.astype(np.float32),
            f'{prefix}_last_clean': last_clean.astype(np.float32),
    
            f'{prefix}_mean_clean': np.nan_to_num(mean_clean, nan=0.0).astype(np.float32),
            f'{prefix}_std_clean': np.nan_to_num(std_clean, nan=0.0).astype(np.float32),
            f'{prefix}_q10_clean': np.nan_to_num(q10_clean, nan=0.0).astype(np.float32),
            f'{prefix}_q25_clean': np.nan_to_num(q25_clean, nan=0.0).astype(np.float32),
            f'{prefix}_median_clean': np.nan_to_num(median_clean, nan=0.0).astype(np.float32),
            f'{prefix}_q75_clean': np.nan_to_num(q75_clean, nan=0.0).astype(np.float32),
            f'{prefix}_q90_clean': np.nan_to_num(q90_clean, nan=0.0).astype(np.float32),
            f'{prefix}_min_clean': np.nan_to_num(min_clean, nan=0.0).astype(np.float32),
            f'{prefix}_max_clean': np.nan_to_num(max_clean, nan=0.0).astype(np.float32),
            f'{prefix}_range_clean': np.nan_to_num(range_clean, nan=0.0).astype(np.float32),
            f'{prefix}_iqr_clean': np.nan_to_num(iqr_clean, nan=0.0).astype(np.float32),
            f'{prefix}_p90_p10_range_clean': np.nan_to_num(p90_p10_range_clean, nan=0.0).astype(np.float32),
            f'{prefix}_mad_clean': np.nan_to_num(mad_clean, nan=0.0).astype(np.float32),
            f'{prefix}_abs_mean_clean': np.nan_to_num(abs_mean_clean, nan=0.0).astype(np.float32),
            f'{prefix}_rms_clean': np.nan_to_num(rms_clean, nan=0.0).astype(np.float32),
    
            f'{prefix}_signed_area': np.nan_to_num(signed_area, nan=0.0).astype(np.float32),
            f'{prefix}_abs_area': np.nan_to_num(abs_area, nan=0.0).astype(np.float32),
            f'{prefix}_pos_area_frac': np.nan_to_num(pos_area_frac, nan=0.0).astype(np.float32),
            f'{prefix}_neg_area_frac': np.nan_to_num(neg_area_frac, nan=0.0).astype(np.float32),
    
            f'{prefix}_pos_mean': np.nan_to_num(pos_mean, nan=0.0).astype(np.float32),
            f'{prefix}_pos_max': np.nan_to_num(pos_max, nan=0.0).astype(np.float32),
            f'{prefix}_pos_sum': np.nan_to_num(pos_sum, nan=0.0).astype(np.float32),
            f'{prefix}_neg_mean': np.nan_to_num(neg_mean, nan=0.0).astype(np.float32),
            f'{prefix}_neg_min': np.nan_to_num(neg_min, nan=0.0).astype(np.float32),
            f'{prefix}_neg_sum_abs': np.nan_to_num(neg_sum_abs, nan=0.0).astype(np.float32),
    
            f'{prefix}_minutes_ahead': minutes_ahead.astype(np.float32),
            f'{prefix}_minutes_behind': minutes_behind.astype(np.float32),
            f'{prefix}_minutes_zero': minutes_zero.astype(np.float32),
            f'{prefix}_ahead_frac': np.nan_to_num(ahead_frac, nan=0.0).astype(np.float32),
            f'{prefix}_behind_frac': np.nan_to_num(behind_frac, nan=0.0).astype(np.float32),
            f'{prefix}_zero_frac': np.nan_to_num(zero_frac, nan=0.0).astype(np.float32),
    
            f'{prefix}_longest_ahead_streak': longest_ahead_streak.astype(np.float32),
            f'{prefix}_longest_behind_streak': longest_behind_streak.astype(np.float32),
    
            f'{prefix}_first_ahead_idx': first_ahead_idx.astype(np.float32),
            f'{prefix}_last_ahead_idx': last_ahead_idx.astype(np.float32),
            f'{prefix}_first_behind_idx': first_behind_idx.astype(np.float32),
            f'{prefix}_last_behind_idx': last_behind_idx.astype(np.float32),
    
            f'{prefix}_sign_changes': sign_changes.astype(np.float32),
    
            f'{prefix}_diff_mean': np.nan_to_num(diff_mean, nan=0.0).astype(np.float32),
            f'{prefix}_diff_std': np.nan_to_num(diff_std, nan=0.0).astype(np.float32),
            f'{prefix}_diff_abs_mean': np.nan_to_num(diff_abs_mean, nan=0.0).astype(np.float32),
            f'{prefix}_up_moves': up_moves.astype(np.float32),
            f'{prefix}_down_moves': down_moves.astype(np.float32),
    
            f'{prefix}_peak_val': peak_val.astype(np.float32),
            f'{prefix}_trough_val': trough_val.astype(np.float32),
            f'{prefix}_peak_idx': peak_idx.astype(np.float32),
            f'{prefix}_trough_idx': trough_idx.astype(np.float32),
            f'{prefix}_swing_range': (peak_val - trough_val).astype(np.float32),
            f'{prefix}_max_drawdown': max_drawdown.astype(np.float32),
            f'{prefix}_max_rebound': max_rebound.astype(np.float32),
    
            f'{prefix}_weighted_mean_lin': np.nan_to_num(weighted_mean_lin, nan=0.0).astype(np.float32),
            f'{prefix}_weighted_mean_exp': np.nan_to_num(weighted_mean_exp, nan=0.0).astype(np.float32),
            f'{prefix}_weighted_abs_mean_lin': np.nan_to_num(weighted_abs_mean_lin, nan=0.0).astype(np.float32),
            f'{prefix}_weighted_abs_mean_exp': np.nan_to_num(weighted_abs_mean_exp, nan=0.0).astype(np.float32),
    
            f'{prefix}_first_last_diff': (last_clean - first_clean).astype(np.float32),
            f'{prefix}_last_minus_mean': np.nan_to_num(last_clean - mean_clean, nan=0.0).astype(np.float32),
            f'{prefix}_last_over_abs_mean_ratio': self._safe_divide(np.abs(last_clean), np.nan_to_num(abs_mean_clean, nan=0.0) + 1e-6),
    
            f'{prefix}_broken_prefix_len': broken_prefix_len.astype(np.float32),
            f'{prefix}_broken_suffix_len': broken_suffix_len.astype(np.float32),
            f'{prefix}_has_broken': broken_mask.any(axis=1).astype(np.float32),
            f'{prefix}_n_broken': broken_mask.sum(axis=1).astype(np.float32),
        }
    
        for thr in thresholds:
            res[f'{prefix}_n_ge_{thr}'] = (clean >= thr).sum(axis=1).astype(np.float32)
            res[f'{prefix}_n_le_minus_{thr}'] = (clean <= -thr).sum(axis=1).astype(np.float32)
            res[f'{prefix}_frac_ge_{thr}'] = self._safe_divide((clean >= thr).sum(axis=1), valid_count)
            res[f'{prefix}_frac_le_minus_{thr}'] = self._safe_divide((clean <= -thr).sum(axis=1), valid_count)
    
        base_df = pd.DataFrame(res)
        window_df = self._make_window_feature_block(clean, prefix=prefix, windows=(2, 3, 4, 5, 6, 8, 10, 12, 16))
    
        return pd.concat([base_df, window_df], axis=1)

    def _add_binned_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        cols: list[str],
        n_bins: int = 5,
        strategy: str = 'quantile',
        keep_original: bool = True
    ):
        X_train = X_train.copy()
        X_test = X_test.copy()
    
        discretizer = KBinsDiscretizer(n_bins=n_bins,encode='onehot-dense',strategy=strategy)
    
        train_binned = discretizer.fit_transform(X_train[cols])
        test_binned = discretizer.transform(X_test[cols])
    
        bin_feature_names = []
        for col, edges in zip(cols, discretizer.bin_edges_):
            n_real_bins = len(edges) - 1
            for i in range(n_real_bins):
                bin_feature_names.append(f'{col}_bin_{i}')
    
        train_binned = pd.DataFrame(
            train_binned,
            columns=bin_feature_names,
            index=X_train.index
        ).astype(np.float32)
    
        test_binned = pd.DataFrame(
            test_binned,
            columns=bin_feature_names,
            index=X_test.index
        ).astype(np.float32)
    
        if keep_original:
            X_train = pd.concat([X_train, train_binned], axis=1)
            X_test = pd.concat([X_test, test_binned], axis=1)
        else:
            X_train = pd.concat([X_train.drop(columns=cols), train_binned], axis=1)
            X_test = pd.concat([X_test.drop(columns=cols), test_binned], axis=1)
    
        return X_train, X_test

    def _make_trend_features_from_matrix(self, arr, kind: str, method: str = 'ols'):
        if kind == 'gold':
            broken_threshold = 100_000
            prefix = 'gold'
        elif kind == 'xp':
            broken_threshold = 50_000
            prefix = 'xp'
    
        broken_mask = np.abs(arr) > broken_threshold
        clean = arr.copy()
        clean[broken_mask] = np.nan
    
        mask = ~np.isnan(clean)
        n_valid = mask.sum(axis=1)
    
        rows = np.arange(clean.shape[0])
        t = np.arange(clean.shape[1], dtype=np.float32)
    
        first_idx = mask.argmax(axis=1)
        rev_mask = mask[:, ::-1]
        last_idx_from_end = rev_mask.argmax(axis=1)
        last_idx = clean.shape[1] - 1 - last_idx_from_end
    
        first_val = np.zeros(clean.shape[0], dtype=np.float32)
        last_val = np.zeros(clean.shape[0], dtype=np.float32)
    
        has_valid = n_valid > 0
        first_val[has_valid] = clean[rows[has_valid], first_idx[has_valid]]
        last_val[has_valid] = clean[rows[has_valid], last_idx[has_valid]]
    
        if method == 'delta':
            delta = np.zeros(clean.shape[0], dtype=np.float32)
            delta[has_valid] = last_val[has_valid] - first_val[has_valid]
    
            delta_t = np.maximum(last_idx - first_idx, 1)
            slope_delta = np.zeros(clean.shape[0], dtype=np.float32)
            valid_delta = has_valid & (n_valid >= 2)
            slope_delta[valid_delta] = delta[valid_delta] / delta_t[valid_delta]
    
            return pd.DataFrame({
                f'{prefix}_trend_delta': delta.astype(np.float32),
                f'{prefix}_trend_slope_delta': slope_delta.astype(np.float32)
            })
    
        elif method == 'ols':
            t_row = np.broadcast_to(t, clean.shape)
    
            t_sum = np.sum(np.where(mask, t_row, 0.0), axis=1)
            y_sum = np.nansum(clean, axis=1)
    
            t_mean = np.divide(
                t_sum,
                n_valid,
                out=np.zeros_like(t_sum, dtype=np.float32),
                where=n_valid > 0
            )
            y_mean = np.divide(
                y_sum,
                n_valid,
                out=np.zeros_like(y_sum, dtype=np.float32),
                where=n_valid > 0
            )
    
            t_centered = t_row - t_mean[:, None]
            y_centered = clean - y_mean[:, None]
    
            cov_ty = np.nansum(np.where(mask, t_centered * y_centered, np.nan), axis=1)
            var_t = np.nansum(np.where(mask, t_centered ** 2, np.nan), axis=1)
    
            slope = np.divide(
                cov_ty,
                var_t,
                out=np.zeros_like(cov_ty, dtype=np.float32),
                where=(n_valid >= 2) & (var_t > 0)
            )
    
            intercept = y_mean - slope * t_mean
    
            y_pred = slope[:, None] * t_row + intercept[:, None]
    
            ss_res = np.nansum(np.where(mask, (clean - y_pred) ** 2, np.nan), axis=1)
            ss_tot = np.nansum(np.where(mask, (clean - y_mean[:, None]) ** 2, np.nan), axis=1)
    
            r2 = np.divide(
                1.0 - np.divide(
                    ss_res,
                    ss_tot,
                    out=np.zeros_like(ss_res, dtype=np.float32),
                    where=ss_tot > 0
                ),
                1.0,
                out=np.zeros_like(ss_res, dtype=np.float32),
                where=(n_valid >= 2) & (ss_tot > 0)
            )
    
            r2 = np.nan_to_num(r2, nan=0.0)
            slope = np.nan_to_num(slope, nan=0.0)
            intercept = np.nan_to_num(intercept, nan=0.0)
    
            return pd.DataFrame({
                f'{prefix}_trend_slope_ols': slope.astype(np.float32),
                f'{prefix}_trend_intercept_ols': intercept.astype(np.float32),
                f'{prefix}_trend_r2_ols': r2.astype(np.float32)
            })

    def _make_advantages_trend_features(self, method: str = 'ols'):
        adv = self.advantages_info
    
        gold_trend = self._make_trend_features_from_matrix(
            adv['gold_matrix'],
            kind='gold',
            method=method
        )
    
        xp_trend = self._make_trend_features_from_matrix(
            adv['xp_matrix'],
            kind='xp',
            method=method
        )
    
        res = pd.concat(
            [
                pd.DataFrame({'match_id': adv['match_id']}),
                gold_trend,
                xp_trend
            ],
            axis=1
        )
    
        train_block = self.matches_train[['match_id']].merge(res, on='match_id', how='left')
        test_block = self.matches_test[['match_id']].merge(res, on='match_id', how='left')
    
        return train_block, test_block
    
    def _make_advantages_agg_features(self):
        adv = self.advantages_info
    
        gold_features = self._make_adv_features_from_matrix(adv['gold_matrix'], kind='gold')
        xp_features = self._make_adv_features_from_matrix(adv['xp_matrix'], kind='xp')
    
        res = pd.concat(
            [
                pd.DataFrame({
                    'match_id': adv['match_id'],
                    'history_adv_missing': adv['history_adv_missing']
                }),
                gold_features,
                xp_features
            ],
            axis=1
        )
    
        train_block = self.matches_train[['match_id']].merge(res, on='match_id', how='left')
        test_block = self.matches_test[['match_id']].merge(res, on='match_id', how='left')
    
        return train_block, test_block
        
    def _make_game_mode_features(self):

        # 1) OHE для признака game_mode
        df_train = self.matches_train.copy()
        df_test = self.matches_test.copy()

        ohe = ce.OneHotEncoder(use_cat_names=True)
        
        ohe_game_modes_train = ohe.fit_transform(df_train['game_mode'])
        ohe_game_modes_test = ohe.transform(df_test['game_mode'])
        
        ohe_game_modes_train_with_ids = pd.concat([df_train[['match_id']], ohe_game_modes_train], axis=1)
        ohe_game_modes_test_with_ids = pd.concat([df_test[['match_id']], ohe_game_modes_test], axis=1)

        return ohe_game_modes_train_with_ids, ohe_game_modes_test_with_ids


    def _make_time_features_for_matches(self):
        
        patch_dates = {
            '7.35b': pd.Timestamp('2023-12-21'),
            '7.35c': pd.Timestamp('2024-02-21'),
            '7.35d': pd.Timestamp('2024-03-21'),
            '7.36': pd.Timestamp('2024-05-22'),
            '7.36a': pd.Timestamp('2024-05-26'),
            '7.36b': pd.Timestamp('2024-06-05'),
            '7.36c': pd.Timestamp('2024-06-24'),
            '7.37': pd.Timestamp('2024-07-31'),
            '7.37b': pd.Timestamp('2024-08-14'),
            '7.37c': pd.Timestamp('2024-08-28'),
            '7.37d': pd.Timestamp('2024-10-03'),
            '7.37e': pd.Timestamp('2024-11-19'),
        }
    
        patch_order = sorted(patch_dates.items(), key=lambda x: x[1])
            
        def assign_patch_regime(date):
            cur = patch_order[0][0]
            for patch, patch_date in patch_order:
                if date >= patch_date:
                    cur = patch
                else:
                    break
            return cur
    
        df_train = self.matches_train.copy()
        df_test = self.matches_test.copy()
    
        for df in [df_train, df_test]:
            df['date'] = pd.to_datetime(df['date'])
            df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
            df['patch_regime'] = df['date'].apply(assign_patch_regime)
    
        present_patches = sorted(
            set(df_train['patch_regime']) | set(df_test['patch_regime']),
            key=lambda x: patch_dates[x]
        )
    
        df_train['patch_regime'] = pd.Categorical(df_train['patch_regime'], categories=present_patches)
        df_test['patch_regime'] = pd.Categorical(df_test['patch_regime'], categories=present_patches)
    
        ohe = ce.OneHotEncoder(cols=['patch_regime'], use_cat_names=True)
    
        patch_train = ohe.fit_transform(df_train[['patch_regime']])
        patch_test = ohe.transform(df_test[['patch_regime']])
    
        time_features_train = pd.concat([df_train[['match_id', 'is_weekend']].reset_index(drop=True), patch_train.reset_index(drop=True)], axis=1)
        time_features_test = pd.concat([df_test[['match_id', 'is_weekend']].reset_index(drop=True), patch_test.reset_index(drop=True)], axis=1)
    
        return time_features_train, time_features_test

    def _make_hero_winrate_features(self, alpha: float = 20.0):
        df = self.players_info.copy()
    
        train_match_ids = self.matches_train['match_id'].unique()
        test_match_ids = self.matches_test['match_id'].unique()
        all_match_ids = np.union1d(train_match_ids, test_match_ids)
    
        df_hist = df[df['match_id'].isin(train_match_ids)].copy()
        df_all = df[df['match_id'].isin(all_match_ids)].copy()
    
        df_hist = df_hist.merge(self.matches_train[['match_id', 'radiant_win']], how='left', on='match_id')
        df_hist['hero_win'] = np.where(df_hist['player_slot'].isin(self.radiant_slots), df_hist['radiant_win'], 1 - df_hist['radiant_win'])
    
        global_wr = df_hist['hero_win'].mean()
    
        hero_stats = df_hist.groupby('hero_id')['hero_win'].agg(['sum', 'count', 'mean']).reset_index()
        hero_stats = hero_stats.rename(columns={'sum': 'hero_wins_sum', 'count': 'hero_n_matches', 'mean': 'hero_win_rate_raw'})
        hero_stats['hero_win_rate_smoothed'] = (hero_stats['hero_wins_sum'] + alpha * global_wr) / (hero_stats['hero_n_matches'] + alpha)
        hero_stats = hero_stats[['hero_id', 'hero_win_rate_raw', 'hero_win_rate_smoothed', 'hero_n_matches']]
    
        df_all = df_all.merge(hero_stats, how='left', on='hero_id')
        df_all['hero_win_rate_raw'] = df_all['hero_win_rate_raw'].fillna(global_wr)
        df_all['hero_win_rate_smoothed'] = df_all['hero_win_rate_smoothed'].fillna(global_wr)
        df_all['hero_n_matches'] = df_all['hero_n_matches'].fillna(0)
    
        df_radiant = df_all[df_all['player_slot'].isin(self.radiant_slots)].copy()
        df_dire = df_all[df_all['player_slot'].isin(self.dire_slots)].copy()
    
        df_radiant = df_radiant.sort_values(['match_id', 'player_slot']).copy()
        df_dire = df_dire.sort_values(['match_id', 'player_slot']).copy()
    
        df_radiant['hero_pos'] = df_radiant.groupby('match_id').cumcount()
        df_dire['hero_pos'] = df_dire.groupby('match_id').cumcount()
    
        raw_feature_cols = ['hero_win_rate_raw', 'hero_win_rate_smoothed', 'hero_n_matches']
    
        radiant_raw_blocks = []
        dire_raw_blocks = []
    
        for col in raw_feature_cols:
            cur_radiant = df_radiant.pivot(index='match_id', columns='hero_pos', values=col).reset_index()
            cur_radiant = cur_radiant.rename(columns={i: f'{col}_radiant_{i}' for i in range(5)})
            radiant_raw_blocks.append(cur_radiant)
    
            cur_dire = df_dire.pivot(index='match_id', columns='hero_pos', values=col).reset_index()
            cur_dire = cur_dire.rename(columns={i: f'{col}_dire_{i}' for i in range(5)})
            dire_raw_blocks.append(cur_dire)
    
        df_radiant_raw = radiant_raw_blocks[0]
        for block in radiant_raw_blocks[1:]:
            df_radiant_raw = df_radiant_raw.merge(block, how='left', on='match_id')
    
        df_dire_raw = dire_raw_blocks[0]
        for block in dire_raw_blocks[1:]:
            df_dire_raw = df_dire_raw.merge(block, how='left', on='match_id')
    
        df_radiant_mean = df_radiant.groupby('match_id')[raw_feature_cols].mean().reset_index()
        df_radiant_mean = df_radiant_mean.rename(columns={'hero_win_rate_raw': 'hero_win_rate_raw_radiant_mean', 'hero_win_rate_smoothed': 'hero_win_rate_smoothed_radiant_mean', 'hero_n_matches': 'hero_n_matches_radiant_mean'})
    
        df_dire_mean = df_dire.groupby('match_id')[raw_feature_cols].mean().reset_index()
        df_dire_mean = df_dire_mean.rename(columns={'hero_win_rate_raw': 'hero_win_rate_raw_dire_mean', 'hero_win_rate_smoothed': 'hero_win_rate_smoothed_dire_mean', 'hero_n_matches': 'hero_n_matches_dire_mean'})
    
        df_match = df_radiant_raw.merge(df_dire_raw, how='inner', on='match_id')
        df_match = df_match.merge(df_radiant_mean, how='left', on='match_id')
        df_match = df_match.merge(df_dire_mean, how='left', on='match_id')
    
        train_ids = self.matches_train[['match_id']].copy()
        test_ids = self.matches_test[['match_id']].copy()
    
        train_block = train_ids.merge(df_match, how='left', on='match_id')
        test_block = test_ids.merge(df_match, how='left', on='match_id')
    
        return train_block, test_block

    def _make_regions_features(self):
        ohe = ce.OneHotEncoder(use_cat_names=True)
    
        ohe_regions_train = ohe.fit_transform(self.matches_train['region'])
        ohe_regions_test = ohe.transform(self.matches_test['region'])

        ohe_regions_train_with_ids = pd.concat([self.matches_train['match_id'], ohe_regions_train], axis=1)
        ohe_regions_test_with_ids = pd.concat([self.matches_test['match_id'], ohe_regions_test], axis=1)

        return  ohe_regions_train_with_ids, ohe_regions_test_with_ids

    def _merge_feature_blocks(self, base_train, base_test, train_blocks, test_blocks):
        X_train = base_train.copy()
        X_test = base_test.copy()
    
        for block in train_blocks:
            X_train = X_train.merge(block, on='match_id', how='left')
    
        for block in test_blocks:
            X_test = X_test.merge(block, on='match_id', how='left')

        X_train = X_train.drop(columns='match_id')
        X_test = X_test.drop(columns='match_id')

        return X_train, X_test

    def _make_base_match_features(self):
        base_cols = ['match_id', 'avg_mmr', 'mmr_missing']
        
        train_block = self.matches_train[base_cols].copy()
        test_block = self.matches_test[base_cols].copy()
        
        return train_block, test_block

    def _make_heroes_features(self):
        df = self.players_info.copy()
        
        radiant_slots = set(range(0, 5))
        dire_slots = set(range(128, 133))

        unique_radiant_heroes= df[df['player_slot'].isin(radiant_slots)]['hero_id'].unique()
        unique_dire_heroes= df[df['player_slot'].isin(dire_slots)]['hero_id'].unique()
        all_heroes = np.union1d(unique_radiant_heroes, unique_dire_heroes)

        mlb = MultiLabelBinarizer(classes=all_heroes)
        mlb.fit([[]])
        feature_heroes = [f'hero_{i}' for i in mlb.classes_]

        df_radiant_matches = df[df['player_slot'].isin(radiant_slots)].groupby('match_id')['hero_id'].apply(list).reset_index()
        df_dire_matches = df[df['player_slot'].isin(dire_slots)].groupby('match_id')['hero_id'].apply(list).reset_index()
        
        radiant_ohe = pd.DataFrame(mlb.transform(df_radiant_matches['hero_id']), columns=[f'hero_{i}' for i in mlb.classes_])
        dire_ohe = pd.DataFrame(mlb.transform(df_dire_matches['hero_id']), columns=[f'hero_{i}' for i in mlb.classes_]) * (-1)

        df_radiant_matches = pd.concat([df_radiant_matches[['match_id']], radiant_ohe], axis=1).set_index('match_id')
        df_dire_matches = pd.concat([df_dire_matches[['match_id']], dire_ohe], axis=1).set_index('match_id')

        heroes_enc = df_radiant_matches.add(df_dire_matches).reset_index()

        train_ids = self.matches_train[['match_id']].copy()
        test_ids = self.matches_test[['match_id']].copy()

        train_block = train_ids.merge(heroes_enc, on='match_id', how='left')
        test_block = test_ids.merge(heroes_enc, on='match_id', how='left')

        return train_block, test_block

    def _make_team_composotion_features(self):
        hero_df = self.heroes_info.copy()
        players_info = self.players_info.copy()

        radiant_slots = list(range(0, 5))
        dire_slots = list(range(128, 133))

        heroes_count_features = self.heroes_count_features
        heroes_count_features_df = hero_df[['hero_id'] + heroes_count_features]

        players_info = players_info.merge(heroes_count_features_df, how='left', on='hero_id')

        df_radiant_matches = players_info[players_info['player_slot'].isin(radiant_slots)]
        df_dire_matches = players_info[players_info['player_slot'].isin(dire_slots)]

        df_radiant_team_composotion_features = df_radiant_matches.groupby('match_id')[heroes_count_features].sum().reset_index()
        df_dire_team_composotion_features = df_dire_matches.groupby('match_id')[heroes_count_features].sum().reset_index()

        df_radiant_team_composotion_features = df_radiant_team_composotion_features.rename(columns={feature:feature + '_radiant' for feature in heroes_count_features})
        df_dire_team_composotion_features = df_dire_team_composotion_features.rename(columns={feature:feature + '_dire' for feature in heroes_count_features}) 
        
        res = pd.merge(left=df_radiant_team_composotion_features, right=df_dire_team_composotion_features, how='left', on='match_id')

        train_ids = self.matches_train[['match_id']].copy()
        test_ids = self.matches_test[['match_id']].copy()

        train_block = train_ids.merge(res, on='match_id', how='left')
        test_block = test_ids.merge(res, on='match_id', how='left')

        return train_block, test_block

    def _make_hero_stat_std_features(self):
        hero_df = self.heroes_info.copy()
        players_info = self.players_info.copy()
    
        radiant_slots = list(range(0, 5))
        dire_slots = list(range(128, 133))
    
        hero_stat_std_features = [
            'base_health',
            'base_health_regen',
            'base_mana',
            'base_mana_regen',
            'base_armor',
            'base_mr',
            'base_attack_min',
            'base_attack_max',
            'base_str',
            'base_agi',
            'base_int',
            'str_gain',
            'agi_gain',
            'int_gain',
            'attack_range',
            'projectile_speed',
            'attack_rate',
            'base_attack_time',
            'attack_point',
            'move_speed',
            'day_vision',
            'night_vision'
        ]
    
    
        hero_stat_std_features_df = hero_df[['hero_id'] + hero_stat_std_features]
    
        players_info = players_info.merge(hero_stat_std_features_df, how='left', on='hero_id')
    
        df_radiant_matches = players_info[players_info['player_slot'].isin(radiant_slots)]
        df_dire_matches = players_info[players_info['player_slot'].isin(dire_slots)]
    
        df_radiant_hero_stat_std_features = df_radiant_matches.groupby('match_id')[hero_stat_std_features].std(ddof=0).reset_index()
        df_dire_hero_stat_std_features = df_dire_matches.groupby('match_id')[hero_stat_std_features].std(ddof=0).reset_index()
    
        df_radiant_hero_stat_std_features = df_radiant_hero_stat_std_features.rename(columns={feature: feature + '_radiant_std' for feature in hero_stat_std_features})
        df_dire_hero_stat_std_features = df_dire_hero_stat_std_features.rename(columns={feature: feature + '_dire_std' for feature in hero_stat_std_features})
    
        res = pd.merge(left=df_radiant_hero_stat_std_features, right=df_dire_hero_stat_std_features, how='left', on='match_id')
        
        train_ids = self.matches_train[['match_id']].copy()
        test_ids = self.matches_test[['match_id']].copy()
    
        train_block = train_ids.merge(res, on='match_id', how='left')
        test_block = test_ids.merge(res, on='match_id', how='left')
    
        return train_block, test_block

    def _make_hero_stat_mean_features(self):
        hero_df = self.heroes_info.copy()
        players_info = self.players_info.copy()
    
        radiant_slots = list(range(0, 5))
        dire_slots = list(range(128, 133))
    
        hero_stat_mean_features = [
            'base_health',
            'base_health_regen',
            'base_mana',
            'base_mana_regen',
            'base_armor',
            'base_mr',
            'base_attack_min',
            'base_attack_max',
            'base_str',
            'base_agi',
            'base_int',
            'str_gain',
            'agi_gain',
            'int_gain',
            'attack_range',
            'projectile_speed',
            'attack_rate',
            'base_attack_time',
            'attack_point',
            'move_speed',
            'day_vision',
            'night_vision'
        ]
    
        hero_stat_mean_features_df = hero_df[['hero_id'] + hero_stat_mean_features]
    
        players_info = players_info.merge(hero_stat_mean_features_df, how='left', on='hero_id')
    
        df_radiant_matches = players_info[players_info['player_slot'].isin(radiant_slots)]
        df_dire_matches = players_info[players_info['player_slot'].isin(dire_slots)]
    
        df_radiant_hero_stat_mean_features = (
            df_radiant_matches.groupby('match_id')[hero_stat_mean_features].mean().reset_index()
        )
        df_dire_hero_stat_mean_features = (
            df_dire_matches.groupby('match_id')[hero_stat_mean_features].mean().reset_index()
        )
    
        df_radiant_hero_stat_mean_features = df_radiant_hero_stat_mean_features.rename(
            columns={feature: feature + '_radiant_mean' for feature in hero_stat_mean_features}
        )
        df_dire_hero_stat_mean_features = df_dire_hero_stat_mean_features.rename(
            columns={feature: feature + '_dire_mean' for feature in hero_stat_mean_features}
        )
    
        res = pd.merge(
            left=df_radiant_hero_stat_mean_features,
            right=df_dire_hero_stat_mean_features,
            how='left',
            on='match_id'
        )
    
        train_ids = self.matches_train[['match_id']].copy()
        test_ids = self.matches_test[['match_id']].copy()
    
        train_block = train_ids.merge(res, on='match_id', how='left')
        test_block = test_ids.merge(res, on='match_id', how='left')
    
        return train_block, test_block

    # НАПОМИНАЛКА: ПОПРОБОВАТЬ ДОБАВИТЬ ЧТО-ТО КРОМЕ MEAN ПО КОМАНДА
    def _make_hero_match_stat_features(self, use_diff: bool = True):
        df = self.players_info.copy()
    
        train_match_ids = self.matches_train['match_id'].unique()
        test_match_ids = self.matches_test['match_id'].unique()
        all_match_ids = np.union1d(train_match_ids, test_match_ids)
    
        df_hist = df[df['match_id'].isin(train_match_ids)].copy()
        df_all = df[df['match_id'].isin(all_match_ids)].copy()
    
        stat_features = [
            'kills',
            'deaths',
            'assists',
            'gold',
            'last_hits',
            'denies',
            'gold_per_min',
            'xp_per_min',
            'hero_damage',
            'tower_damage'
        ]
    
        df_hist_stats = df_hist.dropna(subset=stat_features)
    
        df_hist_mean = df_hist_stats.groupby('hero_id')[stat_features].mean().reset_index()
        df_hist_std = df_hist_stats.groupby('hero_id')[stat_features].std().reset_index().fillna(0)
    
        df_list = [
            ('mean', df_hist_mean),
            ('std', df_hist_std),
        ]
    
        all_hero_match_stat_features = []
    
        for agg_name, df_with_hero_match_stat in df_list:
            renamed_cols = {col: f'hero_{col}_{agg_name}' for col in df_with_hero_match_stat.columns if col != 'hero_id'}
            df_with_hero_match_stat = df_with_hero_match_stat.rename(columns=renamed_cols)
            df_all = df_all.merge(df_with_hero_match_stat, how='left', on='hero_id')
            all_hero_match_stat_features.extend(list(renamed_cols.values()))
    
        df_radiant_matches = df_all[df_all['player_slot'].isin(self.radiant_slots)]
        df_dire_matches = df_all[df_all['player_slot'].isin(self.dire_slots)]
    
        team_aggs = ['mean', 'max']
    
        df_radiant_matches_with_stats = df_radiant_matches.groupby('match_id')[all_hero_match_stat_features].agg(team_aggs)
        df_radiant_matches_with_stats.columns = [f'{col}_{agg}' for col, agg in df_radiant_matches_with_stats.columns]
        df_radiant_matches_with_stats = df_radiant_matches_with_stats.reset_index()
    
        df_dire_matches_with_stats = df_dire_matches.groupby('match_id')[all_hero_match_stat_features].agg(team_aggs)
        df_dire_matches_with_stats.columns = [f'{col}_{agg}' for col, agg in df_dire_matches_with_stats.columns]
        df_dire_matches_with_stats = df_dire_matches_with_stats.reset_index()
    
        team_feature_cols = [col for col in df_radiant_matches_with_stats.columns if col != 'match_id']
    
        if use_diff:
            df_matches_with_stats = df_radiant_matches_with_stats.merge(df_dire_matches_with_stats, how='inner', on='match_id', suffixes=('_radiant', '_dire'))
            diff_cols = {'match_id': df_matches_with_stats['match_id']}
            for col in team_feature_cols:
                diff_cols[f'{col}_diff'] = df_matches_with_stats[f'{col}_radiant'] - df_matches_with_stats[f'{col}_dire']
            df_matches_with_stats = pd.DataFrame(diff_cols)
        else:
            df_radiant_matches_with_stats = df_radiant_matches_with_stats.rename(columns={col: f'{col}_radiant' for col in team_feature_cols})
            df_dire_matches_with_stats = df_dire_matches_with_stats.rename(columns={col: f'{col}_dire' for col in team_feature_cols})
            df_matches_with_stats = df_radiant_matches_with_stats.merge(df_dire_matches_with_stats, how='inner', on='match_id')
    
        train_ids = self.matches_train[['match_id']].copy()
        test_ids = self.matches_test[['match_id']].copy()
    
        train_block = train_ids.merge(df_matches_with_stats, on='match_id', how='left')
        test_block = test_ids.merge(df_matches_with_stats, on='match_id', how='left')
    
        return train_block, test_block

    def _make_hero_pair_synergy_features(self, use_diff: bool = True, alpha_pair: float = 40):
        df = self.players_info.copy()
    
        train_match_ids = self.matches_train['match_id'].unique()
        test_match_ids = self.matches_test['match_id'].unique()
        all_match_ids = np.union1d(train_match_ids, test_match_ids)
    
        df_hist = df[df['match_id'].isin(train_match_ids)].copy()
        df_all = df[df['match_id'].isin(all_match_ids)].copy()
    
        df_hist = df_hist.merge(self.matches_train[['match_id', 'radiant_win']], how='left', on='match_id')
        df_hist['team'] = np.where(df_hist['player_slot'].isin(self.radiant_slots), 'radiant', 'dire')
        df_hist['team_win'] = np.where(df_hist['team'] == 'radiant', df_hist['radiant_win'], 1 - df_hist['radiant_win'])
    
        global_wr = df_hist['team_win'].mean()
    
        team_hist = df_hist.groupby(['match_id', 'team']).agg({'hero_id': list, 'team_win': 'first'}).reset_index()
    
        pair_rows = []
        for _, row in team_hist.iterrows():
            heroes = sorted(set(row['hero_id']))
            if len(heroes) != 5:
                continue
            for h1, h2 in combinations(heroes, 2):
                pair_rows.append((h1, h2, row['team_win']))
    
        pair_hist = pd.DataFrame(pair_rows, columns=['hero_1', 'hero_2', 'pair_win'])
    
        pair_stats = pair_hist.groupby(['hero_1', 'hero_2'])['pair_win'].agg(['sum', 'count']).reset_index()
        pair_stats = pair_stats.rename(columns={'sum': 'pair_wins_sum', 'count': 'pair_n_matches'})
        pair_stats['pair_win_rate_smoothed'] = (pair_stats['pair_wins_sum'] + alpha_pair * global_wr) / (pair_stats['pair_n_matches'] + alpha_pair)
        pair_stats = pair_stats[['hero_1', 'hero_2', 'pair_win_rate_smoothed', 'pair_n_matches']]
    
        df_all['team'] = np.where(df_all['player_slot'].isin(self.radiant_slots), 'radiant', 'dire')
        team_all = df_all.groupby(['match_id', 'team'])['hero_id'].apply(list).reset_index()
    
        pair_rows_all = []
        for _, row in team_all.iterrows():
            heroes = sorted(set(row['hero_id']))
            if len(heroes) != 5:
                continue
            for h1, h2 in combinations(heroes, 2):
                pair_rows_all.append((row['match_id'], row['team'], h1, h2))
    
        pair_all = pd.DataFrame(pair_rows_all, columns=['match_id', 'team', 'hero_1', 'hero_2'])
    
        pair_all = pair_all.merge(pair_stats, how='left', on=['hero_1', 'hero_2'])
        pair_all['pair_win_rate_smoothed'] = pair_all['pair_win_rate_smoothed'].fillna(global_wr)
        pair_all['pair_n_matches'] = pair_all['pair_n_matches'].fillna(0)
        pair_all['pair_is_unseen'] = (pair_all['pair_n_matches'] == 0).astype(np.float32)
    
        pair_team = pair_all.groupby(['match_id', 'team']).agg({
            'pair_win_rate_smoothed': ['mean', 'max', 'min', 'std'],
            'pair_n_matches': ['mean', 'max'],
            'pair_is_unseen': ['sum']
        }).reset_index()
    
        pair_team.columns = ['match_id', 'team'] + [f'{col}_{agg}' for col, agg in pair_team.columns.tolist()[2:]]
    
        df_radiant = pair_team[pair_team['team'] == 'radiant'].drop(columns='team').reset_index(drop=True)
        df_dire = pair_team[pair_team['team'] == 'dire'].drop(columns='team').reset_index(drop=True)
    
        team_feature_cols = [col for col in df_radiant.columns if col != 'match_id']
    
        if use_diff:
            df_match = df_radiant.merge(df_dire, how='inner', on='match_id', suffixes=('_radiant', '_dire'))
            diff_cols = {'match_id': df_match['match_id']}
            for col in team_feature_cols:
                diff_cols[f'{col}_diff'] = df_match[f'{col}_radiant'] - df_match[f'{col}_dire']
            df_match = pd.DataFrame(diff_cols)
        else:
            df_radiant = df_radiant.rename(columns={col: f'{col}_radiant' for col in team_feature_cols})
            df_dire = df_dire.rename(columns={col: f'{col}_dire' for col in team_feature_cols})
            df_match = df_radiant.merge(df_dire, how='inner', on='match_id')
    
        train_ids = self.matches_train[['match_id']].copy()
        test_ids = self.matches_test[['match_id']].copy()
    
        train_block = train_ids.merge(df_match, how='left', on='match_id')
        test_block = test_ids.merge(df_match, how='left', on='match_id')
    
        return train_block, test_block

    def _make_players_features(self, use_diff: bool = True, alpha: float = 20.0):
        df = self.players_info.copy()
    
        train_match_ids = self.matches_train['match_id'].unique()
        test_match_ids = self.matches_test['match_id'].unique()
        all_match_ids = np.union1d(train_match_ids, test_match_ids)
    
        df_hist = df[df['match_id'].isin(train_match_ids)].copy()
        df_all = df[df['match_id'].isin(all_match_ids)].copy()
    
        df_hist = df_hist.merge(self.matches_train[['match_id', 'radiant_win']], how='left', on='match_id')
        df_hist['player_win'] = np.where(df_hist['player_slot'].isin(self.radiant_slots), df_hist['radiant_win'], 1 - df_hist['radiant_win'])
    
        strange_ids = {-1, 4294967295}
        normal_mask = df_hist['account_id'].notna() & (~df_hist['account_id'].isin(strange_ids))
        df_hist_normal = df_hist[normal_mask].copy()
    
        global_wr = df_hist_normal['player_win'].mean()
    
        player_agg = df_hist_normal.groupby('account_id')['player_win'].agg(['sum', 'count']).reset_index()
        player_agg = player_agg.rename(columns={'sum': 'player_wins_sum', 'count': 'player_n_matches'})
        player_agg['player_win_rate_smoothed'] = (player_agg['player_wins_sum'] + alpha * global_wr) / (player_agg['player_n_matches'] + alpha)
        player_agg['player_has_history'] = 1
        player_agg = player_agg[['account_id', 'player_win_rate_smoothed', 'player_n_matches', 'player_has_history']]
    
        df_all = df_all.merge(player_agg, how='left', on='account_id')
    
        df_all['player_win_rate_smoothed'] = df_all['player_win_rate_smoothed'].fillna(global_wr)
        df_all['player_n_matches'] = df_all['player_n_matches'].fillna(0)
        df_all['player_has_history'] = df_all['player_has_history'].fillna(0)
    
        df_all['account_id_is_4294967295'] = (df_all['account_id'] == 4294967295).astype(int)
        df_all['account_id_is_minus1'] = (df_all['account_id'] == -1).astype(int)
        df_all['account_id_is_strange'] = ((df_all['account_id'] == 4294967295) | (df_all['account_id'] == -1)).astype(int)
    
        player_feature_cols = ['player_win_rate_smoothed', 'player_n_matches', 'player_has_history']
    
        df_radiant = df_all[df_all['player_slot'].isin(self.radiant_slots)]
        df_dire = df_all[df_all['player_slot'].isin(self.dire_slots)]
    
        df_radiant_stats = df_radiant.groupby('match_id')[player_feature_cols].mean().reset_index()
        df_dire_stats = df_dire.groupby('match_id')[player_feature_cols].mean().reset_index()
    
        df_radiant_strange = df_radiant.groupby('match_id')[['account_id_is_4294967295', 'account_id_is_minus1', 'account_id_is_strange']].sum().reset_index()
        df_dire_strange = df_dire.groupby('match_id')[['account_id_is_4294967295', 'account_id_is_minus1', 'account_id_is_strange']].sum().reset_index()
    
        df_radiant_strange = df_radiant_strange.rename(columns={'account_id_is_4294967295': 'account_id_4294967295_count_radiant', 'account_id_is_minus1': 'account_id_minus1_count_radiant', 'account_id_is_strange': 'account_id_strange_count_radiant'})
        df_dire_strange = df_dire_strange.rename(columns={'account_id_is_4294967295': 'account_id_4294967295_count_dire', 'account_id_is_minus1': 'account_id_minus1_count_dire', 'account_id_is_strange': 'account_id_strange_count_dire'})
    
        if use_diff:
            df_match = df_radiant_stats.merge(df_dire_stats, how='inner', on='match_id', suffixes=('_radiant', '_dire'))
            diff_cols = {'match_id': df_match['match_id']}
            for col in player_feature_cols:
                diff_cols[f'{col}_diff'] = df_match[f'{col}_radiant'] - df_match[f'{col}_dire']
    
            df_strange = df_radiant_strange.merge(df_dire_strange, how='inner', on='match_id')
            diff_cols['account_id_4294967295_count_radiant'] = df_strange['account_id_4294967295_count_radiant']
            diff_cols['account_id_4294967295_count_dire'] = df_strange['account_id_4294967295_count_dire']
            diff_cols['account_id_minus1_count_radiant'] = df_strange['account_id_minus1_count_radiant']
            diff_cols['account_id_minus1_count_dire'] = df_strange['account_id_minus1_count_dire']
            diff_cols['account_id_strange_count_radiant'] = df_strange['account_id_strange_count_radiant']
            diff_cols['account_id_strange_count_dire'] = df_strange['account_id_strange_count_dire']
            diff_cols['account_id_4294967295_count_diff'] = df_strange['account_id_4294967295_count_radiant'] - df_strange['account_id_4294967295_count_dire']
            diff_cols['account_id_minus1_count_diff'] = df_strange['account_id_minus1_count_radiant'] - df_strange['account_id_minus1_count_dire']
            diff_cols['account_id_strange_count_diff'] = df_strange['account_id_strange_count_radiant'] - df_strange['account_id_strange_count_dire']
            df_match = pd.DataFrame(diff_cols)
        else:
            df_radiant_stats = df_radiant_stats.rename(columns={col: f'{col}_radiant' for col in player_feature_cols})
            df_dire_stats = df_dire_stats.rename(columns={col: f'{col}_dire' for col in player_feature_cols})
            df_match = df_radiant_stats.merge(df_dire_stats, how='inner', on='match_id')
            df_match = df_match.merge(df_radiant_strange, how='left', on='match_id')
            df_match = df_match.merge(df_dire_strange, how='left', on='match_id')
    
        train_ids = self.matches_train[['match_id']].copy()
        test_ids = self.matches_test[['match_id']].copy()
    
        train_block = train_ids.merge(df_match, how='left', on='match_id')
        test_block = test_ids.merge(df_match, how='left', on='match_id')
    
        return train_block, test_block

    def build_train_test(
        self,
        use_base=True,
        use_game_mode=True,
        use_time=True,
        use_region=True,
        use_heroes=True,
        use_team_comp=True,
        use_hero_stat_std=True,
        use_hero_stat_mean=True,
        use_advantages_agg=True,
        use_hero_match_stat=True,
        use_advantages_trend=True,
        use_hero_winrate=True,
        adv_trend_method='ols',
        use_players=False, # СУПЕР руина
        use_players_diff=False,
        use_adv_bins=False,
        adv_bin_cols=None,
        adv_n_bins=3,
        adv_bin_strategy='uniform',
        adv_bins_keep_original=True,
        use_chat_stats=True,
        use_chat_slang=True,
        return_chat_sparse=True,
        chat_max_features=None,
        chat_min_df=5,
        chat_ngram_range=(1, 2),
        use_chat_char=False, # руина
        chat_char_max_features=3000,
        chat_char_min_df=30,
        chat_char_ngram_range=(2, 5),
        use_hero_pair_synergy=False,
        use_hero_pair_synergy_diff=True,
        target_col='radiant_win'
    ):
        
        train_blocks = []
        test_blocks = []
    
        if use_base:
            tr, te = self._make_base_match_features()
            train_blocks.append(tr)
            test_blocks.append(te)
    
        if use_game_mode:
            tr, te = self._make_game_mode_features()
            train_blocks.append(tr)
            test_blocks.append(te)
    
        if use_time:
            tr, te = self._make_time_features_for_matches()
            train_blocks.append(tr)
            test_blocks.append(te)
    
        if use_region:
            tr, te = self._make_regions_features()
            train_blocks.append(tr)
            test_blocks.append(te)

        if use_heroes:
            tr, te = self._make_heroes_features()
            train_blocks.append(tr)
            test_blocks.append(te)

        if use_team_comp:
            tr, te = self._make_team_composotion_features()
            train_blocks.append(tr)
            test_blocks.append(te)

        if use_hero_stat_std:
            tr, te = self._make_hero_stat_std_features()
            train_blocks.append(tr)
            test_blocks.append(te)

        if use_hero_stat_mean:
            tr, te = self._make_hero_stat_mean_features()
            train_blocks.append(tr)
            test_blocks.append(te)

        if use_advantages_agg:
            tr, te = self._make_advantages_agg_features()
            train_blocks.append(tr)
            test_blocks.append(te)

        if use_advantages_trend:
            tr, te = self._make_advantages_trend_features(method=adv_trend_method)
            train_blocks.append(tr)
            test_blocks.append(te)

        if use_hero_match_stat:
            tr, te = self._make_hero_match_stat_features()
            train_blocks.append(tr)
            test_blocks.append(te)

        if use_players:
            tr, te = self._make_players_features(use_diff=use_players_diff)
            train_blocks.append(tr)
            test_blocks.append(te)

        if use_hero_winrate:
            tr, te = self._make_hero_winrate_features()
            train_blocks.append(tr)
            test_blocks.append(te)

        if use_chat_stats:
            tr, te = self._make_chat_stats_features()
            train_blocks.append(tr)
            test_blocks.append(te)

        if use_chat_slang:
            tr, te = self._make_chat_slang_features()
            train_blocks.append(tr)
            test_blocks.append(te)

        if use_hero_pair_synergy:
            tr, te = self._make_hero_pair_synergy_features(use_diff=use_hero_pair_synergy_diff)
            train_blocks.append(tr)
            test_blocks.append(te)

        base_train = self.matches_train[['match_id']].copy()
        base_test = self.matches_test[['match_id']].copy()
    
        X_train, X_test = self._merge_feature_blocks(base_train, base_test, train_blocks, test_blocks)

        if use_adv_bins:
            if adv_bin_cols is None:
                adv_bin_cols = self.adv_bin_cols_default
        
            X_train, X_test = self._add_binned_features(
                X_train=X_train,
                X_test=X_test,
                cols=adv_bin_cols,
                n_bins=adv_n_bins,
                strategy=adv_bin_strategy,
                keep_original=adv_bins_keep_original
            )

        # Заполненение пропусков в X_train/X_test
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        y_train = self.matches_train[target_col].astype(int).to_numpy()

        if not return_chat_sparse:
            return X_train, y_train, X_test

        X_train_chat, X_test_chat = self._make_chat_sparse_features(
            chat_max_features=chat_max_features,
            chat_min_df=chat_min_df,
            chat_ngram_range=chat_ngram_range,
            use_chat_char=use_chat_char,
            chat_char_max_features=chat_char_max_features,
            chat_char_min_df=chat_char_min_df,
            chat_char_ngram_range=chat_char_ngram_range
        )
    
        return X_train, y_train, X_test, X_train_chat, X_test_chat
