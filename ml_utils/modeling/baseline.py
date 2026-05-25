"""Run a baseline trial (LGBM / CatBoost) with cross-validated OOF AUC."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

_DEFAULT_LGB = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'n_estimators': 200,
    'random_state': 42,
    'verbosity': -1,
}

_DEFAULT_CAT = {
    'iterations': 200,
    'depth': 5,
    'learning_rate': 0.05,
    'eval_metric': 'AUC',
    'random_seed': 42,
    'logging_level': 'Silent',
}


def run_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    models: tuple = ('lgb', 'cat'),
    n_splits: int = 5,
    cv=None,
    lgb_params: dict | None = None,
    cat_params: dict | None = None,
    return_models: bool = True,
) -> dict:
    """
    ----------
    summary
    ----------
    クリーニング後・FE後などの節目で、現在の特徴量セットのベースライン性能を
    LightGBM / CatBoost で確認する試走関数。StratifiedKFold の OOF AUC と
    5fold 平均の特徴量重要度を算出し、両モデルの比較表を返す。

    「FE が効いたか」を相対比較する用途を想定し、節目で 1 回ずつ回す重め・
    多機能な試走として設計している(連打用の軽量版は対象外)。

    本関数は2値分類専用。objective は params で上書きできるが、AUC 計算等が
    2値前提のため、多クラス分類には対応しない。

    CatBoost のカテゴリ列は select_dtypes(exclude='number')で判定し、欠損は
    'Unknown' 文字列に変換してから cat_features に渡す(fold ループ外で1回処理。
    target 非依存のためリークしない)。この変換は CatBoost 用のコピーに対して
    のみ行い、元の X は変更しない。LGBM は NaN をそのまま扱うため変換しない。

    AUC は全 fold の検証予測を集めた OOF 予測全体で 1 つ算出する(fold 平均では
    なく、分割運に左右されにくい OOF 値)。重要度は fold ごとの値を平均する。

    return_models=True のとき、CV とは別に全データで 1 回学習したモデルを返す
    (SHAP 分析等に使用。CV の fold モデルではなく全データ学習モデル)。

    ----------
    Parameters
    ----------
    X : pd.DataFrame
        特徴量。SK_ID 等の識別子・TARGET は事前に除外しておくこと。
    y : pd.Series
        目的変数(0/1 の二値)。
    models : tuple, default=('lgb', 'cat')
        回すモデル。'lgb' / 'cat' を含むタプル。片方だけも可。
        回さなかったモデルの結果は None。
    n_splits : int, default=5
        StratifiedKFold の分割数(cv=None のときに使用)。
    cv : optional
        CV 分割オブジェクト。None なら StratifiedKFold(n_splits, shuffle=True)。
        時系列データでは TimeSeriesSplit 等を渡してリークを避ける。
    lgb_params : dict or None, default=None
        LGBM パラメータ。デフォルトに対する部分上書き(マージ)。
    cat_params : dict or None, default=None
        CatBoost パラメータ。デフォルトに対する部分上書き(マージ)。
    return_models : bool, default=True
        True なら全データで学習したモデルも返す(SHAP 用)。False なら None。

    ----------
    Returns
    ----------
    dict
        lgb_auc / cat_auc            : OOF AUC(回さないモデルは None)
        lgb_importance / cat_importance : feature と重要度(%)の DataFrame、
                                           5fold 平均(回さないモデルは None)
        comparison                   : 両重要度をマージした比較表
                                       (両モデルを回したときのみ。片方だと None)
        lgb_model / cat_model        : 全データ学習モデル
                                       (return_models=False or 回さないとき None)

    ----------
    Examples
    ----------
    >>> res = run_baseline(X, y)
    >>> res['lgb_auc'], res['cat_auc']
    (0.7785, 0.7821)
    >>> res['comparison'].head()
    """
    import catboost as cb
    import lightgbm as lgb
    
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    if len(X) == 0:
        raise ValueError("X is empty.")
    valid_models = {'lgb', 'cat'}
    if not set(models) <= valid_models:
        raise ValueError(f"models must be a subset of {valid_models}, got {models}")

    lgb_p = {**_DEFAULT_LGB, **(lgb_params or {})}
    cat_p = {**_DEFAULT_CAT, **(cat_params or {})}

    splitter = cv if cv is not None else StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=42
    )

    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)

    # CatBoost 用カテゴリ列の準備(fold ループ外で1回。target 非依存。元X不変)
    cat_features = X.select_dtypes(exclude='number').columns.tolist()
    if 'cat' in models and cat_features:
        X_cat = X.copy()
        for col in cat_features:
            X_cat[col] = (
                X_cat[col].astype(str).replace('nan', 'Unknown').fillna('Unknown')
            )
    else:
        X_cat = X

    result = {
        'lgb_auc': None, 'cat_auc': None,
        'lgb_importance': None, 'cat_importance': None,
        'comparison': None, 'lgb_model': None, 'cat_model': None,
    }

    # --- LightGBM ---
    if 'lgb' in models:
        oof = np.zeros(len(X))
        imp_acc = np.zeros(X.shape[1])
        n_folds = 0
        for tr_idx, va_idx in splitter.split(X, y):
            model = lgb.LGBMClassifier(**lgb_p)
            model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
                callbacks=[lgb.log_evaluation(period=0),
                           lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
            imp_acc += model.feature_importances_
            n_folds += 1
        result['lgb_auc'] = roc_auc_score(y, oof)
        imp_mean = imp_acc / n_folds
        result['lgb_importance'] = pd.DataFrame({
            'feature': X.columns,
            'LGBM_pct': 100 * imp_mean / imp_mean.sum(),
        })
        if return_models:
            m = lgb.LGBMClassifier(**lgb_p)
            m.fit(X, y, callbacks=[lgb.log_evaluation(period=0)])
            result['lgb_model'] = m

    # --- CatBoost ---
    if 'cat' in models:
        oof = np.zeros(len(X))
        imp_acc = np.zeros(X_cat.shape[1])
        n_folds = 0
        for tr_idx, va_idx in splitter.split(X_cat, y):
            model = cb.CatBoostClassifier(cat_features=cat_features, **cat_p)
            model.fit(
                X_cat.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=(X_cat.iloc[va_idx], y.iloc[va_idx]),
                early_stopping_rounds=50,
            )
            oof[va_idx] = model.predict_proba(X_cat.iloc[va_idx])[:, 1]
            imp_acc += model.get_feature_importance()
            n_folds += 1
        result['cat_auc'] = roc_auc_score(y, oof)
        imp_mean = imp_acc / n_folds
        result['cat_importance'] = pd.DataFrame({
            'feature': X_cat.columns,
            'CAT_pct': 100 * imp_mean / imp_mean.sum(),
        })
        if return_models:
            m = cb.CatBoostClassifier(cat_features=cat_features, **cat_p)
            m.fit(X_cat, y)
            result['cat_model'] = m

    # --- 比較表(両モデルを回したときのみ) ---
    if result['lgb_importance'] is not None and result['cat_importance'] is not None:
        result['comparison'] = pd.merge(
            result['lgb_importance'], result['cat_importance'], on='feature'
        ).sort_values('LGBM_pct', ascending=False).reset_index(drop=True)

    return result
