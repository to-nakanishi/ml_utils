"""Target encoding with out-of-fold strategy."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def target_encode_oof(
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
    target: str = 'TARGET',
    cv: Any = None,
    smoothing: float = 10.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    ----------
    summary
    ----------
    ターゲットエンコーディング及びリーク防止(OOF＋Smoothing)
    を行う関数。
    train に対しては cross-validation の各 fold で
    validation 部分を除いた残りから計算した値を埋め込むことで
    target からのリークを防ぐ。test に対しては train 全体から
    計算した値を使う。未知カテゴリは global mean で補完される。
    
    ----------
    Parameters
    ----------
    train : pd.DataFrame
        学習用データ。`col` と `target` を含む。
    test : pd.DataFrame
        テスト用データ。`col` を含む。`target` は不要。
    col : str
        エンコード対象のカテゴリ列名。
    target : str, default='TARGET'
        目的変数の列名。
    cv : sklearn cross-validator, optional
        scikit-learn 互換の CV オブジェクト。
        None の場合は StratifiedKFold(n_splits=5, shuffle=True, random_state=42)。
    smoothing : float, default=10.0
        スムージング係数。大きいほど global mean に寄る。
    verbose : bool, default=True
        True の場合、エンコード完了時にメッセージを表示。
        
    ----------
    Returns
    ----------
    tuple[pd.DataFrame, pd.DataFrame]
        エンコード列 `{col}_TARGET_RATE` (float32) を追加した
        (train_encoded, test_encoded)。入力 DataFrame は変更されない。

    ----------
    Examples
    ----------
    >>> train_enc, test_enc = target_encode_oof(train, test, 'ORGANIZATION_TYPE')
    Encoded: ORGANIZATION_TYPE_TARGET_RATE
    """
    
    if col not in train.columns:
        raise KeyError(f"Column '{col}' not found in train.")
    if col not in test.columns:
        raise KeyError(f"Column '{col}' not found in test.")
    if target not in train.columns:
        raise KeyError(f"Target column '{target}' not found in train.")
    if len(train) == 0:
        raise ValueError("train is empty.")

    train_res = train.copy()
    test_res = test.copy()

    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    new_col = f'{col}_TARGET_RATE'
    global_mean = train_res[target].mean()

    oof_values = np.zeros(len(train_res), dtype=np.float64)

    for train_idx, val_idx in cv.split(train_res, train_res[target]):
        df_t = train_res.iloc[train_idx]
        df_v = train_res.iloc[val_idx]

        agg = df_t.groupby(col, dropna=False)[target].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']
        smooth = (means * counts + global_mean * smoothing) / (counts + smoothing)

        oof_values[val_idx] = df_v[col].map(smooth).values

    oof_values = np.where(np.isnan(oof_values), global_mean, oof_values)
    train_res[new_col] = oof_values.astype(np.float32)

    agg = train_res.groupby(col, dropna=False)[target].agg(['count', 'mean'])
    smooth = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)
    test_values = test_res[col].map(smooth).values.astype(np.float64)
    test_values = np.where(np.isnan(test_values), global_mean, test_values)
    test_res[new_col] = test_values.astype(np.float32)

    if verbose:
        print(f'Encoded: {new_col}')

    return train_res, test_res
