"""Target encoding with out-of-fold strategy."""

from collections.abc import Sequence
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

    本関数は **カテゴリ列** を対象とする。連続値を渡すとほぼ全行が
    一意グループとなり Smoothing で global mean に潰れて無意味になるため、
    数値列はビン化エンコード専用の bin_target_encode_oof を使うこと。

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


def bin_target_encode_oof(
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
    bins: int | Sequence[float],
    target: str = 'TARGET',
    cv: Any = None,
    smoothing: float = 10.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    ----------
    summary
    ----------
    連続値をビン化してから OOF＋Smoothing のターゲットエンコーディングを
    行う関数。target_encode_oof の数値版。数値列をそのまま target 率に
    エンコードすると一意グループが大量に出て潰れるため、先にビンへ変換し
    「そのビンのデフォルト率」をエンコード値とする。

    欠損 (NaN) は捨てずに **独立したビン** として扱い、欠損行自体の
    デフォルト率でエンコードする (欠損自体がシグナルを持つ列に対応)。
    ビン境界は train で確定し、test にも同じ境界を適用する
    (test を単独で切らないことで分布ズレを防ぐ)。

    カテゴリ列にはビン化せず target_encode_oof を使うこと。
    本関数は連続値のビン化エンコード専用。

    ----------
    Parameters
    ----------
    train : pd.DataFrame
        学習用データ。`col` と `target` を含む。
    test : pd.DataFrame
        テスト用データ。`col` を含む。`target` は不要。
    col : str
        エンコード対象の数値列名。
    bins : int | Sequence[float]
        ビンの指定。pd.cut にそのまま渡す。
        - int  : train の min〜max を等幅に bins 分割。境界は train で確定し
                 test に再適用する。外れ値がある列では等幅だと解像度が落ちる
                 ため、明示境界 (下記) を推奨。
        - 列   : 明示的なビン境界 (例: [-np.inf, 0, 1, 2, np.inf])。
                 外れ値は ±np.inf を端に置くことで最外ビンに吸収できる。
                 境界が値域を覆っていない場合、外れた値は欠損ビン扱いになる。
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
        エンコード列 `{col}_BIN_TARGET_RATE` (float32) を追加した
        (train_encoded, test_encoded)。入力 DataFrame は変更されない。
        作業用のビン列は内部で破棄される。

    ----------
    Examples
    ----------
    >>> # 明示境界 (外れ値を ±inf で吸収)
    >>> tr, te = bin_target_encode_oof(
    ...     train, test, 'CNT_CHILDREN', bins=[-np.inf, 0, 1, 2, np.inf]
    ... )
    Encoded: CNT_CHILDREN_BIN_TARGET_RATE
    >>> # 0〜1 スコアを 0.05 刻みで (刻み幅は呼び出し側で任意指定)
    >>> tr, te = bin_target_encode_oof(
    ...     train, test, 'EXT_SOURCES_CUSTOM_MEAN', bins=np.arange(0, 1.05, 0.05)
    ... )
    Encoded: EXT_SOURCES_CUSTOM_MEAN_BIN_TARGET_RATE
    """

    if col not in train.columns:
        raise KeyError(f"Column '{col}' not found in train.")
    if col not in test.columns:
        raise KeyError(f"Column '{col}' not found in test.")
    if target not in train.columns:
        raise KeyError(f"Target column '{target}' not found in train.")
    if len(train) == 0:
        raise ValueError("train is empty.")
    if isinstance(bins, (int, np.integer)) and not isinstance(bins, bool):
        if bins < 1:
            raise ValueError(f"bins as int must be >= 1, got {bins}.")
    elif isinstance(bins, Sequence) or isinstance(bins, np.ndarray):
        if len(bins) < 2:
            raise ValueError("bins as a sequence must have at least 2 edges.")
    else:
        raise TypeError(f"bins must be int or a sequence of edges, got {type(bins)}.")

    train_res = train.copy()
    test_res = test.copy()

    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    new_col = f'{col}_BIN_TARGET_RATE'
    bin_col = f'_{col}_bin'
    missing_label = -1  # 欠損 / どのビンにも入らなかった値を集約する番兵
    global_mean = train_res[target].mean()

    # --- ビン化: 境界は train で確定し test に再適用 ---
    if isinstance(bins, (int, np.integer)) and not isinstance(bins, bool):
        train_codes, edges = pd.cut(
            train_res[col], bins=bins, labels=False,
            include_lowest=True, retbins=True,
        )
    else:
        edges = bins
        train_codes = pd.cut(
            train_res[col], bins=edges, labels=False, include_lowest=True,
        )
    test_codes = pd.cut(
        test_res[col], bins=edges, labels=False, include_lowest=True,
    )

    # NaN (欠損 or 範囲外) を番兵に置換し、独立した1グループとして扱う
    train_res[bin_col] = (
        pd.Series(train_codes, index=train_res.index).fillna(missing_label)
    )
    test_res[bin_col] = (
        pd.Series(test_codes, index=test_res.index).fillna(missing_label)
    )

    # --- OOF (train) ---
    oof_values = np.zeros(len(train_res), dtype=np.float64)

    for train_idx, val_idx in cv.split(train_res, train_res[target]):
        df_t = train_res.iloc[train_idx]
        df_v = train_res.iloc[val_idx]

        agg = df_t.groupby(bin_col)[target].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']
        smooth = (means * counts + global_mean * smoothing) / (counts + smoothing)

        oof_values[val_idx] = df_v[bin_col].map(smooth).values

    oof_values = np.where(np.isnan(oof_values), global_mean, oof_values)
    train_res[new_col] = oof_values.astype(np.float32)

    # --- test (train 全体で計算) ---
    agg = train_res.groupby(bin_col)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    smooth = (means * counts + global_mean * smoothing) / (counts + smoothing)
    test_values = test_res[bin_col].map(smooth).values.astype(np.float64)
    test_values = np.where(np.isnan(test_values), global_mean, test_values)
    test_res[new_col] = test_values.astype(np.float32)

    # 作業用ビン列を破棄
    train_res = train_res.drop(columns=[bin_col])
    test_res = test_res.drop(columns=[bin_col])

    if verbose:
        print(f'Encoded: {new_col}')

    return train_res, test_res


def diagnose_target_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    original_col: str,
    encoded_col: str,
    target: str = 'TARGET',
    small_group_threshold: int = 10,
) -> dict:
    """
    ----------
    summary
    ----------
    target encoding の品質を診断する関数。
    エンコード済みの DataFrame に対し、リーク防止が機能しているか・
    過学習リスクがないかを示す指標を dict で返す。
    target_encode_oof と独立して動作し、再計算は行わない。

    ----------
    Parameters
    ----------
    train : pd.DataFrame
        エンコード済みの学習用データ。`original_col`, `encoded_col`, `target` を含む。
    test : pd.DataFrame
        エンコード済みのテスト用データ。`original_col`, `encoded_col` を含む。
    original_col : str
        エンコード対象だったカテゴリ列名(例: 'ORGANIZATION_TYPE')。
    encoded_col : str
        エンコード後の列名(例: 'ORGANIZATION_TYPE_TARGET_RATE')。
    target : str, default='TARGET'
        目的変数の列名。
    small_group_threshold : int, default=10
        小グループ判定の閾値。サンプル数がこの値未満のグループを過学習リスク
        ありとカウントする。

    ----------
    Returns
    ----------
    dict
        以下のキーを持つ診断レポート:
        - encoded_col (str): エンコード列名
        - groups (int): train 中のユニークカテゴリ数
        - small_groups (int): small_group_threshold 未満のグループ数
        - small_group_threshold (int): 入力された閾値
        - oof_mean (float): train のエンコード値の平均
        - global_mean (float): train の target の平均
        - test_unknown_count (int): test で train に存在しないカテゴリの行数
        - test_total (int): test の総行数
        - test_coverage (float): 1 - test_unknown_count / test_total

    ----------
    Examples
    ----------
    >>> train_enc, test_enc = target_encode_oof(train, test, 'ORGANIZATION_TYPE')
    >>> report = diagnose_target_encoding(
    ...     train_enc, test_enc,
    ...     'ORGANIZATION_TYPE', 'ORGANIZATION_TYPE_TARGET_RATE'
    ... )
    >>> print(report)
    {'encoded_col': 'ORGANIZATION_TYPE_TARGET_RATE', 'groups': 58, ...}
    """
    if original_col not in train.columns:
        raise KeyError(f"Column '{original_col}' not found in train.")
    if encoded_col not in train.columns:
        raise KeyError(f"Column '{encoded_col}' not found in train.")
    if original_col not in test.columns:
        raise KeyError(f"Column '{original_col}' not found in test.")
    if encoded_col not in test.columns:
        raise KeyError(f"Column '{encoded_col}' not found in test.")
    if target not in train.columns:
        raise KeyError(f"Target column '{target}' not found in train.")

    # グループ数とサンプル数の集計
    group_counts = train.groupby(original_col, dropna=False).size()
    groups = len(group_counts)
    small_groups = int((group_counts < small_group_threshold).sum())

    # エンコード値の平均 vs target の平均(リーク疑い検出)
    oof_mean = float(train[encoded_col].mean())
    global_mean = float(train[target].mean())

    # test の未知カテゴリ数
    train_categories = set(train[original_col].unique())
    test_unknown_mask = ~test[original_col].isin(train_categories)
    test_unknown_count = int(test_unknown_mask.sum())
    test_total = len(test)
    test_coverage = 1.0 - (test_unknown_count / test_total) if test_total > 0 else 1.0

    return {
        'encoded_col': encoded_col,
        'groups': groups,
        'small_groups': small_groups,
        'small_group_threshold': small_group_threshold,
        'oof_mean': oof_mean,
        'global_mean': global_mean,
        'test_unknown_count': test_unknown_count,
        'test_total': test_total,
        'test_coverage': test_coverage,
    }
