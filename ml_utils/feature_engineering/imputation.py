"""Impute missing values using target-rate matched bins."""

import warnings

import numpy as np
import pandas as pd


def impute_by_target_rate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
    target: str = 'TARGET',
    n_bins: int = 10,
    suffix: str = '_IMPUTED',
    add_missing_flag: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ----------
    summary
    ----------
    欠損値(NaN)を「デフォルト率が近いビンの代表値」で補完する関数。
    train の非欠損値を等頻度ビン(qcut)に分割し、各ビンのデフォルト率
    (target 平均)を計算する。欠損行のデフォルト率に最も近いビンを選び、
    その代表値(中央値)で train/test の欠損を補完する。

    元の列は変更せず、補完済みの新列 `{col}{suffix}` を追加する。
    補完値は train から決定し、test にも同じ値を適用する(leakage 回避)。
    0 はこの関数では補完対象としない(欠損扱いしたい場合は事前に NaN へ変換)。
    欠損が0件、またはビンが作れない場合は列を作らず、メッセージを表示する。

    ----------
    Parameters
    ----------
    train : pd.DataFrame
        学習用データ。`col` と `target` を含む。
    test : pd.DataFrame
        テスト用データ。`col` を含む。`target` は不要。
    col : str
        補完対象の数値列名。
    target : str, default='TARGET'
        目的変数の列名。
    n_bins : int, default=10
        qcut の分割数。データが大きいほど大きい値が有効(Home Credit では 50 を使用)。
    suffix : str, default='_IMPUTED'
        補完済み列名の接尾辞。
    add_missing_flag : bool, default=False
        True の場合、欠損だった行を示す `{col}_WAS_MISSING` 列(0/1)を追加する。

    ----------
    Returns
    ----------
    tuple[pd.DataFrame, pd.DataFrame]
        補完済み列を追加した (train, test)。入力 DataFrame は変更されない。
        欠損が0件、またはビンが作れない場合は列を作成しない。

    ----------
    Examples
    ----------
    >>> train_imp, test_imp = impute_by_target_rate(train, test, 'EXT_SOURCE_1')
    >>> 'EXT_SOURCE_1_IMPUTED' in train_imp.columns
    True
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

    new_col = f'{col}{suffix}'
    flag_col = f'{col}_WAS_MISSING'

    train_missing = train_res[col].isna()
    test_missing = test_res[col].isna()

    # 欠損が0件 → DF を一切いじらず、補足メッセージのみ
    if train_missing.sum() == 0 and test_missing.sum() == 0:
        print(f"'{col}': no missing values; no columns created.")
        return train_res, test_res

    # 非欠損値が無い → ビンが作れない → 何もせず警告
    non_missing = train_res.loc[~train_missing, col]
    if len(non_missing) == 0:
        warnings.warn(
            f"'{col}': all values are NaN in train; cannot build bins. "
            f"No columns created."
        )
        return train_res, test_res

    # --- ここから補完(欠損あり & ビン作成可能なときだけ) ---

    # 欠損フラグ列(オプション)
    if add_missing_flag:
        train_res[flag_col] = train_missing.astype(np.int8)
        test_res[flag_col] = test_missing.astype(np.int8)

    # 非欠損値を等頻度ビンに分割(境界重複は drop で回避)
    bins = pd.qcut(non_missing, q=n_bins, duplicates='drop')
    bin_target = train_res.loc[~train_missing].groupby(bins, observed=True)[target].mean()
    bin_median = non_missing.groupby(bins, observed=True).median()

    # 欠損グループのデフォルト率
    if train_missing.sum() > 0:
        missing_rate = train_res.loc[train_missing, target].mean()
    else:
        # train に欠損は無いが test にある場合 → 全体平均で代替
        missing_rate = train_res[target].mean()

    # 欠損グループのデフォルト率に最も近いビンを選ぶ
    nearest_bin = (bin_target - missing_rate).abs().idxmin()
    impute_value = bin_median.loc[nearest_bin]

    # 補完列を作成(元列コピー + NaN を impute_value で埋める)
    train_res[new_col] = train_res[col].fillna(impute_value)
    test_res[new_col] = test_res[col].fillna(impute_value)

    return train_res, test_res
