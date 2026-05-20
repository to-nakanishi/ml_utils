"""Composite feature generation from multiple numeric columns."""

import warnings

import numpy as np
import pandas as pd


def make_composite_features(
    df: pd.DataFrame,
    cols: list[str],
    prefix: str = 'COMPOSITE',
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """
    ----------
    summary
    ----------
    複数の数値列から平均系の合成特徴量を生成する関数。
    算術平均・標準偏差・幾何平均を生成し、weights 指定時は加重平均も追加する。
    標準偏差は、平均だけでは捉えられない列間のばらつきをモデルに与える補助特徴量。
    入力 df は変更せず、合成列を追加した新しい DataFrame を返す(copy 方式)。

    欠損値(NaN)や 0 は補完せず、そのまま計算する(fail loud 方針)。
    NaN を含む行は全合成列が NaN になり、0 を含む行は幾何平均が 0 に潰れる。
    該当行があれば計算前に警告するため、事前の補完を推奨する。

    ----------
    Parameters
    ----------
    df : pd.DataFrame
        入力 DataFrame。`cols` で指定する列を含む。
    cols : list[str]
        合成元の数値列名のリスト(例: ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'])。
    prefix : str, default='COMPOSITE'
        生成する合成列名の接頭辞。
    weights : list[float], optional
        加重平均の重み。`cols` と同じ長さが必要。None の場合、加重平均列は生成しない。

    ----------
    Returns
    ----------
    pd.DataFrame
        以下の合成列を追加した新しい DataFrame:
        - {prefix}_MEAN: 算術平均(NaN を含む行は NaN)
        - {prefix}_STD: 標準偏差(NaN を含む行は NaN)
        - {prefix}_GEOM_MEAN: 幾何平均(NaN 行は NaN、0 行は 0)
        - {prefix}_WEIGHTED: 加重平均(weights 指定時のみ、NaN を含む行は NaN)

    ----------
    Examples
    ----------
    >>> cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    >>> df_out = make_composite_features(df, cols, prefix='EXT_SOURCES')
    >>> df_out.columns
    [..., 'EXT_SOURCES_MEAN', 'EXT_SOURCES_STD', 'EXT_SOURCES_GEOM_MEAN']
    """
    if not cols:
        raise ValueError("cols is empty.")
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in df.")
    if weights is not None and len(weights) != len(cols):
        raise ValueError(
            f"weights length ({len(weights)}) must match cols length ({len(cols)})."
        )

    df_res = df.copy()
    values = df_res[cols]

    # --- 警告(計算前にまとめて) ---
    nan_count = int(values.isna().any(axis=1).sum())
    if nan_count > 0:
        warnings.warn(
            f'{nan_count} rows contain NaN in {cols}; '
            f'all composite features will be NaN for these rows. '
            f'Consider imputing beforehand.'
        )
    zero_count = int((values == 0).any(axis=1).sum())
    if zero_count > 0:
        warnings.warn(
            f'{zero_count} rows contain 0 in {cols}; '
            f'geometric mean will be 0 for these rows. '
            f'Consider imputing beforehand.'
        )

    # --- 計算(skipna=False で NaN を残す。NaN を勝手に埋めない) ---
    df_res[f'{prefix}_MEAN'] = values.mean(axis=1, skipna=False)
    df_res[f'{prefix}_STD'] = values.std(axis=1, skipna=False)

    # 幾何平均: 積の (1/n) 乗。0/NaN はそのまま反映される。
    product = values.prod(axis=1, skipna=False)
    df_res[f'{prefix}_GEOM_MEAN'] = np.sign(product) * np.abs(product) ** (1.0 / len(cols))

    # 加重平均(weights 指定時のみ)
    if weights is not None:
        weights_arr = np.array(weights, dtype=np.float64)
        weighted_sum = (values * weights_arr).sum(axis=1, skipna=False)
        df_res[f'{prefix}_WEIGHTED'] = weighted_sum / weights_arr.sum()

    return df_res
