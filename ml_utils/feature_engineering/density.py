"""Find the score at which TARGET-class densities cross (risk-reversal border)."""

import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


def find_density_crossover(
    df: pd.DataFrame,
    columns: list[str],
    target: str = 'TARGET',
    bandwidth: float = 0.03,
    grid_size: int = 1000,
    x_min: float | None = None,
    x_max: float | None = None,
) -> dict[str, float]:
    """
    ----------
    summary
    ----------
    TARGET 別の密度(KDE)が交差するスコアを列ごとに算出する関数。
    各列について target=0 と target=1 の分布を KDE で推定し、密度が逆転する
    点(=リスクの向きが反転する境界)を返す。本関数は交点が1個であることを
    前提とする(複数生じる列は警告し、暫定的に最も右側の交点を返す)。

    【重要・使用前に必ず確認】
    使用前に各列で TARGET 別の KDE 分布を可視化し、密度が明確に1点で交差して
    いること・bandwidth が適切(過剰に滑らか/ギザギザでない)ことを確認すること。
    交差が不明瞭な列や交点が複数生じる列に対する結果は信頼できない。

    【返した交点でフラグを生成する際の欠損の扱い】
    フラグ生成 `(df[col] < crossover)` では、NaN 行は False と評価されて
    自動的に 0(境界以上=低リスク側)に振り分けられる。欠損が多い列では
    フラグが片側に偏り歪むため、フラグ生成の前に impute_by_target_rate 等で
    欠損を補完しておくことを推奨する(補完せず「欠損=低リスク」と扱う設計も
    可能だが、その場合は意図的にそう決めること)。
    なお本関数内の交点算出では NaN は KDE 推定前に除外する(計算に関与しない)。
    補完が必要なのは後段のフラグ生成であり、交点算出自体は補完不要。

    返した交点はそのまま閾値フラグ生成に渡せる:
    例) df[f'{col}_BORDER'] = (df[col] < crossover[col]).astype('int8')

    この関数は target を参照するため、交点は train のみで算出し、その値を
    valid/test に適用すること(leakage 回避)。本関数は「与えられた df から
    交点を出す」算出のみを担い、フラグ生成や train/test への適用は呼び出し側の
    責務とする(fit/transform の fit に相当)。

    ----------
    Parameters
    ----------
    df : pd.DataFrame
        入力データ。`columns` と `target` を含む。通常は train を渡す。
    columns : list[str]
        交点を算出する数値列名のリスト。
    target : str, default='TARGET'
        目的変数の列名。0/1 の二値を想定。
    bandwidth : float, default=0.03
        KDE のバンド幅。大きいほど滑らか(交点が消えやすい)、小さいほど
        ギザギザ(偽の交点が増える)。列のスケールに応じて調整する。
        既定値 0.03 は 0~1 スケールの EXT_SOURCE 系を想定。
    grid_size : int, default=1000
        密度を評価するグリッドの分割数。範囲が広い列ほど大きい値が有効。
    x_min : float or None, default=None
        グリッドの下限。None の場合、対象列の最小値を使用する。
    x_max : float or None, default=None
        グリッドの上限。None の場合、対象列の最大値を使用する。

    ----------
    Returns
    ----------
    dict[str, float]
        {列名: 交点スコア}。交点が見つからない列・片方のクラスにサンプルが
        無い列は含まれない。

    ----------
    Examples
    ----------
    >>> borders = find_density_crossover(
    ...     train, ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    ... )
    EXT_SOURCE_1: 0.4284
    EXT_SOURCE_2: 0.4895
    EXT_SOURCE_3: 0.4484
    >>> train['EXT_SOURCE_3_BORDER'] = (
    ...     train['EXT_SOURCE_3'] < borders['EXT_SOURCE_3']
    ... ).astype('int8')
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in df.")
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in df: {missing_cols}")
    if len(df) == 0:
        raise ValueError("df is empty.")

    crossovers: dict[str, float] = {}

    for col in columns:
        mask = df[col].notna()
        s0 = df.loc[mask & (df[target] == 0), col].values
        s1 = df.loc[mask & (df[target] == 1), col].values

        # どちらかのクラスに非欠損サンプルが無い → KDE 不可
        if len(s0) == 0 or len(s1) == 0:
            warnings.warn(
                f"'{col}': one of the target classes has no non-NaN samples; skipped."
            )
            continue

        # グリッド範囲(指定が無ければ列の min/max)
        lo = x_min if x_min is not None else df.loc[mask, col].min()
        hi = x_max if x_max is not None else df.loc[mask, col].max()
        x_grid = np.linspace(lo, hi, grid_size).reshape(-1, 1)

        kde0 = KernelDensity(bandwidth=bandwidth).fit(s0.reshape(-1, 1))
        kde1 = KernelDensity(bandwidth=bandwidth).fit(s1.reshape(-1, 1))
        dens0 = np.exp(kde0.score_samples(x_grid))
        dens1 = np.exp(kde1.score_samples(x_grid))

        # 密度差の符号反転点 = 交点
        diff = dens0 - dens1
        idx = np.argwhere(np.diff(np.sign(diff))).flatten()

        if len(idx) == 0:
            warnings.warn(f"'{col}': no crossover found; skipped.")
            continue

        # 本関数は交点が1個であることを前提とする。複数生じるのは分布が単純に
        # 分離していない or bandwidth が不適切なサイン。境界の選び方から設計を
        # 見直す必要があるため、警告したうえで暫定的に最も右側の交点を返す。
        if len(idx) > 1:
            warnings.warn(
                f"'{col}': {len(idx)} crossovers found (expected 1). "
                f"Distributions may not separate cleanly or bandwidth may be off. "
                f"Verify the KDE plot; using the rightmost crossover provisionally."
            )

        crossover = float(x_grid[idx[-1]][0])
        crossovers[col] = crossover
        print(f"{col}: {crossover:.4f}")

    return crossovers
