"""Time feature engineering from an elapsed-seconds column."""

import warnings

import numpy as np
import pandas as pd


def add_time_features(
    df: pd.DataFrame,
    col: str,
    cycles: dict[str, float] | None = None,
    linear: dict[str, float] | None = None,
    origin: float = 0.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    ----------
    summary
    ----------
    「基準時刻からの経過秒」を表す1列から、時間特徴量を生成する汎用関数。
    軸を2種類に分けて指定する:

    - cycles : 周期軸(時刻・曜日・月内の位置など)。
               周期で巻き取り sin/cos の2列にエンコードする。
               23時→0時、日→月のような境界の連続性を保持できる。
    - linear : 増え続ける軸(経過日数・通算何ヶ月目など)。
               origin からの経過量をそのまま1列で出す。

    周期の長さ・単位・基準点はすべて呼び出し側が秒で指定する。
    本関数は暦やドメイン固有の値(30日月など)を一切持たない。
    同じ「月」でも、月内の位置が欲しければ cycles に、通算月数が
    欲しければ linear に入れる、という振り分けは呼び出し側の判断。

    リーク制御は不要(全て行単位の変換)なので train/test を分けず、
    それぞれに同じ origin を渡して個別に適用すればよい。

    ----------
    Parameters
    ----------
    df : pd.DataFrame
        `col` を含むデータ。
    col : str
        基準時刻からの経過秒を表す数値列名(例: 'TransactionDT')。
    cycles : dict[str, float], optional
        {出力名: 周期(秒)}。各軸につき `{名前}_sin` / `{名前}_cos` を生成。
        例: {'hour': 86400, 'week': 86400*7, 'month': 86400*30}
        (月内の位置を周期として見たい場合は month を cycles に置く)
    linear : dict[str, float], optional
        {出力名: 単位(秒)}。各軸につき `{名前}` 列を生成。
        値は (col - origin) / 単位 で算出。
        例: {'day': 86400, 'month': 86400*30}
    origin : float, default=0.0
        linear の起点(秒)。既定 0 は経過秒をそのまま使う。
        train 起点に揃えたい場合は train[col].min() を train/test 双方に
        同じ値で渡す(test 単独で min を取らないことで分布ズレを防ぐ)。
        cycles には影響しない(周期は % で巻き取るため起点に依存しない)。
    verbose : bool, default=True
        True の場合、生成した列名を表示。

    ----------
    Returns
    ----------
    pd.DataFrame
        時間特徴量を追加した df のコピー。入力 df は変更されない。
        `col` の欠損行に対応する出力は NaN になる。

    ----------
    Examples
    ----------
    >>> # 24h周期・週周期を sin/cos、経過日数・通算月を線形で
    >>> out = add_time_features(
    ...     train, 'TransactionDT',
    ...     cycles={'hour': 86400, 'week': 86400 * 7},
    ...     linear={'day': 86400, 'month': 86400 * 30},
    ...     origin=train['TransactionDT'].min(),
    ... )
    Created: hour_sin, hour_cos, week_sin, week_cos, day, month
    >>> # 月内の位置(月初/月末の周期)が欲しい場合は month を cycles に
    >>> out = add_time_features(
    ...     train, 'TransactionDT', cycles={'month': 86400 * 30},
    ... )
    Created: month_sin, month_cos
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in df.")
    if len(df) == 0:
        raise ValueError("df is empty.")
    if not cycles and not linear:
        raise ValueError("Specify at least one of `cycles` or `linear`.")

    for name, period in (cycles or {}).items():
        if period <= 0:
            raise ValueError(f"cycles['{name}'] period must be > 0, got {period}.")
    for name, unit in (linear or {}).items():
        if unit <= 0:
            raise ValueError(f"linear['{name}'] unit must be > 0, got {unit}.")

    df_res = df.copy()
    values = df_res[col]

    # 欠損チェック(警告のみ・処理続行・NaN は透過)
    n_missing = int(values.isna().sum())
    if n_missing > 0:
        ratio = n_missing / len(df_res) * 100
        warnings.warn(
            f"'{col}' has {n_missing} missing values ({ratio:.2f}%); "
            f"output rows for these will be NaN.",
            stacklevel=2,
        )

    created = []

    # cycles: 周期で巻き取って sin/cos
    for name, period in (cycles or {}).items():
        phase = (values % period) / period  # [0, 1)
        angle = 2 * np.pi * phase
        df_res[f'{name}_sin'] = np.sin(angle)
        df_res[f'{name}_cos'] = np.cos(angle)
        created.extend([f'{name}_sin', f'{name}_cos'])

    # linear: origin からの経過量
    for name, unit in (linear or {}).items():
        df_res[name] = (values - origin) / unit
        created.append(name)

    if verbose:
        print(f"Created: {', '.join(created)}")

    return df_res