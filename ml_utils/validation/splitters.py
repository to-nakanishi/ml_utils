"""Time-series cross-validation splitters."""
import numpy as np


class SlidingWindowSplit:
    """
    ----------
    summary
    ----------
    時間軸の期間ラベルに基づく時系列クロスバリデーション。
    「直近 train_window 期間で学習 → 次の1期間を検証」を時間順にスライドさせる。
    sklearn 互換の split() を持ち、run_baseline などの cv 引数にそのまま渡せる。

    SKF と異なり時間順を保つため「未来で過去を予測する」リークが起きない。
    IEEE-CIS Fraud のような train/test が時系列で分かれるデータを想定。

    expanding=False(既定)はスライド窓(直近 train_window 期間のみ学習に使用)、
    expanding=True は拡大窓(最初の期間から検証直前まで全て学習に使用)。

    学習窓が train_window 期間分そろわない初期の期間は検証対象から除外する。

    ----------
    Parameters
    ----------
    time_col : str
        時間軸の期間ラベル列名(例: 'time_month')。整数の期間ラベルを想定。
        ラベルの作成(30日割り等)は呼び出し側の責務。分割の基準に使う列であり、
        学習に使う場合は X に残し、使わない場合は事前に除外しておくこと。
    train_window : int, default=2
        スライド窓のとき、学習に使う直近の期間数。
        expanding=True のときは無視される(常に最初から使うため)。
    expanding : bool, default=False
        False ならスライド窓、True なら拡大窓。

    ----------
    Examples
    ----------
    >>> tscv = SlidingWindowSplit(time_col='time_month', train_window=2)
    >>> for tr_idx, va_idx in tscv.split(X, y):
    ...     ...
    >>> run_baseline(X, y, cv=tscv)   # baseline にそのまま渡せる
    """

    def __init__(self, time_col: str, train_window: int = 2, expanding: bool = False):
        self.time_col = time_col
        self.train_window = train_window
        self.expanding = expanding

    def split(self, X, y=None, groups=None):
        if self.time_col not in X.columns:
            raise KeyError(f"time_col '{self.time_col}' not found in X.")

        periods = np.sort(X[self.time_col].unique())
        time_values = X[self.time_col].values
        positions = np.arange(len(X))

        for i, val_period in enumerate(periods):
            # 学習窓が train_window 分そろわない初期の期間はスキップ
            if i < self.train_window:
                continue

            if self.expanding:
                train_periods = periods[:i]
            else:
                train_periods = periods[i - self.train_window:i]

            train_mask = np.isin(time_values, train_periods)
            val_mask = time_values == val_period

            train_idx = positions[train_mask]
            val_idx = positions[val_mask]

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        if X is None:
            raise ValueError("X is required to compute n_splits.")
        n_periods = X[self.time_col].nunique()
        return max(0, n_periods - self.train_window)
