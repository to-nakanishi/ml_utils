"""SlidingWindowSplit のテスト."""
import numpy as np
import pandas as pd
import pytest

from ml_utils.validation.splitters import SlidingWindowSplit


def _make_X(n_periods=6, per=100):
    """time_month 0..n_periods-1 を各 per 行持つデータ."""
    rows = []
    for m in range(n_periods):
        for _ in range(per):
            rows.append({'time_month': m, 'feat': np.random.randn()})
    return pd.DataFrame(rows)


def _periods_of(X, idx, col='time_month'):
    return sorted(X.iloc[idx][col].unique().tolist())


# ===== スライド窓 =====
def test_sliding_window_splits():
    """直近 train_window 期間で学習 → 次の1期間を検証(初期はスキップ)."""
    X = _make_X(6)
    tscv = SlidingWindowSplit('time_month', train_window=2)
    splits = list(tscv.split(X))
    assert len(splits) == 4
    tr, va = splits[0]
    assert _periods_of(X, tr) == [0, 1]
    assert _periods_of(X, va) == [2]
    tr, va = splits[-1]
    assert _periods_of(X, tr) == [3, 4]
    assert _periods_of(X, va) == [5]


def test_sliding_window_size_constant():
    """スライド窓では学習窓のサイズが一定(直近 train_window 期間)."""
    X = _make_X(6)
    tscv = SlidingWindowSplit('time_month', train_window=2)
    for tr, va in tscv.split(X):
        assert len(_periods_of(X, tr)) == 2


# ===== 拡大窓 =====
def test_expanding_window():
    """expanding=True で学習窓が最初から拡大する."""
    X = _make_X(6)
    tscv = SlidingWindowSplit('time_month', train_window=2, expanding=True)
    splits = list(tscv.split(X))
    assert _periods_of(X, splits[0][0]) == [0, 1]
    assert _periods_of(X, splits[1][0]) == [0, 1, 2]
    assert _periods_of(X, splits[-1][0]) == [0, 1, 2, 3, 4]


# ===== エッジ =====
def test_skips_early_periods():
    """学習窓が train_window 分そろわない初期期間は検証対象外."""
    X = _make_X(6)
    tscv = SlidingWindowSplit('time_month', train_window=3)
    splits = list(tscv.split(X))
    assert len(splits) == 3
    assert _periods_of(X, splits[0][1]) == [3]


def test_too_few_periods_yields_nothing():
    """期間が train_window 以下なら有効な分割は0個."""
    X = _make_X(2)
    tscv = SlidingWindowSplit('time_month', train_window=2)
    assert len(list(tscv.split(X))) == 0


def test_error_on_missing_time_col():
    """time_col が X に無い場合 KeyError."""
    X = _make_X(6)
    tscv = SlidingWindowSplit('NOT_EXIST', train_window=2)
    with pytest.raises(KeyError):
        list(tscv.split(X))


def test_get_n_splits():
    """get_n_splits が 期間数 - train_window を返す."""
    X = _make_X(6)
    tscv = SlidingWindowSplit('time_month', train_window=2)
    assert tscv.get_n_splits(X) == 4
