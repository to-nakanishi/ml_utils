"""Tests for add_time_features."""

import numpy as np
import pandas as pd
import pytest

from ml_utils.feature_engineering.temporal import add_time_features

DAY = 86400


def _df(seconds):
    return pd.DataFrame({'TransactionDT': seconds})


def test_creates_expected_columns():
    df = _df([DAY, DAY * 2, DAY * 3])
    out = add_time_features(
        df, 'TransactionDT',
        cycles={'hour': DAY, 'week': DAY * 7},
        linear={'day': DAY, 'month': DAY * 30},
        verbose=False,
    )
    for c in ['hour_sin', 'hour_cos', 'week_sin', 'week_cos', 'day', 'month']:
        assert c in out.columns


def test_cycle_wraps_around():
    """同一周期内の位置が同じなら、別の通算時刻でも sin/cos は一致する。"""
    # 0:00 ちょうど(1日後)と 0:00 ちょうど(2日後)は時刻として同じ位相
    df = _df([DAY, DAY * 2])
    out = add_time_features(df, 'TransactionDT', cycles={'hour': DAY}, verbose=False)
    assert np.isclose(out['hour_sin'].iloc[0], out['hour_sin'].iloc[1])
    assert np.isclose(out['hour_cos'].iloc[0], out['hour_cos'].iloc[1])


def test_cycle_boundary_is_continuous():
    """23時付近と0時付近が円周上で隣接していること(sin/cosが近い)。"""
    almost_midnight = DAY + (23 * 3600 + 3599)  # 23:59:59
    just_after = DAY + DAY                       # 翌0:00:00
    df = _df([almost_midnight, just_after])
    out = add_time_features(df, 'TransactionDT', cycles={'hour': DAY}, verbose=False)
    # 円周上の距離が小さい = sin,cos の差が小さい
    dist = np.hypot(
        out['hour_sin'].iloc[0] - out['hour_sin'].iloc[1],
        out['hour_cos'].iloc[0] - out['hour_cos'].iloc[1],
    )
    assert dist < 0.01


def test_cycle_sin_cos_on_unit_circle():
    """sin^2 + cos^2 = 1 を満たす(正しい円周上の点)。"""
    df = _df([0, 12345, DAY * 5 + 777, DAY * 100])
    out = add_time_features(df, 'TransactionDT', cycles={'hour': DAY}, verbose=False)
    r2 = out['hour_sin'] ** 2 + out['hour_cos'] ** 2
    assert np.allclose(r2, 1.0)


def test_linear_origin_default_zero():
    df = _df([0, DAY, DAY * 3])
    out = add_time_features(df, 'TransactionDT', linear={'day': DAY}, verbose=False)
    assert list(out['day']) == [0.0, 1.0, 3.0]


def test_linear_origin_applied():
    """origin を引いた経過量になること(train基準を test に渡す想定)。"""
    df = _df([DAY * 5, DAY * 6])
    out = add_time_features(
        df, 'TransactionDT', linear={'day': DAY}, origin=DAY * 5, verbose=False
    )
    assert list(out['day']) == [0.0, 1.0]


def test_input_not_mutated():
    df = _df([DAY, DAY * 2])
    before = df.copy()
    add_time_features(df, 'TransactionDT', cycles={'hour': DAY}, verbose=False)
    pd.testing.assert_frame_equal(df, before)


def test_missing_warns_and_passes_through_nan():
    df = _df([DAY, np.nan, DAY * 2])
    with pytest.warns(UserWarning):
        out = add_time_features(
            df, 'TransactionDT',
            cycles={'hour': DAY}, linear={'day': DAY}, verbose=False,
        )
    # 欠損行の出力は NaN、非欠損行は有効値
    assert out['hour_sin'].isna().tolist() == [False, True, False]
    assert out['day'].isna().tolist() == [False, True, False]


def test_guard_missing_column():
    df = _df([DAY])
    with pytest.raises(KeyError):
        add_time_features(df, 'NOPE', cycles={'hour': DAY}, verbose=False)


def test_guard_empty_df():
    df = pd.DataFrame({'TransactionDT': []})
    with pytest.raises(ValueError):
        add_time_features(df, 'TransactionDT', cycles={'hour': DAY}, verbose=False)


def test_guard_no_axes_specified():
    df = _df([DAY])
    with pytest.raises(ValueError):
        add_time_features(df, 'TransactionDT', verbose=False)


def test_guard_nonpositive_period():
    df = _df([DAY])
    with pytest.raises(ValueError):
        add_time_features(df, 'TransactionDT', cycles={'hour': 0}, verbose=False)


def test_guard_nonpositive_unit():
    df = _df([DAY])
    with pytest.raises(ValueError):
        add_time_features(df, 'TransactionDT', linear={'day': -1}, verbose=False)