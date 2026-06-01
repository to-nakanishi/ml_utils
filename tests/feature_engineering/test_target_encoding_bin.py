"""Tests for bin_target_encode_oof."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from ml_utils.feature_engineering.target_encoding import bin_target_encode_oof


def _make_data(n=600, seed=0):
    """数値列 x のビンと target が単調に対応する合成データ。"""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, size=n)
    # x が大きいほどデフォルト率が高い
    prob = x / 100.0
    y = (rng.uniform(0, 1, size=n) < prob).astype(int)
    train = pd.DataFrame({'x': x, 'TARGET': y})
    test = pd.DataFrame({'x': rng.uniform(0, 100, size=n // 3)})
    return train, test


def test_basic_output_shape_and_dtype():
    train, test = _make_data()
    tr, te = bin_target_encode_oof(train, test, 'x', bins=10, verbose=False)

    assert 'x_BIN_TARGET_RATE' in tr.columns
    assert 'x_BIN_TARGET_RATE' in te.columns
    assert tr['x_BIN_TARGET_RATE'].dtype == np.float32
    assert te['x_BIN_TARGET_RATE'].dtype == np.float32
    # train・test とも欠損なし、率なので [0, 1] に収まる
    assert tr['x_BIN_TARGET_RATE'].notna().all()
    assert te['x_BIN_TARGET_RATE'].notna().all()
    assert tr['x_BIN_TARGET_RATE'].between(0, 1).all()


def test_encoding_is_monotonic_with_target():
    """x が大きいビンほどエンコード率が高い、を大まかに確認。"""
    train, test = _make_data(n=2000)
    tr, _ = bin_target_encode_oof(train, test, 'x', bins=10, verbose=False)
    # x の下位25%と上位25%でエンコード値の平均を比較
    low = tr.loc[tr['x'] <= tr['x'].quantile(0.25), 'x_BIN_TARGET_RATE'].mean()
    high = tr.loc[tr['x'] >= tr['x'].quantile(0.75), 'x_BIN_TARGET_RATE'].mean()
    assert high > low


def test_int_and_list_bins_both_work():
    train, test = _make_data()
    tr_i, te_i = bin_target_encode_oof(train, test, 'x', bins=5, verbose=False)
    tr_l, te_l = bin_target_encode_oof(
        train, test, 'x', bins=[-np.inf, 25, 50, 75, np.inf], verbose=False
    )
    for df in (tr_i, te_i, tr_l, te_l):
        assert 'x_BIN_TARGET_RATE' in df.columns
        assert df['x_BIN_TARGET_RATE'].notna().all()


def test_missing_gets_its_own_rate_not_global_mean():
    """
    最重要: 欠損行は独立ビンとして、欠損行自体のデフォルト率でエンコードされ、
    global mean には潰れないこと。
    欠損行のデフォルト率を全体平均から大きく離して構成し、検出する。
    """
    rng = np.random.default_rng(1)
    n_obs = 500
    n_miss = 300

    # 観測あり: デフォルト率 ~0.1
    x_obs = rng.uniform(0, 100, size=n_obs)
    y_obs = (rng.uniform(0, 1, size=n_obs) < 0.1).astype(int)

    # 欠損: デフォルト率 ~0.8 (全体平均から大きく乖離)
    x_miss = np.full(n_miss, np.nan)
    y_miss = (rng.uniform(0, 1, size=n_miss) < 0.8).astype(int)

    train = pd.DataFrame({
        'x': np.concatenate([x_obs, x_miss]),
        'TARGET': np.concatenate([y_obs, y_miss]),
    })
    test = pd.DataFrame({'x': [np.nan, 50.0, np.nan]})

    global_mean = train['TARGET'].mean()  # ~0.36 付近

    tr, te = bin_target_encode_oof(train, test, 'x', bins=10, verbose=False)

    # train の欠損行のエンコード値: 欠損グループの実デフォルト率(~0.8)に近い
    miss_enc = tr.loc[train['x'].isna(), 'x_BIN_TARGET_RATE'].mean()
    assert miss_enc > 0.6, f"欠損ビンが潰れている可能性: {miss_enc:.3f}"
    # global mean に潰れていない
    assert abs(miss_enc - global_mean) > 0.2

    # test の欠損行も欠損グループ率が乗る (50.0 の行とは別の値)
    te_miss = te.loc[te['x'].isna(), 'x_BIN_TARGET_RATE'].values
    te_obs = te.loc[te['x'].notna(), 'x_BIN_TARGET_RATE'].values
    assert (te_miss > 0.6).all()
    assert te_obs[0] < te_miss[0]


def test_train_edges_applied_to_test_outlier():
    """
    int 等幅のとき、test に train 範囲外の外れ値があっても例外なく処理され、
    範囲外は欠損ビン扱い(NaN にならない)になること。
    """
    train, _ = _make_data()
    test = pd.DataFrame({'x': [-999.0, 50.0, 9999.0]})  # 範囲外を含む
    _, te = bin_target_encode_oof(train, test, 'x', bins=10, verbose=False)
    assert te['x_BIN_TARGET_RATE'].notna().all()


def test_input_not_mutated_and_no_temp_column():
    train, test = _make_data()
    train_before = train.copy()
    test_before = test.copy()

    tr, te = bin_target_encode_oof(train, test, 'x', bins=10, verbose=False)

    # 入力は不変
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(test, test_before)
    # 元には新列なし
    assert 'x_BIN_TARGET_RATE' not in train.columns
    # 作業用ビン列が残っていない
    assert not any(c.startswith('_x_bin') for c in tr.columns)
    assert not any(c.startswith('_x_bin') for c in te.columns)


def test_custom_cv_is_respected():
    train, test = _make_data()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    tr, _ = bin_target_encode_oof(train, test, 'x', bins=10, cv=cv, verbose=False)
    assert tr['x_BIN_TARGET_RATE'].notna().all()


def test_guard_missing_column_raises():
    train, test = _make_data()
    with pytest.raises(KeyError):
        bin_target_encode_oof(train, test, 'not_a_col', bins=10, verbose=False)


def test_guard_missing_target_raises():
    train, test = _make_data()
    with pytest.raises(KeyError):
        bin_target_encode_oof(train, test, 'x', bins=10, target='NOPE', verbose=False)


def test_guard_empty_train_raises():
    _, test = _make_data()
    empty = pd.DataFrame({'x': [], 'TARGET': []})
    with pytest.raises(ValueError):
        bin_target_encode_oof(empty, test, 'x', bins=10, verbose=False)


def test_guard_bad_bins_type_raises():
    train, test = _make_data()
    with pytest.raises(TypeError):
        bin_target_encode_oof(train, test, 'x', bins='ten', verbose=False)


def test_guard_too_few_edges_raises():
    train, test = _make_data()
    with pytest.raises(ValueError):
        bin_target_encode_oof(train, test, 'x', bins=[0.0], verbose=False)
