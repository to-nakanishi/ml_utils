"""target_encode_oof のテスト."""
import numpy as np
import pandas as pd
import pytest
from ml_utils.feature_engineering.target_encoding import (
    target_encode_oof,
    diagnose_target_encoding,
)


# ===== 基本動作テスト =====
def test_returns_tuple_of_dataframes():
    """戻り値は (train, test) の DataFrame タプル."""
    train = pd.DataFrame({'CAT': ['A', 'B'] * 50, 'TARGET': [1, 0] * 50})
    test = pd.DataFrame({'CAT': ['A', 'B']})
    result = target_encode_oof(train, test, 'CAT', verbose=False)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], pd.DataFrame)
    assert isinstance(result[1], pd.DataFrame)


def test_new_column_added():
    """エンコード列 `{col}_TARGET_RATE` が train/test 両方に追加される."""
    train = pd.DataFrame({'CAT': ['A', 'B'] * 50, 'TARGET': [1, 0] * 50})
    test = pd.DataFrame({'CAT': ['A', 'B']})
    train_enc, test_enc = target_encode_oof(train, test, 'CAT', verbose=False)
    assert 'CAT_TARGET_RATE' in train_enc.columns
    assert 'CAT_TARGET_RATE' in test_enc.columns


def test_original_dataframe_not_modified():
    """入力 DataFrame は変更されない(copy 保証)."""
    train = pd.DataFrame({'CAT': ['A', 'B'] * 50, 'TARGET': [1, 0] * 50})
    test = pd.DataFrame({'CAT': ['A', 'B']})
    train_cols_before = train.columns.tolist()
    test_cols_before = test.columns.tolist()
    target_encode_oof(train, test, 'CAT', verbose=False)
    assert train.columns.tolist() == train_cols_before
    assert test.columns.tolist() == test_cols_before


def test_encoded_values_are_finite():
    """エンコード値に NaN や inf が含まれない."""
    train = pd.DataFrame({'CAT': ['A', 'B', 'C'] * 50, 'TARGET': [1, 0, 1] * 50})
    test = pd.DataFrame({'CAT': ['A', 'B', 'C']})
    train_enc, test_enc = target_encode_oof(train, test, 'CAT', verbose=False)
    assert np.isfinite(train_enc['CAT_TARGET_RATE']).all()
    assert np.isfinite(test_enc['CAT_TARGET_RATE']).all()


# ===== 動作仕様テスト =====
def test_unknown_category_in_test():
    """test に train にない未知カテゴリがある場合、global_mean で埋められる."""
    train = pd.DataFrame({'CAT': ['A', 'B'] * 50, 'TARGET': [1, 0] * 50})
    test = pd.DataFrame({'CAT': ['A', 'B', 'UNKNOWN']})
    _, test_enc = target_encode_oof(train, test, 'CAT', verbose=False)
    global_mean = train['TARGET'].mean()
    unknown_value = test_enc.loc[test['CAT'] == 'UNKNOWN', 'CAT_TARGET_RATE'].iloc[0]
    assert np.isclose(unknown_value, global_mean)


def test_custom_cv_object():
    """scikit-learn 互換の CV オブジェクト(KFold 等)を渡しても動く."""
    from sklearn.model_selection import KFold
    train = pd.DataFrame({'CAT': ['A', 'B'] * 50, 'TARGET': [1, 0] * 50})
    test = pd.DataFrame({'CAT': ['A', 'B']})
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    train_enc, test_enc = target_encode_oof(train, test, 'CAT', cv=cv, verbose=False)
    assert 'CAT_TARGET_RATE' in train_enc.columns
    assert np.isfinite(train_enc['CAT_TARGET_RATE']).all()


def test_verbose_false_silent(capsys):
    """verbose=False の場合、標準出力に何も出さない."""
    train = pd.DataFrame({'CAT': ['A', 'B'] * 50, 'TARGET': [1, 0] * 50})
    test = pd.DataFrame({'CAT': ['A', 'B']})
    target_encode_oof(train, test, 'CAT', verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ''


def test_error_on_missing_column():
    """指定列が train/test に存在しない場合、KeyError を投げる."""
    train = pd.DataFrame({'CAT': ['A', 'B'] * 50, 'TARGET': [1, 0] * 50})
    test = pd.DataFrame({'CAT': ['A', 'B']})
    with pytest.raises(KeyError):
        target_encode_oof(train, test, 'NOT_EXIST', verbose=False)


# ===== テスト(OOF/smoothing 効果検証) =====
def test_oof_prevents_leakage():
    """OOF により、エンコード値はカテゴリ別 target 率と完全一致しない(リーク防止が機能)."""
    # カテゴリ A は target が全て 1、B は全て 0 という極端なデータ
    train = pd.DataFrame({
        'CAT': ['A'] * 50 + ['B'] * 50,
        'TARGET': [1] * 50 + [0] * 50,
    })
    test = pd.DataFrame({'CAT': ['A', 'B']})
    train_enc, _ = target_encode_oof(train, test, 'CAT', verbose=False)
    # リークしていれば A のエンコード値は 1.0、B は 0.0 で完全一致するはず
    # OOF が機能していれば smoothing と fold 分割の影響でズレる
    a_values = train_enc.loc[train['CAT'] == 'A', 'CAT_TARGET_RATE'].values
    b_values = train_enc.loc[train['CAT'] == 'B', 'CAT_TARGET_RATE'].values
    # A の値が 1.0 から離れていることを確認(リークしていれば 1.0 ピッタリになる)
    assert not np.allclose(a_values, 1.0)
    # B の値が 0.0 から離れていることを確認
    assert not np.allclose(b_values, 0.0)


def test_smoothing_pulls_toward_global_mean():
    """smoothing が大きいほど、エンコード値が global_mean に近づく."""
    train = pd.DataFrame({
        'CAT': ['A'] * 50 + ['B'] * 50,
        'TARGET': [1] * 50 + [0] * 50,
    })
    test = pd.DataFrame({'CAT': ['A', 'B']})
    global_mean = train['TARGET'].mean()  # 0.5
    # smoothing 小: カテゴリ平均に近い値
    _, test_enc_small = target_encode_oof(train, test, 'CAT', smoothing=0.1, verbose=False)
    # smoothing 大: global_mean に近い値
    _, test_enc_large = target_encode_oof(train, test, 'CAT', smoothing=1000.0, verbose=False)
    # global_mean からの距離を比較
    dist_small = np.abs(test_enc_small['CAT_TARGET_RATE'].values - global_mean).mean()
    dist_large = np.abs(test_enc_large['CAT_TARGET_RATE'].values - global_mean).mean()
    assert dist_large < dist_small


# ===== 診断関数テスト =====
def test_diagnose_basic_output():
    """diagnose_target_encoding は所定のキーを持つ dict を返す."""
    train = pd.DataFrame({'CAT': ['A', 'B'] * 50, 'TARGET': [1, 0] * 50})
    test = pd.DataFrame({'CAT': ['A', 'B']})
    train_enc, test_enc = target_encode_oof(train, test, 'CAT', verbose=False)
    report = diagnose_target_encoding(train_enc, test_enc, 'CAT', 'CAT_TARGET_RATE')
    expected_keys = {
        'encoded_col', 'groups', 'small_groups', 'small_group_threshold',
        'oof_mean', 'global_mean', 'test_unknown_count', 'test_total', 'test_coverage',
    }
    assert isinstance(report, dict)
    assert set(report.keys()) == expected_keys


def test_diagnose_small_groups_detected():
    """サンプル数が閾値未満のグループ数を正しく検出する."""
    # A: 50件、B: 50件、C: 3件、D: 2件 (C, D が small group)
    train = pd.DataFrame({
        'CAT': ['A'] * 50 + ['B'] * 50 + ['C'] * 3 + ['D'] * 2,
        'TARGET': [1, 0] * 50 + [1] * 3 + [0] * 2,
    })
    test = pd.DataFrame({'CAT': ['A', 'B']})
    train_enc, test_enc = target_encode_oof(train, test, 'CAT', verbose=False)
    report = diagnose_target_encoding(
        train_enc, test_enc, 'CAT', 'CAT_TARGET_RATE', small_group_threshold=10
    )
    assert report['groups'] == 4
    assert report['small_groups'] == 2  # C と D


def test_diagnose_oof_vs_global_close():
    """リーク防止が機能していれば OOF mean ≈ global mean となる."""
    train = pd.DataFrame({'CAT': ['A', 'B', 'C'] * 50, 'TARGET': [1, 0, 1] * 50})
    test = pd.DataFrame({'CAT': ['A', 'B', 'C']})
    train_enc, test_enc = target_encode_oof(train, test, 'CAT', verbose=False)
    report = diagnose_target_encoding(train_enc, test_enc, 'CAT', 'CAT_TARGET_RATE')
    # OOF + smoothing が機能していれば、両者の差は十分小さい
    assert abs(report['oof_mean'] - report['global_mean']) < 0.01


def test_diagnose_test_coverage():
    """test の未知カテゴリを正しくカウントし、coverage を算出する."""
    train = pd.DataFrame({'CAT': ['A', 'B'] * 50, 'TARGET': [1, 0] * 50})
    # test 10件中、UNKNOWN が 2件 → coverage = 0.8
    test = pd.DataFrame({'CAT': ['A'] * 4 + ['B'] * 4 + ['UNKNOWN'] * 2})
    train_enc, test_enc = target_encode_oof(train, test, 'CAT', verbose=False)
    report = diagnose_target_encoding(train_enc, test_enc, 'CAT', 'CAT_TARGET_RATE')
    assert report['test_unknown_count'] == 2
    assert report['test_total'] == 10
    assert np.isclose(report['test_coverage'], 0.8)
