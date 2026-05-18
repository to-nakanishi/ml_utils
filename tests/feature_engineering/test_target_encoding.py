"""target_encode_oof のテスト."""
import pytest
import numpy as np
import pandas as pd
from ml_utils.feature_engineering.target_encoding import target_encode_oof


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
