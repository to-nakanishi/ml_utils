"""target_encode_oof のテスト."""
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
