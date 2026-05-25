"""make_composite_features のテスト."""
import numpy as np
import pandas as pd
import pytest
from ml_utils.feature_engineering.composite import make_composite_features


# ===== 基本動作テスト =====
def test_returns_dataframe():
    """戻り値は DataFrame."""
    df = pd.DataFrame({'A': [1.0, 2.0], 'B': [3.0, 4.0]})
    result = make_composite_features(df, ['A', 'B'])
    assert isinstance(result, pd.DataFrame)


def test_composite_columns_added():
    """MEAN / STD / GEOM_MEAN 列が追加される."""
    df = pd.DataFrame({'A': [1.0, 2.0], 'B': [3.0, 4.0]})
    result = make_composite_features(df, ['A', 'B'], prefix='X')
    assert 'X_MEAN' in result.columns
    assert 'X_STD' in result.columns
    assert 'X_GEOM_MEAN' in result.columns


def test_original_df_not_modified():
    """入力 DataFrame は変更されない(copy 保証)."""
    df = pd.DataFrame({'A': [1.0, 2.0], 'B': [3.0, 4.0]})
    cols_before = df.columns.tolist()
    make_composite_features(df, ['A', 'B'])
    assert df.columns.tolist() == cols_before


# ===== 動作仕様テスト =====
def test_mean_value_correct():
    """算術平均が正しく計算される."""
    df = pd.DataFrame({'A': [2.0, 4.0], 'B': [4.0, 8.0]})
    result = make_composite_features(df, ['A', 'B'], prefix='X')
    # 行0: (2+4)/2 = 3.0, 行1: (4+8)/2 = 6.0
    assert np.isclose(result['X_MEAN'].iloc[0], 3.0)
    assert np.isclose(result['X_MEAN'].iloc[1], 6.0)


def test_geom_mean_value_correct():
    """幾何平均が正しく計算される."""
    df = pd.DataFrame({'A': [4.0], 'B': [9.0]})
    result = make_composite_features(df, ['A', 'B'], prefix='X')
    # sqrt(4 * 9) = sqrt(36) = 6.0
    assert np.isclose(result['X_GEOM_MEAN'].iloc[0], 6.0)


def test_weighted_only_when_weights_given():
    """weights 無しで WEIGHTED 列なし、有りで生成される."""
    df = pd.DataFrame({'A': [1.0, 2.0], 'B': [3.0, 4.0]})
    # weights 無し
    result_no = make_composite_features(df, ['A', 'B'], prefix='X')
    assert 'X_WEIGHTED' not in result_no.columns
    # weights 有り
    result_yes = make_composite_features(df, ['A', 'B'], prefix='X', weights=[0.5, 0.5])
    assert 'X_WEIGHTED' in result_yes.columns


def test_error_on_missing_column():
    """指定列が存在しない場合、KeyError を投げる."""
    df = pd.DataFrame({'A': [1.0, 2.0]})
    with pytest.raises(KeyError):
        make_composite_features(df, ['A', 'NOT_EXIST'])


def test_error_on_weights_length_mismatch():
    """weights の長さが cols と不一致の場合、ValueError を投げる."""
    df = pd.DataFrame({'A': [1.0, 2.0], 'B': [3.0, 4.0]})
    with pytest.raises(ValueError):
        make_composite_features(df, ['A', 'B'], weights=[0.5])  # cols は 2 個


# ===== 本命テスト(NaN/0 の扱いと警告) =====
def test_warns_on_nan():
    """NaN を含む場合、警告を出し、該当行の合成列は NaN になる."""
    df = pd.DataFrame({'A': [1.0, np.nan], 'B': [3.0, 4.0]})
    with pytest.warns(UserWarning):
        result = make_composite_features(df, ['A', 'B'], prefix='X')
    # NaN を含む行1 は MEAN/STD/GEOM すべて NaN
    assert np.isnan(result['X_MEAN'].iloc[1])
    assert np.isnan(result['X_GEOM_MEAN'].iloc[1])


def test_warns_on_zero():
    """0 を含む場合、警告を出し、該当行の幾何平均は 0 になる."""
    df = pd.DataFrame({'A': [2.0, 0.0], 'B': [4.0, 5.0]})
    with pytest.warns(UserWarning):
        result = make_composite_features(df, ['A', 'B'], prefix='X')
    # 0 を含む行1 の幾何平均は 0
    assert np.isclose(result['X_GEOM_MEAN'].iloc[1], 0.0)
    # ただし MEAN は普通に計算される (0+5)/2 = 2.5
    assert np.isclose(result['X_MEAN'].iloc[1], 2.5)
