"""downcast_numeric のテスト."""
import numpy as np
import pandas as pd

from ml_utils.feature_engineering.memory import downcast_numeric


def test_downcast_int_to_int8():
    """小さい整数値が int8 にダウンキャストされる."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    result = downcast_numeric(df, verbose=False)
    assert result['a'].dtype == np.int8


def test_downcast_int_to_int16():
    """int8 範囲を超える整数値が int16 にダウンキャストされる."""
    df = pd.DataFrame({'a': [1, 1000, 30000]})
    result = downcast_numeric(df, verbose=False)
    assert result['a'].dtype == np.int16


def test_downcast_float_to_float32():
    """float32 範囲に収まる浮動小数が float32 にダウンキャストされる."""
    df = pd.DataFrame({'a': [1.5, 2.5, 3.5]})
    result = downcast_numeric(df, verbose=False)
    assert result['a'].dtype == np.float32


def test_object_columns_untouched():
    """文字列列はダウンキャストの対象外."""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    result = downcast_numeric(df, verbose=False)
    assert result['b'].dtype == object


def test_returns_same_object():
    """返り値は入力DataFrameと同一オブジェクト(in-place)."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    result = downcast_numeric(df, verbose=False)
    assert result is df


def test_verbose_false_silent(capsys):
    """verbose=False の場合、標準出力に何も出さない."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    downcast_numeric(df, verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ''
