"""downcast_numeric のテスト."""

import numpy as np
import pandas as pd

from ml_utils.feature_engineering.memory import downcast_numeric


# ===== 基本動作テスト =====

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


# ===== 境界値テスト(int) =====

def test_downcast_int_at_int8_inner_boundary():
    """int8範囲ぎりぎり内側(-127, 126)は int8 にダウンキャストされる."""
    df = pd.DataFrame({'a': [-127, 126]})
    result = downcast_numeric(df, verbose=False)
    assert result['a'].dtype == np.int8


def test_downcast_int_at_int8_exact_boundary():
    """int8範囲ぴったりの値(-128, 127)は int16 になる(現仕様: 厳密不等号のため)."""
    df = pd.DataFrame({'a': [-128, 127]})
    result = downcast_numeric(df, verbose=False)
    assert result['a'].dtype == np.int16


def test_downcast_int_overflow_to_int16():
    """int8範囲を1だけ超えた値(-129, 128)は int16 にダウンキャストされる."""
    df = pd.DataFrame({'a': [-129, 128]})
    result = downcast_numeric(df, verbose=False)
    assert result['a'].dtype == np.int16


# ===== 境界値テスト(float) =====

def test_downcast_float_at_float32_inner_boundary():
    """float32範囲内の値はfloat32にダウンキャストされる."""
    df = pd.DataFrame({'a': [-1e38, 1e38]})  # float32範囲内(約3.4e38)
    result = downcast_numeric(df, verbose=False)
    assert result['a'].dtype == np.float32


def test_downcast_float_overflow_to_float64():
    """float32範囲を超える値はfloat64のままになる."""
    df = pd.DataFrame({'a': [-1e39, 1e39]})  # float32範囲外
    result = downcast_numeric(df, verbose=False)
    assert result['a'].dtype == np.float64


# ===== 動作仕様テスト =====

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
