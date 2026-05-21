"""find_density_crossover のテスト."""
import pytest
import numpy as np
import pandas as pd
from ml_utils.feature_engineering.density import find_density_crossover


def _make_separated(n=3000, seed=42):
    """きれいに分離した1列を生成(target=0 高スコア寄り / target=1 低スコア寄り).

    EXT_SOURCE と同じ向き。交点は2クラスの中心(0.62, 0.32)の中間に1個出る想定。
    """
    rng = np.random.RandomState(seed)
    target = rng.randint(0, 2, size=n)
    score = np.where(
        target == 0,
        rng.normal(0.62, 0.16, size=n),
        rng.normal(0.32, 0.18, size=n),
    )
    score = np.clip(score, 0.0, 1.0)
    return pd.DataFrame({'EXT': score, 'TARGET': target})


def _make_overlap(n=2000, seed=0):
    """2クラスが同一分布の列を生成(分離しない → 複数交点が出る想定)."""
    rng = np.random.RandomState(seed)
    target = rng.randint(0, 2, size=n)
    score = np.clip(rng.normal(0.5, 0.15, size=n), 0.0, 1.0)
    return pd.DataFrame({'OVERLAP': score, 'TARGET': target})


# ===== 基本動作テスト =====
def test_returns_dict():
    """戻り値は dict."""
    df = _make_separated()
    result = find_density_crossover(df, ['EXT'])
    assert isinstance(result, dict)


def test_single_crossover_in_expected_range():
    """きれいに分離した列は、交点が2クラスの中間に1個返る."""
    df = _make_separated()
    result = find_density_crossover(df, ['EXT'])
    assert 'EXT' in result
    assert 0.3 < result['EXT'] < 0.6


def test_dict_keys_match_input_columns():
    """交点が出た列だけが dict のキーになる."""
    df = _make_separated()
    result = find_density_crossover(df, ['EXT'])
    assert set(result.keys()) == {'EXT'}


def test_original_df_not_modified():
    """入力 DataFrame は変更されない(読み取り専用)."""
    df = _make_separated()
    cols_before = df.columns.tolist()
    df_copy = df.copy()
    find_density_crossover(df, ['EXT'])
    assert df.columns.tolist() == cols_before
    assert df.equals(df_copy)


# ===== 欠損・除外の挙動テスト =====
def test_nan_excluded_from_estimation():
    """NaN は除外して算出され、欠損なし版とほぼ一致する."""
    df_full = _make_separated()
    res_full = find_density_crossover(df_full, ['EXT'])

    df_nan = df_full.copy()
    df_nan.loc[:599, 'EXT'] = np.nan  # 2割を欠損化
    res_nan = find_density_crossover(df_nan, ['EXT'])

    assert 'EXT' in res_nan
    assert abs(res_nan['EXT'] - res_full['EXT']) < 0.05


# ===== 異常系・警告テスト =====
def test_multiple_crossovers_warns_and_returns_rightmost():
    """複数交点が出ると警告し、暫定的に右端を返して dict に含める."""
    df = _make_overlap()
    with pytest.warns(UserWarning, match="crossovers found"):
        result = find_density_crossover(df, ['OVERLAP'])
    assert 'OVERLAP' in result


def test_single_class_skipped_with_warning():
    """片方のクラスにサンプルが無い列は警告して skip(dict に含まれない)."""
    n = 2000
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        'COL': np.clip(rng.normal(0.5, 0.15, n), 0, 1),
        'TARGET': np.zeros(n, dtype=int),
    })
    with pytest.warns(UserWarning, match="no non-NaN samples"):
        result = find_density_crossover(df, ['COL'])
    assert 'COL' not in result


# ===== 引数の効果テスト =====
def test_bandwidth_affects_result():
    """bandwidth を大きくすると滑らかになり、交点の数・位置が変わる."""
    df = _make_overlap()
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        res_small = find_density_crossover(df, ['OVERLAP'], bandwidth=0.03)
        res_large = find_density_crossover(df, ['OVERLAP'], bandwidth=0.3)
    assert res_small.get('OVERLAP') != res_large.get('OVERLAP')


def test_x_min_x_max_constrains_grid():
    """x_min/x_max を指定すると、その範囲内の交点が返る."""
    df = _make_separated()
    result = find_density_crossover(df, ['EXT'], x_min=0.0, x_max=1.0)
    assert 'EXT' in result
    assert 0.0 <= result['EXT'] <= 1.0


# ===== 入力チェックテスト =====
def test_error_on_missing_target():
    """target 列が存在しない場合、KeyError を投げる."""
    df = _make_separated().drop(columns=['TARGET'])
    with pytest.raises(KeyError):
        find_density_crossover(df, ['EXT'])


def test_error_on_missing_column():
    """指定列が存在しない場合、KeyError を投げる."""
    df = _make_separated()
    with pytest.raises(KeyError):
        find_density_crossover(df, ['NOT_EXIST'])


def test_error_on_empty_df():
    """空の DataFrame の場合、ValueError を投げる."""
    df = pd.DataFrame({'EXT': [], 'TARGET': []})
    with pytest.raises(ValueError):
        find_density_crossover(df, ['EXT'])
