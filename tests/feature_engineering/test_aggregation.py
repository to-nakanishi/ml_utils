"""aggregate_table のテスト."""
import pytest
import numpy as np
import pandas as pd
from ml_utils.feature_engineering.aggregation import aggregate_table


def _make_child(seed=42):
    """メイン/サブテーブル(bureau風)を生成。SK 101=3件, 102=3件, 103=2件 の計8行."""
    return pd.DataFrame({
        'SK_ID_CURR': [101, 101, 101, 102, 102, 102, 103, 103],
        'DAYS_CREDIT': [-541, -905, -1879, -1913, -313, -1231, -1615, -816],
        'AMT_CREDIT_SUM': [475850.0, 303342.0, 86437.0,
                           173517.0, 328935.0, 363779.0, 99094.0, 159078.0],
        'CREDIT_ACTIVE': ['Active', 'Active', 'Active',
                          'Closed', 'Active', 'Closed', 'Active', 'Closed'],
    })


# ===== 基本動作テスト =====
def test_returns_dataframe():
    """戻り値は DataFrame."""
    df = _make_child()
    result = aggregate_table(df, 'SK_ID_CURR', 'DAYS_CREDIT', 'BUREAU_')
    assert isinstance(result, pd.DataFrame)


def test_one_row_per_group_key():
    """行数はユニークな group_key 数に畳まれる(1対多 → 1対1)."""
    df = _make_child()
    result = aggregate_table(df, 'SK_ID_CURR', 'DAYS_CREDIT', 'BUREAU_')
    assert len(result) == df['SK_ID_CURR'].nunique()
    assert set(result['SK_ID_CURR']) == {101, 102, 103}


def test_prefix_applied_to_columns():
    """生成列(group_key 以外)に prefix が付いている."""
    df = _make_child()
    result = aggregate_table(df, 'SK_ID_CURR', 'DAYS_CREDIT', 'BUREAU_')
    feature_cols = [c for c in result.columns if c != 'SK_ID_CURR']
    assert all(c.startswith('BUREAU_') for c in feature_cols)


# ===== 集約値の正しさテスト =====
def test_count_matches_record_number():
    """sort_col の count が各 group のレコード件数と一致する."""
    df = _make_child()
    result = aggregate_table(df, 'SK_ID_CURR', 'DAYS_CREDIT', 'BUREAU_')
    counts = result.set_index('SK_ID_CURR')['BUREAU_DAYS_CREDIT_count']
    assert counts.loc[101] == 3
    assert counts.loc[102] == 3
    assert counts.loc[103] == 2


def test_latest_picks_max_sort_col_row():
    """直近値(_latest)が sort_col 最大の行から取られる."""
    df = _make_child()
    result = aggregate_table(df, 'SK_ID_CURR', 'DAYS_CREDIT', 'BUREAU_')
    res = result.set_index('SK_ID_CURR')
    # SK101 の DAYS_CREDIT 最大は -541(その行の AMT は 475850)
    assert res.loc[101, 'BUREAU_AMT_CREDIT_SUM_latest'] == 475850.0
    # SK102 の DAYS_CREDIT 最大は -313(その行の AMT は 328935, ACTIVE)
    assert res.loc[102, 'BUREAU_AMT_CREDIT_SUM_latest'] == 328935.0
    assert res.loc[102, 'BUREAU_CREDIT_ACTIVE_latest'] == 'Active'


def test_std_nan_for_single_record_group():
    """レコードが1件のみの group は std が NaN(情報少のシグナル)."""
    df = pd.DataFrame({
        'SK_ID_CURR': [201, 202, 202, 202],
        'DAYS_CREDIT': [-100, -500, -300, -900],
        'AMT_CREDIT_SUM': [50000.0, 100000.0, 200000.0, 150000.0],
        'CREDIT_ACTIVE': ['Active', 'Closed', 'Active', 'Closed'],
    })
    result = aggregate_table(df, 'SK_ID_CURR', 'DAYS_CREDIT', 'X_').set_index('SK_ID_CURR')
    assert pd.isna(result.loc[201, 'X_AMT_CREDIT_SUM_std'])  # 1件 → NaN
    assert not pd.isna(result.loc[202, 'X_AMT_CREDIT_SUM_std'])  # 3件 → 値あり


def test_works_without_categorical_columns():
    """カテゴリ列が無い(数値のみ)テーブルでも落ちずに集約できる."""
    df = _make_child().drop(columns=['CREDIT_ACTIVE'])
    result = aggregate_table(df, 'SK_ID_CURR', 'DAYS_CREDIT', 'X_')
    assert len(result) == 3


# ===== 入力チェックテスト =====
def test_error_on_missing_prefix():
    """prefix を渡さない場合、TypeError(必須引数)を投げる."""
    df = _make_child()
    with pytest.raises(TypeError):
        aggregate_table(df, 'SK_ID_CURR', 'DAYS_CREDIT')


def test_error_on_missing_group_key():
    """group_key 列が存在しない場合、KeyError を投げる."""
    df = _make_child()
    with pytest.raises(KeyError):
        aggregate_table(df, 'NOT_EXIST', 'DAYS_CREDIT', 'X_')


def test_error_on_missing_sort_col():
    """sort_col 列が存在しない場合、KeyError を投げる."""
    df = _make_child()
    with pytest.raises(KeyError):
        aggregate_table(df, 'SK_ID_CURR', 'NOT_EXIST', 'X_')


def test_error_on_nan_group_key():
    """group_key に NaN が含まれる場合、ValueError を投げる."""
    df = _make_child()
    df.loc[0, 'SK_ID_CURR'] = np.nan
    with pytest.raises(ValueError):
        aggregate_table(df, 'SK_ID_CURR', 'DAYS_CREDIT', 'X_')


def test_error_on_empty_df():
    """空の DataFrame の場合、ValueError を投げる."""
    df = _make_child().iloc[0:0]
    with pytest.raises(ValueError):
        aggregate_table(df, 'SK_ID_CURR', 'DAYS_CREDIT', 'X_')
