"""impute_by_target_rate のテスト."""
import pytest
import numpy as np
import pandas as pd
from ml_utils.feature_engineering.imputation import impute_by_target_rate


def _make_train(n=200, missing=20, seed=42):
    """補完テスト用の train を生成(欠損 missing 件を含む)."""
    rng = np.random.RandomState(seed)
    col = rng.uniform(0, 1, size=n)
    target = rng.binomial(1, 0.3, size=n)
    df = pd.DataFrame({'EXT': col, 'TARGET': target})
    # 先頭 missing 件を NaN にする
    df.loc[:missing - 1, 'EXT'] = np.nan
    return df


# ===== 基本動作テスト =====
def test_returns_tuple_of_dataframes():
    """戻り値は (train, test) の DataFrame タプル."""
    train = _make_train()
    test = pd.DataFrame({'EXT': [0.5, np.nan, 0.3]})
    result = impute_by_target_rate(train, test, 'EXT')
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], pd.DataFrame)
    assert isinstance(result[1], pd.DataFrame)


def test_imputed_column_added():
    """補完列 `{col}_IMPUTED` が train/test 両方に追加される."""
    train = _make_train()
    test = pd.DataFrame({'EXT': [0.5, np.nan, 0.3]})
    train_imp, test_imp = impute_by_target_rate(train, test, 'EXT')
    assert 'EXT_IMPUTED' in train_imp.columns
    assert 'EXT_IMPUTED' in test_imp.columns


def test_original_df_not_modified():
    """入力 DataFrame は変更されない(copy 保証)."""
    train = _make_train()
    test = pd.DataFrame({'EXT': [0.5, np.nan, 0.3]})
    train_cols_before = train.columns.tolist()
    test_cols_before = test.columns.tolist()
    impute_by_target_rate(train, test, 'EXT')
    assert train.columns.tolist() == train_cols_before
    assert test.columns.tolist() == test_cols_before


def test_no_nan_in_imputed_column():
    """補完後の `_IMPUTED` 列に NaN が残らない."""
    train = _make_train()
    test = pd.DataFrame({'EXT': [0.5, np.nan, 0.3]})
    train_imp, test_imp = impute_by_target_rate(train, test, 'EXT')
    assert train_imp['EXT_IMPUTED'].isna().sum() == 0
    assert test_imp['EXT_IMPUTED'].isna().sum() == 0


# ===== 動作仕様テスト =====
def test_non_missing_values_preserved():
    """欠損でない値は元のまま `_IMPUTED` にコピーされる."""
    train = _make_train()
    test = pd.DataFrame({'EXT': [0.5, np.nan, 0.3]})
    train_imp, _ = impute_by_target_rate(train, test, 'EXT')
    # 元が非欠損の行は、_IMPUTED でも同じ値
    non_missing_mask = train['EXT'].notna()
    assert np.allclose(
        train_imp.loc[non_missing_mask, 'EXT_IMPUTED'],
        train.loc[non_missing_mask, 'EXT'],
    )


def test_missing_flag_added_when_true():
    """add_missing_flag=True でフラグ列が追加され、値が正しい."""
    train = _make_train()
    test = pd.DataFrame({'EXT': [0.5, np.nan, 0.3]})
    train_imp, test_imp = impute_by_target_rate(
        train, test, 'EXT', add_missing_flag=True
    )
    assert 'EXT_WAS_MISSING' in train_imp.columns
    assert 'EXT_WAS_MISSING' in test_imp.columns
    # フラグは元の欠損位置と一致
    assert (train_imp['EXT_WAS_MISSING'] == train['EXT'].isna().astype(int)).all()
    # test: 行1 が欠損 → フラグ 1
    assert test_imp['EXT_WAS_MISSING'].tolist() == [0, 1, 0]


def test_missing_flag_not_added_by_default():
    """デフォルト(add_missing_flag=False)でフラグ列は追加されない."""
    train = _make_train()
    test = pd.DataFrame({'EXT': [0.5, np.nan, 0.3]})
    train_imp, test_imp = impute_by_target_rate(train, test, 'EXT')
    assert 'EXT_WAS_MISSING' not in train_imp.columns
    assert 'EXT_WAS_MISSING' not in test_imp.columns


def test_error_on_missing_column():
    """指定列が存在しない場合、KeyError を投げる."""
    train = _make_train()
    test = pd.DataFrame({'EXT': [0.5, 0.3]})
    with pytest.raises(KeyError):
        impute_by_target_rate(train, test, 'NOT_EXIST')


# ===== 本命テスト(欠損0件の挙動 / 補完値の正しさ) =====
def test_no_columns_when_no_missing():
    """欠損が0件の場合、列を一切作らない."""
    train = pd.DataFrame({
        'EXT': np.random.RandomState(0).uniform(0, 1, size=100),
        'TARGET': np.random.RandomState(1).binomial(1, 0.3, size=100),
    })
    test = pd.DataFrame({'EXT': [0.5, 0.3]})  # 欠損なし
    train_imp, test_imp = impute_by_target_rate(
        train, test, 'EXT', add_missing_flag=True
    )
    # 補完列もフラグ列も作られない
    assert 'EXT_IMPUTED' not in train_imp.columns
    assert 'EXT_WAS_MISSING' not in train_imp.columns
    assert 'EXT_IMPUTED' not in test_imp.columns


def test_imputed_value_matches_nearest_bin():
    """補完値が、欠損グループのデフォルト率に最も近いビンの median になっている."""
    rng = np.random.RandomState(123)
    # 値が小さいほどデフォルト率が高い、という構造を作る
    n = 300
    ext = rng.uniform(0, 1, size=n)
    # ext が小さいほど TARGET=1 になりやすい
    target = (rng.uniform(0, 1, size=n) > ext).astype(int)
    train = pd.DataFrame({'EXT': ext, 'TARGET': target})
    # 欠損行を追加: デフォルト率が高い(=低 EXT 相当)グループとして 30 件
    missing_rows = pd.DataFrame({'EXT': [np.nan] * 30, 'TARGET': [1] * 25 + [0] * 5})
    train = pd.concat([train, missing_rows], ignore_index=True)
    test = pd.DataFrame({'EXT': [np.nan]})

    train_imp, test_imp = impute_by_target_rate(train, test, 'EXT', n_bins=5)
    impute_value = test_imp['EXT_IMPUTED'].iloc[0]

    # 欠損グループのデフォルト率は 25/30 ≈ 0.83 と高い
    # → 低 EXT(デフォルト率が高い)ビンの median が選ばれるはず
    # 補完値が全体の median より小さいことを確認(高デフォルト率 = 低 EXT 側)
    overall_median = train['EXT'].median()
    assert impute_value < overall_median
