"""run_baseline のテスト."""
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from ml_utils.modeling.baseline import run_baseline
from ml_utils.validation.splitters import SlidingWindowSplit

def _make_xy(n=800, seed=42):
    """signal が target を作る人工データ + ノイズ + カテゴリ列(欠損あり)."""
    rng = np.random.RandomState(seed)
    sig = rng.normal(0, 1, n)
    y = pd.Series((sig + rng.normal(0, 1, n) > 0).astype(int))
    X = pd.DataFrame({
        'signal': sig,
        'noise': rng.normal(0, 1, n),
        'cat': rng.choice(['A', 'B', 'C'], n),
    })
    X.loc[:49, 'cat'] = np.nan
    X['cat'] = X['cat'].astype('category')
    return X, y


# ===== 基本動作 =====
def test_returns_dict_with_expected_keys():
    """戻り値は dict で、必要なキーが揃っている."""
    X, y = _make_xy()
    res = run_baseline(X, y, n_splits=3)
    assert isinstance(res, dict)
    for k in ['lgb_auc', 'cat_auc', 'lgb_importance', 'cat_importance',
              'comparison', 'lgb_model', 'cat_model']:
        assert k in res


def test_oof_auc_in_range_and_learns_signal():
    """OOF AUC が 0.5〜1.0 の範囲(signal を学習できている)."""
    X, y = _make_xy()
    res = run_baseline(X, y, n_splits=3)
    assert 0.5 < res['lgb_auc'] <= 1.0
    assert 0.5 < res['cat_auc'] <= 1.0


def test_importance_dataframe_shape():
    """重要度は feature 列 + pct 列の DataFrame で、全特徴量分の行を持つ."""
    X, y = _make_xy()
    res = run_baseline(X, y, n_splits=3)
    assert list(res['lgb_importance'].columns) == ['feature', 'LGBM_pct']
    assert list(res['cat_importance'].columns) == ['feature', 'CAT_pct']
    assert len(res['lgb_importance']) == X.shape[1]


def test_comparison_present_when_both_models():
    """両モデルを回したとき comparison が両重要度を含む."""
    X, y = _make_xy()
    res = run_baseline(X, y, n_splits=3)
    assert res['comparison'] is not None
    assert 'LGBM_pct' in res['comparison'].columns
    assert 'CAT_pct' in res['comparison'].columns


# ===== 引数の効果 =====
def test_single_model_lgb_only():
    """models=('lgb',) のとき cat 系と comparison は None."""
    X, y = _make_xy()
    res = run_baseline(X, y, models=('lgb',), n_splits=3)
    assert res['lgb_auc'] is not None
    assert res['cat_auc'] is None
    assert res['cat_importance'] is None
    assert res['comparison'] is None


def test_return_models_false():
    """return_models=False のとき model は None、AUC は出る."""
    X, y = _make_xy()
    res = run_baseline(X, y, n_splits=3, return_models=False)
    assert res['lgb_model'] is None
    assert res['cat_model'] is None
    assert res['lgb_auc'] is not None


def test_return_models_true_returns_fitted():
    """return_models=True のとき全データ学習モデルが返る."""
    X, y = _make_xy()
    res = run_baseline(X, y, n_splits=3, return_models=True)
    assert res['lgb_model'] is not None
    assert res['cat_model'] is not None


def test_custom_cv_object():
    """cv に CV オブジェクトを渡せる(TimeSeriesSplit)."""
    X, y = _make_xy()
    res = run_baseline(X, y, models=('lgb',), cv=TimeSeriesSplit(n_splits=3))
    assert res['lgb_auc'] is not None


def test_params_merge_override():
    """lgb_params の部分上書き(マージ)が効く."""
    X, y = _make_xy()
    res = run_baseline(X, y, models=('lgb',), n_splits=3,
                       lgb_params={'n_estimators': 50}, return_models=False)
    assert res['lgb_auc'] is not None


# ===== 元データ不変 =====
def test_original_X_not_modified():
    """元の X を変更しない(Unknown 補完が漏れない)."""
    X, y = _make_xy()
    nan_before = X['cat'].isna().sum()
    dtype_before = X['cat'].dtype
    cols_before = X.columns.tolist()
    run_baseline(X, y, n_splits=3, return_models=False)
    assert X['cat'].isna().sum() == nan_before
    assert X['cat'].dtype == dtype_before
    assert X.columns.tolist() == cols_before


# ===== 入力チェック =====
def test_error_on_empty_X():
    """空の X の場合、ValueError を投げる."""
    X, y = _make_xy()
    with pytest.raises(ValueError):
        run_baseline(X.iloc[0:0], y.iloc[0:0])


def test_error_on_length_mismatch():
    """X と y の長さが違う場合、ValueError を投げる."""
    X, y = _make_xy()
    with pytest.raises(ValueError):
        run_baseline(X, y.iloc[:100])


def test_error_on_invalid_model_name():
    """models に不正な名前が含まれる場合、ValueError を投げる."""
    X, y = _make_xy()
    with pytest.raises(ValueError):
        run_baseline(X, y, models=('xgb',))


def test_baseline_with_x_dependent_cv():
    """X 必須の CV(SlidingWindowSplit)でも動く(get_n_splits 非依存の回帰)."""
    rng = np.random.RandomState(42)
    rows = []
    for m in range(6):
        for _ in range(150):
            sig = rng.normal(0, 1)
            rows.append({'time_month': m, 'signal': sig,
                         'y': int(sig + rng.normal(0, 1) > 0)})
    df = pd.DataFrame(rows)
    y = df['y']
    X = df.drop(columns=['y'])
    tscv = SlidingWindowSplit('time_month', train_window=2)
    res = run_baseline(X, y, models=('lgb',), cv=tscv, return_models=False)
    assert res['lgb_auc'] is not None
    assert 0.0 < res['lgb_auc'] <= 1.0
