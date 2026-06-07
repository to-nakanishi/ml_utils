"""Tests for optimize_threshold."""

import numpy as np
import pytest
from sklearn.metrics import f1_score, recall_score

from ml_utils.evaluation.threshold import optimize_threshold


def _separable_data(n=200, seed=0):
    """確率がラベルときれいに対応する合成データ。"""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # 正例は高確率、負例は低確率(ノイズあり)に寄せる
    prob = np.where(y == 1, rng.uniform(0.55, 1.0, n), rng.uniform(0.0, 0.45, n))
    return y, prob


def test_returns_three_part_tuple():
    y, prob = _separable_data()
    result = optimize_threshold(y, prob)
    assert isinstance(result, tuple) and len(result) == 3
    thr, score, grid = result
    assert 0.0 <= thr <= 1.0
    assert isinstance(grid, list)


def test_grid_length_matches_n_thresholds():
    y, prob = _separable_data()
    _, _, grid = optimize_threshold(y, prob, n_thresholds=51)
    assert len(grid) == 51
    # 各要素は (threshold, score)
    assert all(len(item) == 2 for item in grid)


def test_best_threshold_actually_maximizes_metric():
    """返された best_score が grid 内の最大スコアと一致すること。"""
    y, prob = _separable_data()
    best_thr, best_score, grid = optimize_threshold(y, prob)
    grid_scores = [s for _, s in grid]
    assert np.isclose(best_score, max(grid_scores))
    # best_thr のスコアが best_score
    thr_to_score = dict(grid)
    assert np.isclose(thr_to_score[best_thr], best_score)


def test_default_metric_is_macro_f1():
    """既定 metric が macro-F1 であること(手計算と一致)。"""
    y, prob = _separable_data()
    best_thr, best_score, _ = optimize_threshold(y, prob)
    y_pred = (prob >= best_thr).astype(int)
    assert np.isclose(best_score, f1_score(y, y_pred, average='macro'))


def test_custom_metric_recall():
    """metric を Recall に差し替えると、Recall を最大化する閾値が選ばれる。"""
    y, prob = _separable_data()
    best_thr, best_score, _ = optimize_threshold(
        y, prob, metric=lambda yt, yp: recall_score(yt, yp, zero_division=0)
    )
    # 閾値0(全部1と予測)なら Recall=1。最大化なら best_score は 1.0 になるはず
    assert np.isclose(best_score, 1.0)


def test_separable_data_finds_sensible_threshold():
    """正例>=0.55, 負例<=0.45 の分離データなら、最適閾値はその間に入る。"""
    y, prob = _separable_data(n=500)
    best_thr, best_score, _ = optimize_threshold(y, prob)
    assert 0.45 <= best_thr <= 0.56
    assert best_score > 0.95  # きれいに分かれるので高スコア


def test_accepts_list_input():
    """array-like(list)でも動くこと。"""
    y = [0, 0, 1, 1]
    prob = [0.1, 0.2, 0.8, 0.9]
    thr, score, grid = optimize_threshold(y, prob)
    assert 0.0 <= thr <= 1.0


def test_guard_empty():
    with pytest.raises(ValueError):
        optimize_threshold([], [])


def test_guard_length_mismatch():
    with pytest.raises(ValueError):
        optimize_threshold([0, 1, 1], [0.1, 0.2])


def test_guard_too_few_thresholds():
    y, prob = _separable_data()
    with pytest.raises(ValueError):
        optimize_threshold(y, prob, n_thresholds=1)