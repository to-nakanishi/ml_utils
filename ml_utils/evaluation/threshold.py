"""Threshold optimization for binary classification."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import f1_score


def optimize_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    metric: Callable[[Any, Any], float] | None = None,
    n_thresholds: int = 101,
) -> tuple[float, float, list[tuple[float, float]]]:
    """
    ----------
    summary
    ----------
    2値分類の確率予測に対し、指定した metric を最大化する判定閾値を
    グリッドサーチで探索する。0〜1 を n_thresholds 個に等分割し、各閾値で
    `y_pred = (y_prob >= threshold)` として metric を計算、最大の閾値を返す。

    metric は `(y_true, y_pred) -> float` の関数で、呼び出し側が自由に
    渡せる(既定は macro-F1)。閾値で2値化した後の y_pred を受け取るため、
    F1 / Recall / Precision など閾値依存の指標を最適化できる
    (ROC-AUC のような閾値非依存の指標は対象外)。

    全閾値のスコア(results)も返すので、最適点の選択だけでなく
    「閾値を動かすとスコアがどう動くか」の感度分析にも使える。

    ----------
    Parameters
    ----------
    y_true : array-like
        真のラベル(0/1)。
    y_prob : array-like
        正例である確率(0〜1)。
    metric : Callable[[y_true, y_pred], float], optional
        最大化したい指標。`(y_true, y_pred)` を受けて float を返す関数。
        None の場合は macro-F1
        (`f1_score(y_true, y_pred, average='macro')`)。
    n_thresholds : int, default=101
        試す閾値の数。0〜1 を等分割する(既定 101 = 0.01 刻み)。

    ----------
    Returns
    ----------
    tuple[float, float, list[tuple[float, float]]]
        (best_threshold, best_score, results)
        - best_threshold : metric が最大となる閾値
        - best_score : その時の metric 値
        - results : [(threshold, score), ...] 全候補のスコア(感度分析用)

    ----------
    Examples
    ----------
    >>> thr, score, grid = optimize_threshold(y_true, y_prob)
    >>> # 金額加重リコールを最大化したい場合(metric を差し替え)
    >>> thr, score, grid = optimize_threshold(
    ...     y_true, y_prob,
    ...     metric=lambda yt, yp: amount_weighted_recall(yt, yp, amounts),
    ... )
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0:
        raise ValueError("y_true is empty.")
    if len(y_true) != len(y_prob):
        raise ValueError(
            f"y_true and y_prob length mismatch: {len(y_true)} vs {len(y_prob)}."
        )
    if n_thresholds < 2:
        raise ValueError(f"n_thresholds must be >= 2, got {n_thresholds}.")

    if metric is None:
        def metric(yt, yp):
            return f1_score(yt, yp, average='macro')

    thresholds = np.linspace(0, 1, n_thresholds)
    results: list[tuple[float, float]] = []
    best_threshold, best_score = float(thresholds[0]), -np.inf

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        score = float(metric(y_true, y_pred))
        results.append((float(thr), score))
        if score > best_score:
            best_threshold, best_score = float(thr), score

    return best_threshold, best_score, results