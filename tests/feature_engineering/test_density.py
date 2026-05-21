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


def _
