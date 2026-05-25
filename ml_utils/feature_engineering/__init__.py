"""Feature engineering utilities."""

from .aggregation import aggregate_table, diagnose_aggregation
from .composite import make_composite_features
from .density import find_density_crossover
from .imputation import impute_by_target_rate
from .memory import downcast_numeric
from .target_encoding import diagnose_target_encoding, target_encode_oof

__all__ = [
    'make_composite_features',
    'impute_by_target_rate',
    'downcast_numeric',
    'target_encode_oof',
    'diagnose_target_encoding',
    'find_density_crossover',
    'aggregate_table',
    'diagnose_aggregation',
]
