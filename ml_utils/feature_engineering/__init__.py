"""Feature engineering utilities."""

from .composite import make_composite_features
from .imputation import impute_by_target_rate
from .memory import downcast_numeric
from .target_encoding import target_encode_oof, diagnose_target_encoding

__all__ = [
    'make_composite_features',
    'impute_by_target_rate',
    'downcast_numeric',
    'target_encode_oof',
    'diagnose_target_encoding',
]
