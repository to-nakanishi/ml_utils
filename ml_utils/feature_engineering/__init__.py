"""Feature engineering utilities."""

from .memory import downcast_numeric
from .target_encoding import target_encode_oof, diagnose_target_encoding
from .composite import make_composite_features

__all__ = [
    'downcast_numeric',
    'target_encode_oof',
    'diagnose_target_encoding',
    'make_composite_features',
]
