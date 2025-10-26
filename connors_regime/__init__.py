"""
Connors Regime - Market Regime Detection

A library for detecting market regimes using various algorithmic methods.
"""

from connors_regime.core.market_regime import (
    BaseRegimeDetector,
    RegimeDetection,
    RegimeDetector,
    RegimeMethod,
    RegimeResult,
    RegimeType,
    RuleBasedRegimeDetector,
)
from connors_regime.core.registry import registry
from connors_regime.services.regime_service import (
    RegimeDetectionRequest,
    RegimeService,
    RegimeServiceResult,
)
from connors_regime.version import __version__

__all__ = [
    "RegimeType",
    "RegimeMethod",
    "RegimeDetection",
    "RegimeResult",
    "RegimeDetector",
    "BaseRegimeDetector",
    "RuleBasedRegimeDetector",
    "RegimeService",
    "RegimeDetectionRequest",
    "RegimeServiceResult",
    "registry",
    "__version__",
]
