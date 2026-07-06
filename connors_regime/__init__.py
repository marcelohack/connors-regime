"""
Connors Regime - Market Regime Detection

A library for detecting market regimes using various algorithmic methods.
"""

from connors_regime.core.market_regime import (
    BaseRegimeDetector,
    CompositeRegimeDetector,
    RegimeDetection,
    RegimeDetector,
    RegimeMethod,
    RegimeResult,
    RegimeType,
    TrendState,
    VolatilityState,
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
    "TrendState",
    "VolatilityState",
    "RegimeDetection",
    "RegimeResult",
    "RegimeDetector",
    "BaseRegimeDetector",
    "CompositeRegimeDetector",
    "RegimeService",
    "RegimeDetectionRequest",
    "RegimeServiceResult",
    "registry",
    "__version__",
]
