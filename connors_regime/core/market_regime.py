"""
Core Market Regime Detection interfaces and calculation methods

This module defines the core interfaces and the built-in composite
regime detector. The composite detector classifies two independent
axes — trend and volatility — and maps their combination onto a
single regime label with hysteresis to avoid regime flickering.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


class RegimeType(Enum):
    """Enumeration of market regime types"""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class TrendState(Enum):
    """Trend axis of the composite classification"""

    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"


class VolatilityState(Enum):
    """Volatility axis of the composite classification"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class RegimeMethod(Enum):
    """Enumeration of built-in regime detection methods"""

    COMPOSITE = "composite"


@dataclass
class RegimeDetection:
    """Container for a single regime detection result"""

    date: pd.Timestamp
    regime: RegimeType
    confidence: float  # Confidence level (0.0 to 1.0)
    method: RegimeMethod
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RegimeResult:
    """Container for market regime detection results"""

    ticker: str
    data: pd.DataFrame  # Original OHLCV data with regime columns added
    detections: List[RegimeDetection]  # Regime detections over time
    method: RegimeMethod
    parameters: Dict[str, Any]
    calculation_time: float
    current_regime: RegimeType
    regime_transitions: List[Dict[str, Any]]  # List of regime transitions
    success: bool = True
    error: Optional[str] = None


class RegimeDetector(Protocol):
    """Protocol for market regime detectors"""

    def detect(self, data: pd.DataFrame, **params: Any) -> RegimeResult:
        """Detect market regimes from OHLCV data"""
        ...

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for this detection method"""
        ...

    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available parameters"""
        ...


class BaseRegimeDetector(ABC):
    """Base class for market regime detectors"""

    def __init__(self, method: RegimeMethod):
        self.method = method

    @abstractmethod
    def detect(self, data: pd.DataFrame, **params: Any) -> RegimeResult:
        """Detect market regimes"""
        pass

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters"""
        pass

    @abstractmethod
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter information"""
        pass

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input OHLCV data"""
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if len(data) < 30:  # Minimum data points needed for regime detection
            raise ValueError(
                "Insufficient data points for regime detection (minimum: 30)"
            )


class CompositeRegimeDetector(BaseRegimeDetector):
    """
    Composite trend x volatility regime detector.

    Classifies two independent axes per bar:

    - Trend (bull / neutral / bear): price relative to a long moving
      average combined with the sign of the rolling return.
    - Volatility (low / normal / high / extreme): the current annualized
      volatility's percentile rank within the asset's own trailing
      history, so thresholds self-calibrate across assets (SPY, TSLA,
      BTC) without per-asset tuning.

    The two axes are mapped onto the RegimeType labels:

    - Bear trend with extreme volatility, or a deep drawdown with
      elevated volatility -> CRISIS
    - Positive rolling return while still below the long moving average
      after a drawdown -> RECOVERY
    - Bull / bear trend -> BULL / BEAR (volatility kept in metadata)
    - Neutral trend -> HIGH_VOLATILITY / LOW_VOLATILITY / SIDEWAYS
      depending on the volatility state

    A hysteresis filter commits a regime change only after the new raw
    label persists for `confirm_days` consecutive bars (crisis commits
    after `crisis_confirm_days`), which prevents day-to-day flickering.
    """

    def __init__(self):
        super().__init__(RegimeMethod.COMPOSITE)

    def detect(self, data: pd.DataFrame, **params: Any) -> RegimeResult:
        """Detect market regimes using the composite trend/volatility rules"""
        start_time = time.time()

        try:
            self._validate_data(data)

            parameters = self.get_default_parameters()
            parameters.update(params)

            df = self._calculate_features(data, **parameters)

            raw = self._classify_raw(df, parameters)
            detections, transitions = self._apply_hysteresis(df, raw, parameters)

            df["regime"] = pd.Series(dtype="object", index=df.index)
            df["regime_confidence"] = pd.Series(dtype="float64", index=df.index)
            for detection in detections:
                df.loc[detection.date, "regime"] = detection.regime.value
                df.loc[detection.date, "regime_confidence"] = detection.confidence

            current_regime = (
                detections[-1].regime if detections else RegimeType.SIDEWAYS
            )

            return RegimeResult(
                ticker=getattr(data, "ticker", "UNKNOWN"),
                data=df,
                detections=detections,
                method=self.method,
                parameters=parameters,
                calculation_time=time.time() - start_time,
                current_regime=current_regime,
                regime_transitions=transitions,
                success=True,
            )

        except Exception as e:
            return RegimeResult(
                ticker=getattr(data, "ticker", "UNKNOWN"),
                data=data,
                detections=[],
                method=self.method,
                parameters=params,
                calculation_time=time.time() - start_time,
                current_regime=RegimeType.SIDEWAYS,
                regime_transitions=[],
                success=False,
                error=str(e),
            )

    def _calculate_features(self, data: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """Calculate trend, volatility, and drawdown features"""
        df = data.copy()

        return_window = params["return_window"]
        volatility_window = params["volatility_window"]
        vol_percentile_window = params["vol_percentile_window"]
        trend_window = params["trend_window"]

        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df[f"return_{return_window}d"] = df["log_returns"].rolling(return_window).sum()

        # Short-window return for recovery detection: rebounds off a low
        # move faster than trends, so a long window would keep the
        # preceding crash in view and miss the turn
        recovery_window = max(return_window // 3, 20)
        df["return_recovery"] = df["log_returns"].rolling(recovery_window).sum()

        # Annualized volatility; percentile rank within trailing history
        # makes the low/high buckets self-calibrating per asset
        vol_col = f"volatility_{volatility_window}d"
        df[vol_col] = df["log_returns"].rolling(volatility_window).std() * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
        df["vol_percentile"] = (
            df[vol_col]
            .rolling(vol_percentile_window, min_periods=params["vol_min_history"])
            .rank(pct=True)
        )

        # min_periods lets shorter datasets still get a trend estimate,
        # at reduced reliability
        df["sma_trend"] = (
            df["Close"]
            .rolling(trend_window, min_periods=max(trend_window // 4, 20))
            .mean()
        )
        df["price_vs_sma"] = (df["Close"] - df["sma_trend"]) / df["sma_trend"]

        rolling_peak = df["Close"].rolling(TRADING_DAYS_PER_YEAR, min_periods=30).max()
        df["drawdown"] = df["Close"] / rolling_peak - 1

        return df

    def _classify_raw(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> "pd.Series[Any]":
        """Classify each bar independently (before hysteresis).

        Returns a Series of (RegimeType, TrendState, VolatilityState)
        tuples, NaN where features are not yet available.
        """
        return_col = f"return_{params['return_window']}d"

        labels = pd.Series(index=df.index, dtype="object")

        for idx, row in df.iterrows():
            if (
                pd.isna(row[return_col])
                or pd.isna(row["vol_percentile"])
                or pd.isna(row["price_vs_sma"])
            ):
                continue

            trend = self._classify_trend(
                row["price_vs_sma"], row[return_col], params["trend_threshold"]
            )
            vol_state = self._classify_volatility(row["vol_percentile"], params)
            regime = self._map_to_regime(
                trend,
                vol_state,
                price_vs_sma=row["price_vs_sma"],
                recovery_return=row["return_recovery"],
                drawdown=row["drawdown"],
                params=params,
            )
            labels.loc[idx] = (regime, trend, vol_state)

        return labels

    @staticmethod
    def _classify_trend(
        price_vs_sma: float, rolling_return: float, trend_threshold: float
    ) -> TrendState:
        if price_vs_sma > trend_threshold and rolling_return > 0:
            return TrendState.BULL
        if price_vs_sma < -trend_threshold and rolling_return < 0:
            return TrendState.BEAR
        return TrendState.NEUTRAL

    @staticmethod
    def _classify_volatility(
        vol_percentile: float, params: Dict[str, Any]
    ) -> VolatilityState:
        if vol_percentile >= params["vol_extreme_pct"]:
            return VolatilityState.EXTREME
        if vol_percentile >= params["vol_high_pct"]:
            return VolatilityState.HIGH
        if vol_percentile <= params["vol_low_pct"]:
            return VolatilityState.LOW
        return VolatilityState.NORMAL

    @staticmethod
    def _map_to_regime(
        trend: TrendState,
        vol_state: VolatilityState,
        price_vs_sma: float,
        recovery_return: float,
        drawdown: float,
        params: Dict[str, Any],
    ) -> RegimeType:
        elevated_vol = vol_state in (VolatilityState.HIGH, VolatilityState.EXTREME)

        if (drawdown <= params["crisis_drawdown"] and elevated_vol) or (
            trend == TrendState.BEAR and vol_state == VolatilityState.EXTREME
        ):
            return RegimeType.CRISIS

        if (
            price_vs_sma < 0
            and drawdown <= params["recovery_drawdown"]
            and recovery_return >= params["recovery_return_threshold"]
        ):
            return RegimeType.RECOVERY

        if trend == TrendState.BULL:
            return RegimeType.BULL
        if trend == TrendState.BEAR:
            return RegimeType.BEAR

        # Neutral trend: label by volatility state
        if elevated_vol:
            return RegimeType.HIGH_VOLATILITY
        if vol_state == VolatilityState.LOW:
            return RegimeType.LOW_VOLATILITY
        return RegimeType.SIDEWAYS

    def _apply_hysteresis(
        self,
        df: pd.DataFrame,
        raw_labels: "pd.Series[Any]",
        params: Dict[str, Any],
    ) -> "tuple[List[RegimeDetection], List[Dict[str, Any]]]":
        """Commit regime changes only after they persist.

        A new raw label must repeat for `confirm_days` consecutive bars
        before it replaces the committed regime (`crisis_confirm_days`
        for crisis, so genuine market stress is flagged quickly).
        Confidence is the share of recent raw labels that agree with the
        committed regime.
        """
        confirm_days = params["confirm_days"]
        crisis_confirm_days = params["crisis_confirm_days"]
        return_col = f"return_{params['return_window']}d"
        vol_col = f"volatility_{params['volatility_window']}d"

        detections: List[RegimeDetection] = []
        transitions: List[Dict[str, Any]] = []

        committed: Optional[RegimeType] = None
        pending: Optional[RegimeType] = None
        pending_count = 0
        recent_raw: List[RegimeType] = []

        for idx, label in raw_labels.items():
            if not isinstance(label, tuple):
                continue

            raw_regime, trend, vol_state = label

            recent_raw.append(raw_regime)
            if len(recent_raw) > max(confirm_days * 2, 10):
                recent_raw.pop(0)

            if committed is None:
                committed = raw_regime
            elif raw_regime == committed:
                pending = None
                pending_count = 0
            else:
                if pending is not None and raw_regime == pending:
                    pending_count += 1
                else:
                    pending = raw_regime
                    pending_count = 1

                required = (
                    crisis_confirm_days
                    if pending == RegimeType.CRISIS
                    else confirm_days
                )
                if pending_count >= required:
                    transitions.append(
                        {
                            "date": idx,
                            "from_regime": committed.value,
                            "to_regime": pending.value,
                            "confidence": pending_count / max(len(recent_raw), 1),
                        }
                    )
                    committed = pending
                    pending = None
                    pending_count = 0

            agreement = sum(1 for r in recent_raw if r == committed) / len(recent_raw)
            confidence = float(np.clip(agreement, 0.1, 0.95))

            row = df.loc[idx]
            detections.append(
                RegimeDetection(
                    date=idx,
                    regime=committed,
                    confidence=confidence,
                    method=self.method,
                    metadata={
                        "raw_regime": raw_regime.value,
                        "trend": trend.value,
                        "volatility_state": vol_state.value,
                        "return_value": row[return_col],
                        "volatility_value": row[vol_col],
                        "vol_percentile": row["vol_percentile"],
                        "price_vs_sma": row["price_vs_sma"],
                        "drawdown": row["drawdown"],
                    },
                )
            )

        return detections, transitions

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for composite regime detection"""
        return {
            "return_window": 60,  # Bars for rolling return (trend strength)
            "volatility_window": 20,  # Bars for annualized volatility
            "vol_percentile_window": 252,  # Trailing history for vol percentile
            "vol_min_history": 60,  # Min bars before vol percentile is valid
            "trend_window": 200,  # Long moving average for trend axis
            "trend_threshold": 0.02,  # Min |price vs SMA| to call a trend
            "vol_low_pct": 0.20,  # Vol percentile <= this -> low vol
            "vol_high_pct": 0.80,  # Vol percentile >= this -> high vol
            "vol_extreme_pct": 0.95,  # Vol percentile >= this -> extreme vol
            "crisis_drawdown": -0.20,  # Drawdown for crisis classification
            "recovery_drawdown": -0.10,  # Min drawdown for recovery context
            "recovery_return_threshold": 0.05,  # Rolling return for recovery
            "confirm_days": 5,  # Bars a new regime must persist
            "crisis_confirm_days": 2,  # Faster confirmation for crisis
        }

    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available parameters"""
        return {
            "return_window": {
                "type": "int",
                "default": 60,
                "min": 20,
                "max": 252,
                "description": "Number of bars for rolling return calculation",
            },
            "volatility_window": {
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 100,
                "description": "Number of bars for annualized volatility calculation",
            },
            "vol_percentile_window": {
                "type": "int",
                "default": 252,
                "min": 60,
                "max": 756,
                "description": "Trailing bars used to rank current volatility",
            },
            "vol_min_history": {
                "type": "int",
                "default": 60,
                "min": 30,
                "max": 252,
                "description": "Minimum bars of history before volatility percentile is valid",
            },
            "trend_window": {
                "type": "int",
                "default": 200,
                "min": 50,
                "max": 300,
                "description": "Moving average window for the trend axis",
            },
            "trend_threshold": {
                "type": "float",
                "default": 0.02,
                "min": 0.0,
                "max": 0.10,
                "description": "Minimum price distance from trend SMA to classify bull/bear",
            },
            "vol_low_pct": {
                "type": "float",
                "default": 0.20,
                "min": 0.05,
                "max": 0.40,
                "description": "Volatility percentile at or below which volatility is low",
            },
            "vol_high_pct": {
                "type": "float",
                "default": 0.80,
                "min": 0.60,
                "max": 0.95,
                "description": "Volatility percentile at or above which volatility is high",
            },
            "vol_extreme_pct": {
                "type": "float",
                "default": 0.95,
                "min": 0.85,
                "max": 1.0,
                "description": "Volatility percentile at or above which volatility is extreme",
            },
            "crisis_drawdown": {
                "type": "float",
                "default": -0.20,
                "min": -0.50,
                "max": -0.10,
                "description": "Drawdown from trailing peak that qualifies as crisis (with elevated volatility)",
            },
            "recovery_drawdown": {
                "type": "float",
                "default": -0.10,
                "min": -0.40,
                "max": -0.05,
                "description": "Minimum remaining drawdown for recovery classification",
            },
            "recovery_return_threshold": {
                "type": "float",
                "default": 0.05,
                "min": 0.01,
                "max": 0.20,
                "description": "Minimum rolling return for recovery classification",
            },
            "confirm_days": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 20,
                "description": "Consecutive bars a new regime must persist before committing",
            },
            "crisis_confirm_days": {
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 10,
                "description": "Consecutive bars before committing a crisis regime",
            },
        }
