"""
Core Market Regime Detection interfaces and calculation methods

This module defines the core interfaces and calculation methods for
market regime identification using various algorithmic approaches.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd


class RegimeType(Enum):
    """Enumeration of market regime types"""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class RegimeMethod(Enum):
    """Enumeration of available regime detection methods"""

    RULE_BASED = "rule_based"
    # Future methods to be added:
    # CLUSTERING = "clustering"
    # HMM = "hmm"
    # REGIME_SWITCHING = "regime_switching"


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

    def _calculate_features(self, data: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """Calculate regime detection features from OHLCV data"""
        df = data.copy()

        # Extract parameters
        return_window = params.get("return_window", 60)
        volatility_window = params.get("volatility_window", 20)
        correlation_window = params.get("correlation_window", 20)

        # Calculate rolling returns
        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df[f"return_{return_window}d"] = df["log_returns"].rolling(return_window).sum()

        # Calculate rolling volatility
        df[f"volatility_{volatility_window}d"] = (
            df["log_returns"].rolling(volatility_window).std()
        )

        # Calculate price relative to moving average
        df["sma_200"] = df["Close"].rolling(200).mean()
        df["price_vs_sma"] = (df["Close"] - df["sma_200"]) / df["sma_200"]

        # Calculate volume relative to average
        df["volume_sma"] = df["Volume"].rolling(20).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_sma"]

        # Calculate RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        return df


class RuleBasedRegimeDetector(BaseRegimeDetector):
    """
    Simple Rule-Based Threshold regime detector

    Classifies market regimes based on configurable thresholds for:
    - Rolling returns (trend detection)
    - Rolling volatility (volatility regime)
    - Price vs moving average (trend strength)
    - Volume patterns
    """

    def __init__(self):
        super().__init__(RegimeMethod.RULE_BASED)

    def detect(self, data: pd.DataFrame, **params: Any) -> RegimeResult:
        """Detect market regimes using rule-based thresholds"""
        start_time = time.time()

        try:
            # Validate data
            self._validate_data(data)

            # Get parameters with defaults
            parameters = self.get_default_parameters()
            parameters.update(params)

            # Calculate features
            df = self._calculate_features(data, **parameters)

            # Apply rule-based classification
            regime_detections = []
            regime_transitions = []
            previous_regime = None

            # Extract thresholds
            bull_return_threshold = parameters["bull_return_threshold"]
            bear_return_threshold = parameters["bear_return_threshold"]
            high_vol_threshold = parameters["high_volatility_threshold"]
            low_vol_threshold = parameters["low_volatility_threshold"]
            crisis_return_threshold = parameters["crisis_return_threshold"]
            crisis_vol_threshold = parameters["crisis_volatility_threshold"]

            return_col = f"return_{parameters['return_window']}d"
            vol_col = f"volatility_{parameters['volatility_window']}d"

            for idx, row in df.iterrows():
                if pd.isna(row[return_col]) or pd.isna(row[vol_col]):
                    continue

                return_val = row[return_col]
                vol_val = row[vol_col]
                price_vs_sma = row.get("price_vs_sma", 0)

                # Rule-based classification
                regime = self._classify_regime(
                    return_val, vol_val, price_vs_sma, parameters
                )

                # Calculate confidence based on how far from thresholds
                confidence = self._calculate_confidence(
                    return_val, vol_val, regime, parameters
                )

                detection = RegimeDetection(
                    date=idx,
                    regime=regime,
                    confidence=confidence,
                    method=self.method,
                    metadata={
                        "return_value": return_val,
                        "volatility_value": vol_val,
                        "price_vs_sma": price_vs_sma,
                    },
                )

                regime_detections.append(detection)

                # Track regime transitions
                if previous_regime is not None and previous_regime != regime:
                    regime_transitions.append(
                        {
                            "date": idx,
                            "from_regime": previous_regime.value,
                            "to_regime": regime.value,
                            "confidence": confidence,
                        }
                    )

                previous_regime = regime

            # Add regime columns to dataframe with proper dtypes
            df["regime"] = pd.Series(dtype="object", index=df.index)
            df["regime_confidence"] = pd.Series(dtype="float64", index=df.index)

            # Fill in the regime data for rows with detections
            for detection in regime_detections:
                df.loc[detection.date, "regime"] = detection.regime.value
                df.loc[detection.date, "regime_confidence"] = detection.confidence

            # Determine current regime (last detection)
            current_regime = (
                regime_detections[-1].regime
                if regime_detections
                else RegimeType.SIDEWAYS
            )

            calculation_time = time.time() - start_time

            return RegimeResult(
                ticker=getattr(data, "ticker", "UNKNOWN"),
                data=df,
                detections=regime_detections,
                method=self.method,
                parameters=parameters,
                calculation_time=calculation_time,
                current_regime=current_regime,
                regime_transitions=regime_transitions,
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

    def _classify_regime(
        self,
        return_val: float,
        vol_val: float,
        price_vs_sma: float,
        params: Dict[str, Any],
    ) -> RegimeType:
        """Classify regime based on feature values and thresholds"""

        # Crisis detection (highest priority)
        if (
            return_val < params["crisis_return_threshold"]
            and vol_val > params["crisis_volatility_threshold"]
        ):
            return RegimeType.CRISIS

        # Recovery detection
        if (
            return_val > params["recovery_return_threshold"] and price_vs_sma < -0.1
        ):  # Still below SMA but recovering
            return RegimeType.RECOVERY

        # High/Low volatility detection
        if vol_val > params["high_volatility_threshold"]:
            return RegimeType.HIGH_VOLATILITY
        elif vol_val < params["low_volatility_threshold"]:
            return RegimeType.LOW_VOLATILITY

        # Trend-based detection
        if return_val > params["bull_return_threshold"]:
            return RegimeType.BULL
        elif return_val < params["bear_return_threshold"]:
            return RegimeType.BEAR
        else:
            return RegimeType.SIDEWAYS

    def _calculate_confidence(
        self,
        return_val: float,
        vol_val: float,
        regime: RegimeType,
        params: Dict[str, Any],
    ) -> float:
        """Calculate confidence level for regime classification"""

        # Base confidence
        confidence = 0.5

        # Adjust based on how extreme the values are relative to thresholds
        if regime == RegimeType.BULL:
            excess = return_val - params["bull_return_threshold"]
            confidence = min(0.95, 0.6 + excess * 2)
        elif regime == RegimeType.BEAR:
            excess = abs(return_val - params["bear_return_threshold"])
            confidence = min(0.95, 0.6 + excess * 2)
        elif regime == RegimeType.CRISIS:
            vol_excess = vol_val - params["crisis_volatility_threshold"]
            ret_excess = abs(return_val - params["crisis_return_threshold"])
            confidence = min(0.95, 0.7 + (vol_excess + ret_excess))
        elif regime == RegimeType.HIGH_VOLATILITY:
            excess = vol_val - params["high_volatility_threshold"]
            confidence = min(0.90, 0.6 + excess * 5)
        elif regime == RegimeType.LOW_VOLATILITY:
            excess = params["low_volatility_threshold"] - vol_val
            confidence = min(0.90, 0.6 + excess * 10)

        return max(0.1, confidence)  # Minimum confidence

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for rule-based regime detection"""
        return {
            "return_window": 60,  # Days for rolling return calculation
            "volatility_window": 20,  # Days for rolling volatility
            "correlation_window": 20,  # Days for correlation calculation
            "bull_return_threshold": 0.10,  # 10% positive return over window
            "bear_return_threshold": -0.10,  # 10% negative return over window
            "high_volatility_threshold": 0.25,  # 25% annualized volatility
            "low_volatility_threshold": 0.10,  # 10% annualized volatility
            "crisis_return_threshold": -0.20,  # 20% drawdown
            "crisis_volatility_threshold": 0.35,  # 35% volatility
            "recovery_return_threshold": 0.05,  # 5% recovery return
        }

    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available parameters"""
        return {
            "return_window": {
                "type": "int",
                "default": 60,
                "min": 20,
                "max": 252,
                "description": "Number of days for rolling return calculation",
            },
            "volatility_window": {
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 100,
                "description": "Number of days for rolling volatility calculation",
            },
            "bull_return_threshold": {
                "type": "float",
                "default": 0.10,
                "min": 0.01,
                "max": 0.50,
                "description": "Minimum return threshold for bull market classification",
            },
            "bear_return_threshold": {
                "type": "float",
                "default": -0.10,
                "min": -0.50,
                "max": -0.01,
                "description": "Maximum return threshold for bear market classification",
            },
            "high_volatility_threshold": {
                "type": "float",
                "default": 0.25,
                "min": 0.15,
                "max": 0.60,
                "description": "Minimum volatility threshold for high volatility regime",
            },
            "low_volatility_threshold": {
                "type": "float",
                "default": 0.10,
                "min": 0.05,
                "max": 0.20,
                "description": "Maximum volatility threshold for low volatility regime",
            },
            "crisis_return_threshold": {
                "type": "float",
                "default": -0.20,
                "min": -0.50,
                "max": -0.10,
                "description": "Return threshold for crisis regime detection",
            },
            "crisis_volatility_threshold": {
                "type": "float",
                "default": 0.35,
                "min": 0.25,
                "max": 0.80,
                "description": "Volatility threshold for crisis regime detection",
            },
            "recovery_return_threshold": {
                "type": "float",
                "default": 0.05,
                "min": 0.01,
                "max": 0.20,
                "description": "Return threshold for recovery regime detection",
            },
        }
