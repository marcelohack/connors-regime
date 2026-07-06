"""
Tests for the composite market regime detector and regime service.

The synthetic-data tests double as regressions for the bugs that made
the previous rule-based detector untrustworthy:
- volatility must be annualized before classification
- a persistently volatile asset must never classify as low volatility
- regimes must not flicker day-to-day (hysteresis)
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from connors_regime import (
    CompositeRegimeDetector,
    RegimeDetectionRequest,
    RegimeMethod,
    RegimeService,
    RegimeType,
    TrendState,
    VolatilityState,
)


def make_ohlcv(closes: np.ndarray, start: str = "2020-01-01") -> pd.DataFrame:
    """Build an OHLCV DataFrame from a series of closes"""
    dates = pd.date_range(start=start, periods=len(closes), freq="B")
    df = pd.DataFrame({"Close": closes}, index=dates)
    df["Open"] = df["Close"].shift(1).fillna(closes[0])
    df["High"] = df[["Open", "Close"]].max(axis=1) * 1.005
    df["Low"] = df[["Open", "Close"]].min(axis=1) * 0.995
    df["Volume"] = 1_000_000
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.ticker = "TEST"
    return df


def closes_from_returns(returns: np.ndarray, base: float = 100.0) -> np.ndarray:
    """Build a close price series from daily log returns"""
    return base * np.exp(np.cumsum(returns))


@pytest.fixture
def detector():
    return CompositeRegimeDetector()


@pytest.fixture
def quiet_bull_data():
    """~2 years of steady uptrend with modest volatility"""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0008, 0.008, 500)
    return make_ohlcv(closes_from_returns(returns))


@pytest.fixture
def high_vol_data():
    """~2 years of a persistently volatile asset (3% daily moves)"""
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.03, 500)
    return make_ohlcv(closes_from_returns(returns))


@pytest.fixture
def crash_data():
    """Calm uptrend, then a sharp high-volatility crash, then a rebound"""
    rng = np.random.default_rng(3)
    calm = rng.normal(0.0006, 0.007, 300)
    crash = rng.normal(-0.02, 0.035, 60)
    rebound = rng.normal(0.006, 0.012, 120)
    returns = np.concatenate([calm, crash, rebound])
    return make_ohlcv(closes_from_returns(returns))


class TestCompositeRegimeDetector:
    def test_detector_initialization(self, detector):
        assert detector.method == RegimeMethod.COMPOSITE

    def test_default_parameters(self, detector):
        params = detector.get_default_parameters()
        info = detector.get_parameter_info()

        # Every parameter must be documented and every documented
        # parameter must have a default
        assert set(params.keys()) == set(info.keys())
        assert params["confirm_days"] >= 1
        assert 0 < params["vol_low_pct"] < params["vol_high_pct"]
        assert params["vol_high_pct"] <= params["vol_extreme_pct"]

    def test_data_validation_missing_columns(self, detector):
        df = pd.DataFrame({"Close": range(100)})
        with pytest.raises(ValueError, match="Missing required columns"):
            detector._validate_data(df)

    def test_data_validation_insufficient_data(self, detector, quiet_bull_data):
        with pytest.raises(ValueError, match="Insufficient data points"):
            detector._validate_data(quiet_bull_data.iloc[:10])

    def test_volatility_is_annualized(self, detector, high_vol_data):
        """Regression: raw daily std must be scaled by sqrt(252).

        The original rule-based detector compared daily volatility
        (~0.03) against annualized thresholds (0.25), classifying
        everything as low volatility.
        """
        result = detector.detect(high_vol_data)
        vol = result.data["volatility_20d"].dropna()

        expected = 0.03 * np.sqrt(252)  # ~0.48 annualized
        assert vol.median() == pytest.approx(expected, rel=0.2)

    def test_high_vol_asset_not_labeled_low_volatility(self, detector, high_vol_data):
        """Regression: TSLA-style asset was 100% low_volatility before.

        With percentile-based volatility a persistently volatile asset
        should distribute across states rather than pin to LOW.
        """
        result = detector.detect(high_vol_data)
        assert result.success
        assert len(result.detections) > 50

        low_vol_share = sum(
            1 for d in result.detections if d.regime == RegimeType.LOW_VOLATILITY
        ) / len(result.detections)
        assert low_vol_share < 0.9

    def test_quiet_bull_detected(self, detector, quiet_bull_data):
        result = detector.detect(quiet_bull_data)
        assert result.success

        # Once the trend SMA is established the uptrend should dominate
        late = [d for d in result.detections if d.date >= result.data.index[300]]
        bull_share = sum(1 for d in late if d.regime == RegimeType.BULL) / len(late)
        assert bull_share > 0.6

        # A quiet uptrend must never look like a crisis
        assert all(d.regime != RegimeType.CRISIS for d in result.detections)

    def test_crash_detected_as_crisis(self, detector, crash_data):
        result = detector.detect(crash_data)
        assert result.success

        crash_window = result.data.index[310:380]
        in_crash = [d for d in result.detections if d.date in crash_window]
        crisis_share = sum(1 for d in in_crash if d.regime == RegimeType.CRISIS) / max(
            len(in_crash), 1
        )
        assert crisis_share > 0.3

    def test_recovery_after_crash(self, detector, crash_data):
        result = detector.detect(crash_data)
        rebound_window = result.data.index[380:]
        in_rebound = {d.regime for d in result.detections if d.date in rebound_window}
        # The rebound should register as recovery and/or a return to bull
        assert in_rebound & {RegimeType.RECOVERY, RegimeType.BULL}

    def test_regimes_do_not_flicker(self, detector, crash_data):
        """Committed regimes must persist; raw day-to-day noise is
        absorbed by the hysteresis filter."""
        result = detector.detect(crash_data)

        regimes = [d.regime for d in result.detections]
        durations = []
        run = 1
        for prev, cur in zip(regimes, regimes[1:]):
            if cur == prev:
                run += 1
            else:
                durations.append(run)
                run = 1
        durations.append(run)

        assert np.mean(durations) >= 5

    def test_transitions_match_detections(self, detector, crash_data):
        result = detector.detect(crash_data)
        regime_changes = sum(
            1
            for prev, cur in zip(result.detections, result.detections[1:])
            if prev.regime != cur.regime
        )
        assert len(result.regime_transitions) == regime_changes

    def test_detection_metadata_has_both_axes(self, detector, quiet_bull_data):
        result = detector.detect(quiet_bull_data)
        meta = result.detections[-1].metadata

        assert meta["trend"] in {t.value for t in TrendState}
        assert meta["volatility_state"] in {v.value for v in VolatilityState}
        assert "vol_percentile" in meta
        assert "drawdown" in meta

    def test_confidence_bounds(self, detector, crash_data):
        result = detector.detect(crash_data)
        assert all(0.1 <= d.confidence <= 0.95 for d in result.detections)

    def test_custom_parameters(self, detector, quiet_bull_data):
        result = detector.detect(quiet_bull_data, volatility_window=10, confirm_days=3)
        assert result.success
        assert result.parameters["volatility_window"] == 10
        assert "volatility_10d" in result.data.columns

    def test_detect_with_invalid_data(self, detector):
        df = pd.DataFrame({"Close": range(5)})
        result = detector.detect(df)
        assert result.success is False
        assert result.error is not None


class TestRegimeService:
    @pytest.fixture
    def service(self, tmp_path):
        service = RegimeService()
        service.regime_base_dir = tmp_path / "regime_detections"
        service.regime_base_dir.mkdir(parents=True, exist_ok=True)
        return service

    @pytest.fixture
    def mock_data(self, quiet_bull_data):
        data = quiet_bull_data.copy()
        data.ticker = "AAPL"
        return data

    def test_service_initialization(self, service):
        assert RegimeMethod.COMPOSITE in service.detectors
        assert isinstance(
            service.detectors[RegimeMethod.COMPOSITE], CompositeRegimeDetector
        )

    def test_get_available_methods(self, service):
        methods = service.get_available_methods()
        assert methods == ["composite"]

    def test_get_method_info(self, service):
        info = service.get_method_info("composite")
        assert info["name"] == "composite"
        assert "default_parameters" in info
        assert "parameter_info" in info

        assert service.get_method_info("invalid_method") == {}

    @patch("connors_regime.services.regime_service.RegimeService._download_data")
    def test_detect_regime_success(self, mock_download, service, mock_data):
        mock_download.return_value = mock_data

        request = RegimeDetectionRequest(
            ticker="AAPL", method="composite", start="2020-01-01", end="2021-12-31"
        )
        result = service.detect_regime(request)

        assert result.success is True
        assert result.ticker == "AAPL"
        assert result.method == RegimeMethod.COMPOSITE
        assert len(result.results.detections) > 0

    @patch("connors_regime.services.regime_service.RegimeService._download_data")
    def test_detect_regime_unknown_method(self, mock_download, service, mock_data):
        mock_download.return_value = mock_data

        request = RegimeDetectionRequest(ticker="AAPL", method="rule_based")
        result = service.detect_regime(request)

        assert result.success is False
        assert "Detector not found" in result.error

    @patch("connors_regime.services.regime_service.RegimeService._download_data")
    def test_detect_regime_with_save(self, mock_download, service, mock_data):
        mock_download.return_value = mock_data

        request = RegimeDetectionRequest(
            ticker="AAPL", method="composite", save_results=True
        )
        result = service.detect_regime(request)

        assert result.success is True
        assert result.results_path is not None
        assert Path(result.results_path).exists()

        with open(result.results_path) as f:
            saved = json.load(f)

        assert saved["ticker"] == "AAPL"
        assert saved["method"] == "composite"
        assert saved["current_regime"] in {r.value for r in RegimeType}
        assert len(saved["detections"]) > 0

    def test_load_dataset_file_csv(self, service, tmp_path, mock_data):
        csv_path = tmp_path / "data.csv"
        mock_data.to_csv(csv_path)

        loaded = service._load_dataset_file(str(csv_path), "AAPL")
        assert list(loaded.columns[:5]) == ["Open", "High", "Low", "Close", "Volume"]
        assert loaded.ticker == "AAPL"

    def test_load_dataset_file_missing(self, service):
        with pytest.raises(FileNotFoundError):
            service._load_dataset_file("/nonexistent/file.csv", "AAPL")

    def test_str2bool(self, service):
        assert service.str2bool(True) is True
        assert service.str2bool("yes") is True
        assert service.str2bool("false") is False
        with pytest.raises(ValueError):
            service.str2bool("maybe")
