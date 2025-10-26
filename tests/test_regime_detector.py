"""
Unit tests for Market Regime Detector

Tests for the core regime detection functionality, service layer,
and CLI interface.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from connors_regime.core.market_regime import (
    RegimeMethod,
    RegimeResult,
    RegimeType,
    RuleBasedRegimeDetector,
)
from connors_regime.services.regime_service import RegimeDetectionRequest, RegimeService


class TestRuleBasedRegimeDetector:
    """Test the RuleBasedRegimeDetector class"""

    @pytest.fixture
    def detector(self):
        """Create a RuleBasedRegimeDetector instance"""
        return RuleBasedRegimeDetector()

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)

        # Generate realistic OHLCV data
        base_price = 100.0
        prices = []
        volumes = []

        for i in range(len(dates)):
            # Add some trend and noise
            trend_factor = 1 + (i / 1000)  # Slight upward trend
            noise = np.random.normal(0, 0.02)  # 2% daily volatility
            price = base_price * trend_factor * (1 + noise)
            prices.append(price)

            # Volume with some randomness
            volume = np.random.randint(1000000, 5000000)
            volumes.append(volume)

        # Create OHLC from close prices
        df = pd.DataFrame({"Close": prices, "Volume": volumes}, index=dates)

        # Generate Open, High, Low from Close
        df["Open"] = df["Close"].shift(1).fillna(df["Close"].iloc[0])
        df["High"] = df[["Open", "Close"]].max(axis=1) * (
            1 + np.random.uniform(0, 0.01, len(df))
        )
        df["Low"] = df[["Open", "Close"]].min(axis=1) * (
            1 - np.random.uniform(0, 0.01, len(df))
        )

        # Reorder columns
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.ticker = "TEST"

        return df

    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector.method == RegimeMethod.RULE_BASED

    def test_get_default_parameters(self, detector):
        """Test default parameters retrieval"""
        params = detector.get_default_parameters()

        expected_params = {
            "return_window",
            "volatility_window",
            "correlation_window",
            "bull_return_threshold",
            "bear_return_threshold",
            "high_volatility_threshold",
            "low_volatility_threshold",
            "crisis_return_threshold",
            "crisis_volatility_threshold",
            "recovery_return_threshold",
        }

        assert set(params.keys()) == expected_params
        assert params["return_window"] == 60
        assert params["volatility_window"] == 20
        assert params["bull_return_threshold"] == 0.10
        assert params["bear_return_threshold"] == -0.10

    def test_get_parameter_info(self, detector):
        """Test parameter information retrieval"""
        param_info = detector.get_parameter_info()

        assert "return_window" in param_info
        assert "volatility_window" in param_info
        assert "bull_return_threshold" in param_info

        # Check parameter structure
        return_window_info = param_info["return_window"]
        assert return_window_info["type"] == "int"
        assert return_window_info["default"] == 60
        assert "description" in return_window_info

    def test_data_validation_success(self, detector, sample_data):
        """Test successful data validation"""
        # Should not raise an exception
        detector._validate_data(sample_data)

    def test_data_validation_missing_columns(self, detector):
        """Test data validation with missing columns"""
        invalid_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                # Missing Low, Close, Volume
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            detector._validate_data(invalid_data)

    def test_data_validation_insufficient_data(self, detector):
        """Test data validation with insufficient data points"""
        insufficient_data = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [105, 106],
                "Low": [95, 96],
                "Close": [102, 104],
                "Volume": [1000, 1100],
            }
        )

        with pytest.raises(ValueError, match="Insufficient data points"):
            detector._validate_data(insufficient_data)

    def test_feature_calculation(self, detector, sample_data):
        """Test feature calculation from OHLCV data"""
        features_df = detector._calculate_features(sample_data)

        # Check that new columns are added
        expected_new_cols = {
            "log_returns",
            "return_60d",
            "volatility_20d",
            "sma_200",
            "price_vs_sma",
            "volume_sma",
            "volume_ratio",
            "rsi",
        }

        new_cols = set(features_df.columns) - set(sample_data.columns)
        assert expected_new_cols.issubset(new_cols)

        # Check that features have reasonable values
        assert not features_df["log_returns"].isna().all()
        assert features_df["volatility_20d"].min() >= 0  # Volatility should be positive
        assert 0 <= features_df["rsi"].max() <= 100  # RSI should be 0-100

    def test_regime_classification(self, detector):
        """Test regime classification logic"""
        params = detector.get_default_parameters()

        # Test bull market classification
        bull_regime = detector._classify_regime(
            return_val=0.15,  # Above bull threshold
            vol_val=0.15,  # Normal volatility
            price_vs_sma=0.05,  # Above SMA
            params=params,
        )
        assert bull_regime == RegimeType.BULL

        # Test bear market classification
        bear_regime = detector._classify_regime(
            return_val=-0.15,  # Below bear threshold
            vol_val=0.15,  # Normal volatility
            price_vs_sma=-0.05,  # Below SMA
            params=params,
        )
        assert bear_regime == RegimeType.BEAR

        # Test crisis classification
        crisis_regime = detector._classify_regime(
            return_val=-0.25,  # Below crisis threshold
            vol_val=0.40,  # Above crisis volatility threshold
            price_vs_sma=-0.15,  # Well below SMA
            params=params,
        )
        assert crisis_regime == RegimeType.CRISIS

        # Test high volatility classification
        high_vol_regime = detector._classify_regime(
            return_val=0.05,  # Neutral return
            vol_val=0.30,  # High volatility
            price_vs_sma=0.02,  # Slightly above SMA
            params=params,
        )
        assert high_vol_regime == RegimeType.HIGH_VOLATILITY

        # Test sideways classification
        sideways_regime = detector._classify_regime(
            return_val=0.02,  # Low return
            vol_val=0.15,  # Normal volatility
            price_vs_sma=0.01,  # Near SMA
            params=params,
        )
        assert sideways_regime == RegimeType.SIDEWAYS

    def test_confidence_calculation(self, detector):
        """Test confidence level calculation"""
        params = detector.get_default_parameters()

        # Test strong bull signal confidence
        strong_bull_conf = detector._calculate_confidence(
            return_val=0.20,  # Well above threshold
            vol_val=0.15,
            regime=RegimeType.BULL,
            params=params,
        )
        assert strong_bull_conf > 0.7  # Should be high confidence

        # Test weak signal confidence
        weak_conf = detector._calculate_confidence(
            return_val=0.11,  # Just above threshold
            vol_val=0.15,
            regime=RegimeType.BULL,
            params=params,
        )
        assert 0.5 < weak_conf < 0.8  # Should be moderate confidence

    def test_detect_success(self, detector, sample_data):
        """Test successful regime detection"""
        result = detector.detect(sample_data)

        assert isinstance(result, RegimeResult)
        assert result.success is True
        assert result.error is None
        assert result.method == RegimeMethod.RULE_BASED
        assert result.ticker == "TEST"
        assert len(result.detections) > 0
        assert result.current_regime is not None
        assert isinstance(result.current_regime, RegimeType)
        assert result.calculation_time > 0

        # Check that regime columns were added to data
        assert "regime" in result.data.columns
        assert "regime_confidence" in result.data.columns

    def test_detect_with_custom_parameters(self, detector, sample_data):
        """Test regime detection with custom parameters"""
        custom_params = {
            "bull_return_threshold": 0.05,  # Lower threshold
            "bear_return_threshold": -0.05,
            "volatility_window": 10,  # Shorter window
        }

        result = detector.detect(sample_data, **custom_params)

        assert result.success is True
        assert result.parameters["bull_return_threshold"] == 0.05
        assert result.parameters["volatility_window"] == 10

    def test_detect_with_invalid_data(self, detector):
        """Test regime detection with invalid data"""
        invalid_data = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [105, 106],
                "Low": [95, 96],
                "Close": [102, 104],
                "Volume": [1000, 1100],
            }
        )

        result = detector.detect(invalid_data)

        assert result.success is False
        assert result.error is not None
        assert "Insufficient data points" in result.error


class TestRegimeService:
    """Test the RegimeService class"""

    @pytest.fixture
    def service(self):
        """Create a RegimeService instance"""
        return RegimeService()

    @pytest.fixture
    def sample_request(self):
        """Create a sample regime detection request"""
        return RegimeDetectionRequest(
            ticker="AAPL",
            method="rule_based",
            start="2023-01-01",
            end="2023-12-31",
        )

    def test_service_initialization(self, service):
        """Test service initialization"""
        assert service.regime_base_dir.exists()
        assert RegimeMethod.RULE_BASED in service.detectors
        assert isinstance(
            service.detectors[RegimeMethod.RULE_BASED], RuleBasedRegimeDetector
        )

    def test_get_available_methods(self, service):
        """Test getting available methods"""
        methods = service.get_available_methods()
        assert "rule_based" in methods
        assert len(methods) >= 1

    def test_get_method_info(self, service):
        """Test getting method information"""
        info = service.get_method_info("rule_based")

        assert info["name"] == "rule_based"
        assert "description" in info
        assert "default_parameters" in info
        assert "parameter_info" in info

        # Test invalid method
        invalid_info = service.get_method_info("invalid_method")
        assert invalid_info == {}

    def test_get_all_methods_info(self, service):
        """Test getting all methods information"""
        all_info = service.get_all_methods_info()
        assert "rule_based" in all_info

    @patch("connors_regime.services.regime_service.RegimeService._download_data")
    def test_detect_regime_success(self, mock_download, service, sample_request):
        """Test successful regime detection"""
        # Mock the download data method
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        mock_data = pd.DataFrame(
            {
                "Open": np.random.randint(95, 105, 100),
                "High": np.random.randint(100, 110, 100),
                "Low": np.random.randint(90, 100, 100),
                "Close": np.random.randint(95, 105, 100),
                "Volume": np.random.randint(1000000, 5000000, 100),
            },
            index=dates,
        )
        mock_data.ticker = "AAPL"
        mock_download.return_value = mock_data

        result = service.detect_regime(sample_request)

        assert result.success is True
        assert result.ticker == "AAPL"
        assert result.method == RegimeMethod.RULE_BASED
        assert result.results is not None

    @patch("connors_regime.services.regime_service.RegimeService._download_data")
    def test_detect_regime_with_save(self, mock_download, service, tmp_path):
        """Test regime detection with result saving"""
        # Setup temp directory
        service.regime_base_dir = tmp_path / "regimes"
        service.regime_base_dir.mkdir(exist_ok=True)

        # Mock data
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        mock_data = pd.DataFrame(
            {
                "Open": np.random.randint(95, 105, 50),
                "High": np.random.randint(100, 110, 50),
                "Low": np.random.randint(90, 100, 50),
                "Close": np.random.randint(95, 105, 50),
                "Volume": np.random.randint(1000000, 5000000, 50),
            },
            index=dates,
        )
        mock_data.ticker = "AAPL"
        mock_download.return_value = mock_data

        request = RegimeDetectionRequest(
            ticker="AAPL", method="rule_based", save_results=True
        )

        result = service.detect_regime(request)

        assert result.success is True
        assert result.results_path is not None
        assert Path(result.results_path).exists()

        # Check saved file content
        with open(result.results_path, "r") as f:
            saved_data = json.load(f)

        assert saved_data["ticker"] == "AAPL"
        assert saved_data["method"] == "rule_based"
        assert "detections" in saved_data
        assert "current_regime" in saved_data

    def test_load_dataset_file_csv(self, service, tmp_path):
        """Test loading dataset from CSV file"""
        # Create temporary CSV file
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        test_data = pd.DataFrame(
            {
                "Open": np.random.randint(95, 105, 50),
                "High": np.random.randint(100, 110, 50),
                "Low": np.random.randint(90, 100, 50),
                "Close": np.random.randint(95, 105, 50),
                "Volume": np.random.randint(1000000, 5000000, 50),
            },
            index=dates,
        )

        csv_file = tmp_path / "test_data.csv"
        test_data.to_csv(csv_file)

        loaded_data = service._load_dataset_file(str(csv_file), "TEST")

        assert loaded_data.ticker == "TEST"
        assert len(loaded_data) == 50
        assert all(
            col in loaded_data.columns
            for col in ["Open", "High", "Low", "Close", "Volume"]
        )

    def test_load_dataset_file_json(self, service, tmp_path):
        """Test loading dataset from JSON file"""
        # Create temporary JSON file
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        test_data = pd.DataFrame(
            {
                "Open": np.random.randint(95, 105, 50),
                "High": np.random.randint(100, 110, 50),
                "Low": np.random.randint(90, 100, 50),
                "Close": np.random.randint(95, 105, 50),
                "Volume": np.random.randint(1000000, 5000000, 50),
            },
            index=dates,
        )

        json_file = tmp_path / "test_data.json"
        test_data.to_json(json_file, orient="index")

        loaded_data = service._load_dataset_file(str(json_file), "TEST")

        assert loaded_data.ticker == "TEST"
        assert len(loaded_data) == 50
        assert all(
            col in loaded_data.columns
            for col in ["Open", "High", "Low", "Close", "Volume"]
        )

    def test_load_dataset_file_missing(self, service):
        """Test loading non-existent dataset file"""
        with pytest.raises(FileNotFoundError):
            service._load_dataset_file("non_existent_file.csv", "TEST")

    def test_load_dataset_file_invalid_format(self, service, tmp_path):
        """Test loading dataset with unsupported format"""
        invalid_file = tmp_path / "test_data.txt"
        invalid_file.write_text("invalid data")

        with pytest.raises(ValueError, match="Unsupported dataset file format"):
            service._load_dataset_file(str(invalid_file), "TEST")

    def test_list_saved_results_empty(self, service, tmp_path):
        """Test listing saved results when none exist"""
        service.regime_base_dir = tmp_path / "empty_regimes"
        service.regime_base_dir.mkdir(exist_ok=True)

        results = service.list_saved_results()
        assert results == []

    def test_list_saved_results_with_data(self, service, tmp_path):
        """Test listing saved results with existing data"""
        service.regime_base_dir = tmp_path / "regimes"

        # Create mock saved result structure
        method_dir = service.regime_base_dir / "rule_based" / "america"
        method_dir.mkdir(parents=True, exist_ok=True)

        # Create mock result file
        result_file = method_dir / "AAPL_2023-01-01_2023-12-31.json"
        mock_result = {
            "ticker": "AAPL",
            "method": "rule_based",
            "current_regime": "bull",
        }

        with open(result_file, "w") as f:
            json.dump(mock_result, f)

        results = service.list_saved_results()

        assert len(results) == 1
        assert results[0]["ticker"] == "AAPL"
        assert results[0]["method"] == "rule_based"
        assert results[0]["market"] == "america"

    def test_delete_saved_result(self, service, tmp_path):
        """Test deleting saved results"""
        service.regime_base_dir = tmp_path / "regimes"

        # Create mock saved result
        method_dir = service.regime_base_dir / "rule_based" / "america"
        method_dir.mkdir(parents=True, exist_ok=True)

        result_file = method_dir / "AAPL_2023-01-01_2023-12-31.json"
        result_file.write_text('{"test": "data"}')

        assert result_file.exists()

        success = service.delete_saved_result(str(result_file))

        assert success is True
        assert not result_file.exists()

    def test_str2bool(self, service):
        """Test string to boolean conversion"""
        assert service.str2bool(True) is True
        assert service.str2bool(False) is False
        assert service.str2bool("true") is True
        assert service.str2bool("false") is False
        assert service.str2bool("yes") is True
        assert service.str2bool("no") is False
        assert service.str2bool("1") is True
        assert service.str2bool("0") is False

        with pytest.raises(ValueError):
            service.str2bool("invalid")


class TestCLIIntegration:
    """Test CLI integration for regime detector"""

    def test_cli_import(self):
        """Test that CLI module can be imported"""
        from connors.cli.regime_detector import main

        assert callable(main)

    @patch("connors.cli.regime_detector.RegimeService")
    @patch("sys.argv", ["regime_detector", "--list-methods"])
    def test_cli_list_methods(self, mock_service_class):
        """Test CLI list methods functionality"""
        mock_service = Mock()

        # Mock all the methods called in the CLI
        mock_service.get_available_methods.return_value = ["rule_based"]
        mock_service.get_datasources.return_value = ["yfinance", "polygon"]
        mock_service.get_market_configs.return_value = ["america", "australia"]
        mock_service.get_available_timeframes.return_value = ["1D", "1W", "1M", "1Y"]

        mock_service.get_all_methods_info.return_value = {
            "rule_based": {
                "description": "Rule-based regime detection",
                "default_parameters": {"bull_return_threshold": 0.10},
            }
        }
        mock_service_class.return_value = mock_service

        from connors.cli.regime_detector import main

        # Should not raise exception
        try:
            main()
        except SystemExit:
            pass  # Expected for argparse when --list-methods is used


if __name__ == "__main__":
    pytest.main([__file__])
