"""
Test Market Regime Detector CLI

Tests the CLI interface for regime detection, including:
- Method selection (composite, external methods)
- Parameter overrides
- Visualization generation
- File operations
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from connors_regime.core.market_regime import RegimeType, RegimeDetection
from connors_regime.services.regime_service import RegimeResult, RegimeServiceResult


class TestRegimeDetectorCLI:
    """Test Regime Detector CLI functionality"""

    def _create_test_data(self) -> pd.DataFrame:
        """Create synthetic OHLCV data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'Open': [100 + i for i in range(100)],
            'High': [105 + i for i in range(100)],
            'Low': [95 + i for i in range(100)],
            'Close': [102 + i for i in range(100)],
            'Volume': [1000000] * 100
        }, index=dates)

    def test_import_cli_module(self) -> None:
        """Test that the CLI module can be imported"""
        from connors_regime import cli

        assert hasattr(cli, "main")
        assert callable(cli.main)

    @patch("connors_regime.cli.RegimeService")
    @patch("connors_regime.cli.argparse.ArgumentParser.parse_args")
    def test_cli_composite_method(
        self, mock_parse_args: Mock, mock_service_class: Mock
    ) -> None:
        """Test CLI with composite detection method"""
        mock_args = Mock()
        mock_args.ticker = "AAPL"
        mock_args.method = "composite"
        mock_args.external_method = None
        mock_args.datasource = "yfinance"
        mock_args.dataset_file = None
        mock_args.start = "2024-01-01"
        mock_args.end = "2024-12-31"
        mock_args.timespan = None
        mock_args.interval = "1d"
        mock_args.market = "america"
        mock_args.method_params = None
        mock_args.show_method_params = False
        mock_args.save_results = False
        mock_args.plot = False
        mock_args.save_plot = False
        mock_args.list_methods = False
        mock_args.list_datasources = False
        mock_args.list_markets = False
        mock_args.list_saved = False
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args

        # Setup mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.get_available_methods.return_value = ["composite"]
        mock_service.get_datasources.return_value = ["yfinance"]
        mock_service.get_market_configs.return_value = ["america"]
        mock_service.get_available_timeframes.return_value = ["1M", "3M", "6M"]
        mock_service.get_market_config_info.return_value = {
            "name": "United States",
            "yf_ticker_suffix": ""
        }
        mock_service.get_default_dates.return_value = {
            "start": "2024-01-01",
            "end": "2024-12-31"
        }

        # Mock detection result
        test_data = self._create_test_data()
        test_detections = [
            RegimeDetection(
                date=test_data.index[0],
                regime=RegimeType.BULL,
                confidence=0.85,
                method="composite",
                metadata={"return_value": 0.15, "volatility_value": 0.20}
            )
        ]
        regime_result = RegimeResult(
            ticker="AAPL",
            current_regime=RegimeType.BULL,
            detections=test_detections,
            regime_transitions=[],
            data=test_data,
            method="composite",
            parameters={},
            calculation_time=1.5
        )
        service_result = RegimeServiceResult(
            ticker="AAPL",
            method="composite",
            success=True,
            results=regime_result,
            results_path="/tmp/results.json",
            plot_path="/tmp/plot.html"
        )
        mock_service.detect_regime.return_value = service_result

        from connors_regime.cli import main

        with patch("builtins.print"):
            main()

        # Verify service was called
        mock_service.detect_regime.assert_called_once()

    @patch("connors_regime.cli.RegimeService")
    @patch("connors_regime.cli.argparse.ArgumentParser.parse_args")
    def test_cli_parameter_overrides(
        self, mock_parse_args: Mock, mock_service_class: Mock
    ) -> None:
        """Test parameter override from command line"""
        mock_args = Mock()
        mock_args.ticker = "AAPL"
        mock_args.method = "composite"
        mock_args.external_method = None
        mock_args.datasource = "yfinance"
        mock_args.dataset_file = None
        mock_args.start = None
        mock_args.end = None
        mock_args.timespan = None
        mock_args.interval = "1d"
        mock_args.market = "america"
        mock_args.method_params = "sma_period:50;rsi_period:7"
        mock_args.show_method_params = False
        mock_args.save_results = False
        mock_args.plot = False
        mock_args.save_plot = False
        mock_args.list_methods = False
        mock_args.list_datasources = False
        mock_args.list_markets = False
        mock_args.list_saved = False
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args

        # Setup mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.get_available_methods.return_value = ["composite"]
        mock_service.get_datasources.return_value = ["yfinance"]
        mock_service.get_market_configs.return_value = ["america"]
        mock_service.get_available_timeframes.return_value = ["1M"]
        mock_service.get_market_config_info.return_value = {"name": "US", "yf_ticker_suffix": ""}
        mock_service.get_default_dates.return_value = {"start": "2024-01-01", "end": "2024-12-31"}

        # Mock result
        test_data = self._create_test_data()
        regime_result = RegimeResult(
            ticker="AAPL",
            current_regime=RegimeType.BULL,
            detections=[RegimeDetection(
                date=test_data.index[0],
                regime=RegimeType.BULL,
                confidence=0.8,
                method="composite"
            )],
            regime_transitions=[],
            data=test_data,
            method="composite",
            parameters={"sma_period": 50, "rsi_period": 7},
            calculation_time=1.0
        )
        service_result = RegimeServiceResult(ticker="AAPL", method="composite", success=True, results=regime_result)
        mock_service.detect_regime.return_value = service_result

        from connors_regime.cli import main

        with patch("builtins.print"):
            main()

        # Verify parameters were passed
        call_args = mock_service.detect_regime.call_args
        request = call_args[0][0]
        assert request.parameters is not None
        assert request.parameters["sma_period"] == 50
        assert request.parameters["rsi_period"] == 7

    @patch("connors_regime.cli.RegimeService")
    @patch("connors_regime.cli.argparse.ArgumentParser.parse_args")
    def test_cli_list_methods(
        self, mock_parse_args: Mock, mock_service_class: Mock
    ) -> None:
        """Test --list-methods flag"""
        mock_args = Mock()
        mock_args.list_methods = True
        mock_args.list_datasources = False
        mock_args.list_markets = False
        mock_args.list_saved = False
        mock_parse_args.return_value = mock_args

        # Setup mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.get_available_methods.return_value = ["composite"]
        mock_service.get_datasources.return_value = ["yfinance"]
        mock_service.get_market_configs.return_value = ["america"]
        mock_service.get_available_timeframes.return_value = ["1M"]
        mock_service.get_all_methods_info.return_value = {
            "composite": {
                "description": "Composite regime detection",
                "default_parameters": {}
            }
        }

        from connors_regime.cli import main

        with patch("builtins.print") as mock_print:
            main()

        # Verify methods were listed
        mock_service.get_all_methods_info.assert_called_once()
        mock_print.assert_called()

    @patch("connors_regime.cli.RegimeService")
    @patch("connors_regime.cli.argparse.ArgumentParser.parse_args")
    def test_cli_list_datasources(
        self, mock_parse_args: Mock, mock_service_class: Mock
    ) -> None:
        """Test --list-datasources flag"""
        mock_args = Mock()
        mock_args.list_methods = False
        mock_args.list_datasources = True
        mock_args.list_markets = False
        mock_args.list_saved = False
        mock_parse_args.return_value = mock_args

        # Setup mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.get_available_methods.return_value = []
        mock_service.get_datasources.return_value = ["yfinance", "polygon"]
        mock_service.get_market_configs.return_value = []
        mock_service.get_available_timeframes.return_value = []

        from connors_regime.cli import main

        with patch("builtins.print") as mock_print:
            main()

        # Verify datasources were listed (called at least once, may be called during argparse setup too)
        assert mock_service.get_datasources.call_count >= 1
        mock_print.assert_called()

    @patch("connors_regime.cli.RegimeService")
    @patch("connors_regime.cli.argparse.ArgumentParser.parse_args")
    def test_cli_list_markets(
        self, mock_parse_args: Mock, mock_service_class: Mock
    ) -> None:
        """Test --list-markets flag"""
        mock_args = Mock()
        mock_args.list_methods = False
        mock_args.list_datasources = False
        mock_args.list_markets = True
        mock_args.list_saved = False
        mock_parse_args.return_value = mock_args

        # Setup mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.get_available_methods.return_value = []
        mock_service.get_datasources.return_value = []
        mock_service.get_market_configs.return_value = ["america", "australia"]
        mock_service.get_available_timeframes.return_value = []
        mock_service.get_market_config_info.side_effect = lambda x: {
            "america": {"name": "United States", "yf_ticker_suffix": ""},
            "australia": {"name": "Australia", "yf_ticker_suffix": ".AX"}
        }.get(x)

        from connors_regime.cli import main

        with patch("builtins.print") as mock_print:
            main()

        # Verify markets were listed (called at least once, may be called during argparse setup too)
        assert mock_service.get_market_configs.call_count >= 1
        mock_print.assert_called()

    @patch("connors_regime.cli.RegimeService")
    @patch("connors_regime.cli.argparse.ArgumentParser.parse_args")
    def test_cli_list_saved_results(
        self, mock_parse_args: Mock, mock_service_class: Mock
    ) -> None:
        """Test --list-saved flag"""
        mock_args = Mock()
        mock_args.list_methods = False
        mock_args.list_datasources = False
        mock_args.list_markets = False
        mock_args.list_saved = True
        mock_parse_args.return_value = mock_args

        # Setup mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.get_available_methods.return_value = []
        mock_service.get_datasources.return_value = []
        mock_service.get_market_configs.return_value = []
        mock_service.get_available_timeframes.return_value = []
        mock_service.list_saved_results.return_value = [
            {
                "ticker": "AAPL",
                "method": "composite",
                "market": "america",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "file_path": "/tmp/results.json",
                "modified": "2024-11-01"
            }
        ]

        from connors_regime.cli import main

        with patch("builtins.print") as mock_print:
            main()

        # Verify saved results were listed
        mock_service.list_saved_results.assert_called_once()
        mock_print.assert_called()

    @patch("connors_regime.cli.RegimeService")
    @patch("connors_regime.cli.argparse.ArgumentParser.parse_args")
    def test_cli_show_method_params(
        self, mock_parse_args: Mock, mock_service_class: Mock
    ) -> None:
        """Test --show-method-params flag"""
        mock_args = Mock()
        mock_args.ticker = None
        mock_args.method = "composite"
        mock_args.external_method = None
        mock_args.show_method_params = True
        mock_args.list_methods = False
        mock_args.list_datasources = False
        mock_args.list_markets = False
        mock_args.list_saved = False
        mock_parse_args.return_value = mock_args

        # Setup mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.get_available_methods.return_value = ["composite"]
        mock_service.get_datasources.return_value = []
        mock_service.get_market_configs.return_value = []
        mock_service.get_available_timeframes.return_value = []
        mock_service.get_method_info.return_value = {
            "description": "Composite regime detection",
            "parameter_info": {
                "sma_period": {
                    "type": "int",
                    "default": 200,
                    "min": 50,
                    "max": 300,
                    "description": "SMA period for trend detection"
                }
            },
            "default_parameters": {"sma_period": 200}
        }

        from connors_regime.cli import main

        with patch("builtins.print") as mock_print:
            main()

        # Verify method info was retrieved
        mock_service.get_method_info.assert_called_once_with("composite")
        mock_print.assert_called()

    @patch("connors_regime.cli.RegimeService")
    @patch("connors_regime.cli.argparse.ArgumentParser.parse_args")
    def test_cli_dataset_file_loading(
        self, mock_parse_args: Mock, mock_service_class: Mock
    ) -> None:
        """Test loading from dataset file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Open,High,Low,Close,Volume\n")
            f.write("2024-01-01,100,105,95,102,1000000\n")
            f.write("2024-01-02,102,107,97,104,1100000\n")
            test_file = f.name

        try:
            mock_args = Mock()
            mock_args.ticker = "AAPL"
            mock_args.method = "composite"
            mock_args.external_method = None
            mock_args.datasource = "yfinance"
            mock_args.dataset_file = test_file
            mock_args.start = None
            mock_args.end = None
            mock_args.timespan = None
            mock_args.interval = "1d"
            mock_args.market = "america"
            mock_args.method_params = None
            mock_args.show_method_params = False
            mock_args.save_results = False
            mock_args.plot = False
            mock_args.save_plot = False
            mock_args.list_methods = False
            mock_args.list_datasources = False
            mock_args.list_markets = False
            mock_args.list_saved = False
            mock_args.verbose = False
            mock_parse_args.return_value = mock_args

            # Setup mock service
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_available_methods.return_value = ["composite"]
            mock_service.get_datasources.return_value = ["yfinance"]
            mock_service.get_market_configs.return_value = ["america"]
            mock_service.get_available_timeframes.return_value = ["1M"]
            mock_service.get_market_config_info.return_value = {"name": "US", "yf_ticker_suffix": ""}

            # Mock result
            test_data = pd.DataFrame()
            regime_result = RegimeResult(
                ticker="AAPL",
                current_regime=RegimeType.BULL,
                detections=[RegimeDetection(
                    date=pd.Timestamp('2024-01-01'),
                    regime=RegimeType.BULL,
                    confidence=0.8,
                    method="composite"
                )],
                regime_transitions=[],
                data=test_data,
                method="composite",
                parameters={},
                calculation_time=1.0
            )
            service_result = RegimeServiceResult(ticker="AAPL", method="composite", success=True, results=regime_result)
            mock_service.detect_regime.return_value = service_result

            from connors_regime.cli import main

            with patch("builtins.print"):
                main()

            # Verify dataset file was used
            call_args = mock_service.detect_regime.call_args
            request = call_args[0][0]
            assert request.dataset_file == test_file
            assert request.datasource == "file"

        finally:
            Path(test_file).unlink()

    @patch("connors_regime.cli.RegimeService")
    @patch("connors_regime.cli.argparse.ArgumentParser.parse_args")
    def test_cli_external_method_loading(
        self, mock_parse_args: Mock, mock_service_class: Mock
    ) -> None:
        """Test external method loading"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# Custom regime method\n")
            method_file = f.name

        try:
            mock_args = Mock()
            mock_args.ticker = "AAPL"
            mock_args.method = None
            mock_args.external_method = method_file
            mock_args.datasource = "yfinance"
            mock_args.dataset_file = None
            mock_args.start = None
            mock_args.end = None
            mock_args.timespan = None
            mock_args.interval = "1d"
            mock_args.market = "america"
            mock_args.method_params = None
            mock_args.show_method_params = False
            mock_args.save_results = False
            mock_args.plot = False
            mock_args.save_plot = False
            mock_args.list_methods = False
            mock_args.list_datasources = False
            mock_args.list_markets = False
            mock_args.list_saved = False
            mock_args.verbose = False
            mock_parse_args.return_value = mock_args

            # Setup mock service
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_available_methods.return_value = []
            mock_service.get_datasources.return_value = ["yfinance"]
            mock_service.get_market_configs.return_value = ["america"]
            mock_service.get_available_timeframes.return_value = ["1M"]
            mock_service.get_market_config_info.return_value = {"name": "US", "yf_ticker_suffix": ""}
            mock_service.get_default_dates.return_value = {"start": "2024-01-01", "end": "2024-12-31"}
            mock_service.load_external_method.return_value = "custom_method"

            # Mock result
            test_data = self._create_test_data()
            regime_result = RegimeResult(
                ticker="AAPL",
                current_regime=RegimeType.BULL,
                detections=[RegimeDetection(
                    date=test_data.index[0],
                    regime=RegimeType.BULL,
                    confidence=0.8,
                    method="custom_method"
                )],
                regime_transitions=[],
                data=test_data,
                method="custom_method",
                parameters={},
                calculation_time=1.0
            )
            service_result = RegimeServiceResult(ticker="AAPL", method="custom_method", success=True, results=regime_result)
            mock_service.detect_regime.return_value = service_result

            from connors_regime.cli import main

            with patch("builtins.print"):
                main()

            # Verify external method was loaded
            mock_service.load_external_method.assert_called_once_with(method_file)

        finally:
            Path(method_file).unlink()

    @patch("connors_regime.cli.RegimeService")
    @patch("connors_regime.cli.argparse.ArgumentParser.parse_args")
    def test_cli_detection_failure(
        self, mock_parse_args: Mock, mock_service_class: Mock
    ) -> None:
        """Test handling of detection failure"""
        mock_args = Mock()
        mock_args.ticker = "INVALID"
        mock_args.method = "composite"
        mock_args.external_method = None
        mock_args.datasource = "yfinance"
        mock_args.dataset_file = None
        mock_args.start = None
        mock_args.end = None
        mock_args.timespan = None
        mock_args.interval = "1d"
        mock_args.market = "america"
        mock_args.method_params = None
        mock_args.show_method_params = False
        mock_args.save_results = False
        mock_args.plot = False
        mock_args.save_plot = False
        mock_args.list_methods = False
        mock_args.list_datasources = False
        mock_args.list_markets = False
        mock_args.list_saved = False
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args

        # Setup mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.get_available_methods.return_value = ["composite"]
        mock_service.get_datasources.return_value = ["yfinance"]
        mock_service.get_market_configs.return_value = ["america"]
        mock_service.get_available_timeframes.return_value = ["1M"]
        mock_service.get_market_config_info.return_value = {"name": "US", "yf_ticker_suffix": ""}
        mock_service.get_default_dates.return_value = {"start": "2024-01-01", "end": "2024-12-31"}

        # Mock failure result
        service_result = RegimeServiceResult(
            ticker="INVALID",
            method="composite",
            results=None,
            success=False,
            error="Failed to fetch data for INVALID"
        )
        mock_service.detect_regime.return_value = service_result

        from connors_regime.cli import main

        with patch("builtins.print") as mock_print:
            main()

        # Verify error was printed
        mock_print.assert_called()
        error_calls = [call for call in mock_print.call_args_list if "Failed to fetch data" in str(call)]
        assert len(error_calls) > 0

    @patch("connors_regime.cli.RegimeService")
    @patch("connors_regime.cli.argparse.ArgumentParser.parse_args")
    def test_cli_missing_ticker(
        self, mock_parse_args: Mock, mock_service_class: Mock
    ) -> None:
        """Test error handling for missing ticker"""
        mock_args = Mock()
        mock_args.ticker = None
        mock_args.method = "composite"
        mock_args.external_method = None
        mock_args.list_methods = False
        mock_args.list_datasources = False
        mock_args.list_markets = False
        mock_args.list_saved = False
        mock_args.show_method_params = False
        mock_parse_args.return_value = mock_args

        # Setup mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.get_available_methods.return_value = ["composite"]
        mock_service.get_datasources.return_value = []
        mock_service.get_market_configs.return_value = []
        mock_service.get_available_timeframes.return_value = []

        from connors_regime.cli import main

        with patch("builtins.print") as mock_print:
            main()

        # Verify error was printed
        error_calls = [call for call in mock_print.call_args_list if "ticker is required" in str(call)]
        assert len(error_calls) > 0


class TestParseMethodParams:
    """Test CLI method parameter string parsing"""

    def test_basic_parsing(self) -> None:
        from connors_regime.cli import _parse_method_params

        result = _parse_method_params("confirm_days:10;volatility_window:30")
        assert result == {"confirm_days": 10, "volatility_window": 30}

    def test_float_parsing(self) -> None:
        from connors_regime.cli import _parse_method_params

        result = _parse_method_params("trend_threshold:0.02;vol_high_pct:0.8")
        assert result == {"trend_threshold": 0.02, "vol_high_pct": 0.8}

    def test_list_parsing(self) -> None:
        from connors_regime.cli import _parse_method_params

        result = _parse_method_params("thresholds:[0.1,0.2,0.3]")
        assert result == {"thresholds": [0.1, 0.2, 0.3]}

        result = _parse_method_params("values:[1,2,3,4]")
        assert result == {"values": [1, 2, 3, 4]}

        result = _parse_method_params("mixed:[1.5,2,text]")
        assert result == {"mixed": [1.5, 2, "text"]}

    def test_boolean_parsing(self) -> None:
        from connors_regime.cli import _parse_method_params

        assert _parse_method_params("use_feature:true") == {"use_feature": True}
        assert _parse_method_params("flag:false") == {"flag": False}

    def test_string_parsing(self) -> None:
        from connors_regime.cli import _parse_method_params

        result = _parse_method_params("name:test_value")
        assert result == {"name": "test_value"}

    def test_edge_cases(self) -> None:
        from connors_regime.cli import _parse_method_params

        assert _parse_method_params("") == {}
        assert _parse_method_params("empty:") == {"empty": ""}
        assert _parse_method_params("single:42") == {"single": 42}
        assert _parse_method_params("spaced: 42 ; another : test ") == {
            "spaced": 42,
            "another": "test",
        }
