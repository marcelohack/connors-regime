"""
Market Regime Service

Provides high-level interface for market regime detection operations,
integrating with data sources, detectors, and file storage.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import connors_datafetch.datasources.finnhub  # noqa: F401
import connors_datafetch.datasources.fmp  # noqa: F401
import connors_datafetch.datasources.polygon  # noqa: F401

# Import all datasources to ensure registration
import connors_datafetch.datasources.yfinance  # noqa: F401
import pandas as pd
import plotly.graph_objects as go
from connors_datafetch.config.manager import DataFetchConfigManager
from connors_datafetch.core.timespan import TimespanCalculator
from connors_datafetch.services.datafetch_service import DataFetchService
from plotly.subplots import make_subplots

from connors_regime.core.market_regime import (
    RegimeMethod,
    RegimeResult,
    RegimeType,
    RuleBasedRegimeDetector,
)
from connors_regime.core.registry import registry
from connors_regime.services.base import BaseService


@dataclass
class RegimeDetectionRequest:
    """Request for market regime detection"""

    ticker: str
    method: Union[RegimeMethod, str]
    parameters: Optional[Dict[str, Any]] = None
    datasource: str = "yfinance"
    dataset_file: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    interval: str = "1d"
    market_config: str = "america"
    timeframe: Optional[str] = None
    save_results: bool = False
    save_plot: bool = False
    show_plot: bool = False


@dataclass
class RegimeServiceResult:
    """Container for regime service results"""

    ticker: str
    method: RegimeMethod
    results: RegimeResult
    plot_path: Optional[str] = None
    results_path: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class RegimeService(BaseService):
    """Service for market regime detection"""

    def __init__(self) -> None:
        super().__init__()
        self.registry = registry
        self.config_manager = DataFetchConfigManager()
        self.download_service = DataFetchService()
        self.timespan_calculator = TimespanCalculator()

        # Initialize detectors
        self.detectors = {
            RegimeMethod.RULE_BASED: RuleBasedRegimeDetector(),
            # Future detectors to be added:
            # RegimeMethod.CLUSTERING: ClusteringRegimeDetector(),
            # RegimeMethod.HMM: HMMRegimeDetector(),
            # RegimeMethod.REGIME_SWITCHING: RegimeSwitchingDetector(),
        }

        # Ensure CONNORS_HOME directory structure
        self.connors_home = Path(
            os.environ.get("CONNORS_HOME", Path.home() / ".connors")
        )
        self.regime_base_dir = self.connors_home / "regime_detections"
        self._ensure_directory_exists(self.regime_base_dir)

    def get_available_methods(self) -> List[str]:
        """Get list of available regime detection methods"""
        return [method.value for method in RegimeMethod]

    def get_method_info(self, method: Union[RegimeMethod, str]) -> Dict[str, Any]:
        """Get information about a specific regime detection method"""
        try:
            if isinstance(method, str):
                method = RegimeMethod(method)
        except ValueError:
            # Invalid method name
            return {}

        detector = self.detectors.get(method)
        if not detector:
            return {}

        return {
            "name": method.value,
            "description": detector.__class__.__doc__
            or f"{method.value.title()} detector",
            "default_parameters": detector.get_default_parameters(),
            "parameter_info": detector.get_parameter_info(),
        }

    def get_all_methods_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available regime detection methods"""
        return {method.value: self.get_method_info(method) for method in RegimeMethod}

    def detect_regime(self, request: RegimeDetectionRequest) -> RegimeServiceResult:
        """
        Detect market regime for a given ticker and method

        Args:
            request: RegimeDetectionRequest containing all parameters

        Returns:
            RegimeServiceResult with detection results and file paths
        """
        try:
            # Handle method (enum for built-in, string for external)
            if isinstance(request.method, str):
                # Check if it's a built-in method first
                try:
                    method = RegimeMethod(request.method.lower())
                except ValueError:
                    # It's an external method
                    method = request.method
            else:
                method = request.method

            # Get or download data
            if request.dataset_file:
                data = self._load_dataset_file(request.dataset_file, request.ticker)
            else:
                data = self._download_data(request)

            # Get detector
            detector = self.detectors.get(method)
            if not detector:
                raise ValueError(f"Detector not found for method: {method}")

            # Prepare parameters
            detect_params = detector.get_default_parameters()
            if request.parameters:
                detect_params.update(request.parameters)

            # Detect regime
            regime_result = detector.detect(data, **detect_params)

            # Save results if requested
            results_path = None
            if request.save_results:
                results_path = self._save_results(regime_result, request)

            # Generate and save plot if requested
            plot_path = None
            if request.save_plot or request.show_plot:
                plot_path = self._generate_plot(regime_result, request)
                if request.show_plot:
                    self._show_plot(plot_path)

            return RegimeServiceResult(
                ticker=request.ticker,
                method=method,
                results=regime_result,
                plot_path=plot_path,
                results_path=results_path,
                success=regime_result.success,
                error=regime_result.error,
            )

        except Exception as e:
            self.logger.error(f"Regime detection failed for {request.ticker}: {e}")
            return RegimeServiceResult(
                ticker=request.ticker,
                method=method if "method" in locals() else RegimeMethod.RULE_BASED,
                results=None,
                success=False,
                error=str(e),
            )

    def _download_data(self, request: RegimeDetectionRequest) -> pd.DataFrame:
        """Download data using the download service"""
        # Calculate date range
        if request.timeframe:
            date_result = self.download_service.calculate_dates_from_timeframe(
                timeframe=request.timeframe,
                start_date=request.start,
                end_date=request.end,
            )
            start_date = date_result["start"]
            end_date = date_result["end"]
        else:
            defaults = self.download_service.get_default_dates()
            start_date = request.start or defaults["start"]
            end_date = request.end or defaults["end"]

        # Download data
        download_result = self.download_service.download_data(
            datasource=request.datasource,
            ticker=request.ticker,
            start=start_date,
            end=end_date,
            interval=request.interval,
            market=request.market_config,
            timeframe=request.timeframe,
        )

        if not download_result.success:
            raise ValueError(f"Data download failed: {download_result.error}")

        # Add ticker attribute to dataframe for reference
        download_result.data.ticker = request.ticker

        # Normalize column names for regime detector (expects title case)
        data = self._prepare_dataframe_for_regime_detection(download_result.data)
        data.ticker = request.ticker
        return data

    def _load_dataset_file(self, dataset_file: str, ticker: str) -> pd.DataFrame:
        """Load data from a dataset file"""
        file_path = Path(dataset_file)

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        # Load data based on file extension
        if file_path.suffix.lower() == ".csv":
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        elif file_path.suffix.lower() == ".json":
            # Try different JSON loading approaches
            try:
                # First try standard JSON loading (for OHLCV data)
                data = pd.read_json(file_path)

                # If the result is a dict-like structure, try converting to DataFrame
                if hasattr(data, "keys") and not isinstance(data, pd.DataFrame):
                    data = pd.DataFrame(data)

                # Ensure we have a proper datetime index
                if not isinstance(data.index, pd.DatetimeIndex):
                    if "date" in data.columns:
                        data.set_index("date", inplace=True)
                        data.index = pd.to_datetime(data.index, format="ISO8601")
                    elif data.index.name in ["date", "Date", "datetime"]:
                        data.index = pd.to_datetime(data.index, format="ISO8601")
                    else:
                        data.index = pd.to_datetime(data.index, format="ISO8601")

            except Exception as e:
                # Fallback: try orient='index' approach
                try:
                    data = pd.read_json(file_path, orient="index")
                    data.index = pd.to_datetime(data.index, format="ISO8601")
                except Exception as e2:
                    raise ValueError(
                        f"Could not load JSON file: {e}. Fallback attempt also failed: {e2}"
                    )
        else:
            raise ValueError(f"Unsupported dataset file format: {file_path.suffix}")

        # Validate required columns (check both lowercase and title case)
        required_columns_title = ["Open", "High", "Low", "Close", "Volume"]
        required_columns_lower = ["open", "high", "low", "close", "volume"]

        # Check if we have title case columns
        title_case_available = all(
            col in data.columns for col in required_columns_title
        )
        # Check if we have lowercase columns
        lower_case_available = all(
            col in data.columns for col in required_columns_lower
        )

        if not title_case_available and not lower_case_available:
            # Show what columns we actually have
            available_cols = list(data.columns)
            raise ValueError(
                f"Dataset file missing required columns. "
                f"Need either {required_columns_title} or {required_columns_lower}. "
                f"Available columns: {available_cols}"
            )

        # Add ticker attribute
        data.ticker = ticker

        # Normalize column names for regime detector (only if needed)
        if lower_case_available and not title_case_available:
            # Convert lowercase to title case
            data = self._prepare_dataframe_for_regime_detection(data)
        # If title case is already available, use as-is

        data.ticker = ticker
        return data

    def _prepare_dataframe_for_regime_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame with lowercase columns to title case format required by regime detectors
        """
        df_regime = df.copy()

        # Map lowercase column names to title case format expected by regime detectors
        column_mapping = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }

        # Rename columns to title case
        df_regime = df_regime.rename(columns=column_mapping)

        return df_regime

    def _save_results(
        self, regime_result: RegimeResult, request: RegimeDetectionRequest
    ) -> str:
        """Save regime detection results to file"""
        # Create organized directory structure
        # Use request method name for external methods, otherwise use enum value
        if isinstance(request.method, str):
            # External method - use the method name directly
            method_folder_name = request.method
        else:
            # Built-in method - use enum value
            method_folder_name = regime_result.method.value

        method_dir = self.regime_base_dir / method_folder_name
        self._ensure_directory_exists(method_dir)

        # Generate filename with new pattern: {ticker}_{market}_{start}_{end}.json
        if hasattr(regime_result.data, "index") and not regime_result.data.empty:
            start_date = regime_result.data.index.min().strftime("%Y-%m-%d")
            end_date = regime_result.data.index.max().strftime("%Y-%m-%d")
        else:
            start_date = "unknown"
            end_date = "unknown"

        filename = (
            f"{request.ticker}_{request.market_config}_{start_date}_{end_date}.json"
        )
        results_path = method_dir / filename

        # Prepare results data
        results_data = {
            "ticker": regime_result.ticker,
            "method": method_folder_name,  # Use the same method name as folder
            "calculation_time": regime_result.calculation_time,
            "parameters": regime_result.parameters,
            "current_regime": regime_result.current_regime.value,
            "regime_transitions": regime_result.regime_transitions,
            "detections": [
                {
                    "date": detection.date.isoformat(),
                    "regime": detection.regime.value,
                    "confidence": detection.confidence,
                    "method": detection.method.value,
                    "metadata": detection.metadata,
                }
                for detection in regime_result.detections
            ],
            "data_shape": regime_result.data.shape,
            "data_columns": list(regime_result.data.columns),
            "date_range": {"start": start_date, "end": end_date},
            "calculated_at": datetime.now().isoformat(),
        }

        # Save to JSON file
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        self.logger.info(f"Regime results saved to: {results_path}")
        return str(results_path)

    def _generate_plot(
        self, regime_result: RegimeResult, request: RegimeDetectionRequest
    ) -> str:
        """Generate interactive plot of regime detections"""
        # Create organized directory structure for plots
        # Use request method name for external methods, otherwise use enum value
        if isinstance(request.method, str):
            # External method - use the method name directly
            method_folder_name = request.method
        else:
            # Built-in method - use enum value
            method_folder_name = regime_result.method.value

        plots_dir = self.regime_base_dir / method_folder_name / "plots"
        self._ensure_directory_exists(plots_dir)

        # Generate filename with new pattern: {ticker}_{market}_{start}_{end}.html
        if hasattr(regime_result.data, "index") and not regime_result.data.empty:
            start_date = regime_result.data.index.min().strftime("%Y-%m-%d")
            end_date = regime_result.data.index.max().strftime("%Y-%m-%d")
        else:
            start_date = "unknown"
            end_date = "unknown"

        filename = (
            f"{request.ticker}_{request.market_config}_{start_date}_{end_date}.html"
        )
        plot_path = plots_dir / filename

        # Create subplot with multiple panels
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=(
                f"{request.ticker} - {regime_result.method.value.title()} Market Regime Detection",
                "Volume",
                "Volatility",
                "Market Regimes",
            ),
            row_heights=[0.4, 0.2, 0.2, 0.2],
        )

        # Add OHLC candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=regime_result.data.index,
                open=regime_result.data["Open"],
                high=regime_result.data["High"],
                low=regime_result.data["Low"],
                close=regime_result.data["Close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        # Add volume bars
        colors = [
            "red" if close < open else "green"
            for close, open in zip(
                regime_result.data["Close"], regime_result.data["Open"]
            )
        ]

        fig.add_trace(
            go.Bar(
                x=regime_result.data.index,
                y=regime_result.data["Volume"],
                marker_color=colors,
                name="Volume",
                opacity=0.6,
            ),
            row=2,
            col=1,
        )

        # Add volatility line
        vol_col = [col for col in regime_result.data.columns if "volatility" in col]
        if vol_col:
            fig.add_trace(
                go.Scatter(
                    x=regime_result.data.index,
                    y=regime_result.data[vol_col[0]],
                    mode="lines",
                    name="Volatility",
                    line=dict(color="orange", width=2),
                ),
                row=3,
                col=1,
            )

        # Add regime visualization
        regime_colors = {
            RegimeType.BULL: "green",
            RegimeType.BEAR: "red",
            RegimeType.SIDEWAYS: "gray",
            RegimeType.HIGH_VOLATILITY: "orange",
            RegimeType.LOW_VOLATILITY: "blue",
            RegimeType.CRISIS: "darkred",
            RegimeType.RECOVERY: "lightgreen",
        }

        # Create regime bands
        current_regime = None
        regime_start = None

        for i, detection in enumerate(regime_result.detections):
            if current_regime != detection.regime:
                # End previous regime band
                if current_regime is not None and regime_start is not None:
                    fig.add_vrect(
                        x0=regime_start,
                        x1=detection.date,
                        fillcolor=regime_colors.get(current_regime, "gray"),
                        opacity=0.3,
                        layer="below",
                        line_width=0,
                        row=4,
                        col=1,
                    )

                # Start new regime
                current_regime = detection.regime
                regime_start = detection.date

        # End final regime band
        if current_regime is not None and regime_start is not None:
            fig.add_vrect(
                x0=regime_start,
                x1=regime_result.data.index[-1],
                fillcolor=regime_colors.get(current_regime, "gray"),
                opacity=0.3,
                layer="below",
                line_width=0,
                row=4,
                col=1,
            )

        # Add regime transition markers
        for transition in regime_result.regime_transitions:
            fig.add_vline(
                x=transition["date"],
                line_dash="dash",
                line_color="black",
                line_width=1,
                opacity=0.7,
                annotation_text=f"{transition['from_regime']} → {transition['to_regime']}",
                annotation_position="top",
                row=4,
                col=1,
            )

        # Update layout
        fig.update_layout(
            title=f"{request.ticker} - {regime_result.method.value.title()} Market Regime Analysis",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template="plotly_white",
        )

        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Volatility", row=3, col=1)
        fig.update_yaxes(title_text="Regime", row=4, col=1)

        # Save plot
        fig.write_html(plot_path)

        self.logger.info(f"Regime plot saved to: {plot_path}")
        return str(plot_path)

    def _show_plot(self, plot_path: str) -> None:
        """Show plot in browser"""
        import webbrowser

        webbrowser.open(f"file://{plot_path}")

    def list_saved_results(
        self, method: Optional[str] = None, market: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """List saved regime detection results"""
        results = []

        if method:
            method_dirs = [self.regime_base_dir / method]
        else:
            method_dirs = [d for d in self.regime_base_dir.iterdir() if d.is_dir()]

        for method_dir in method_dirs:
            if not method_dir.is_dir():
                continue

            if market:
                market_dirs = [method_dir / market]
            else:
                market_dirs = [d for d in method_dir.iterdir() if d.is_dir()]

            for market_dir in market_dirs:
                if not market_dir.is_dir() or market_dir.name == "plots":
                    continue

                for result_file in market_dir.glob("*.json"):
                    try:
                        # Parse filename to extract info
                        name_parts = result_file.stem.split("_")
                        if len(name_parts) >= 3:
                            ticker = name_parts[0]
                            start_date = name_parts[1]
                            end_date = name_parts[2]
                        else:
                            ticker = result_file.stem
                            start_date = "unknown"
                            end_date = "unknown"

                        results.append(
                            {
                                "ticker": ticker,
                                "method": method_dir.name,
                                "market": market_dir.name,
                                "start_date": start_date,
                                "end_date": end_date,
                                "file_path": str(result_file),
                                "modified": datetime.fromtimestamp(
                                    result_file.stat().st_mtime
                                ).isoformat(),
                            }
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Could not parse result file {result_file}: {e}"
                        )

        # Sort by modification time (newest first)
        results.sort(key=lambda x: x["modified"], reverse=True)
        return results

    def load_saved_result(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load a saved regime detection result"""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load saved result from {file_path}: {e}")
            return None

    def delete_saved_result(self, file_path: str) -> bool:
        """Delete a saved regime detection result"""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()

                # Also delete corresponding plot if it exists
                plot_path = (
                    path.parent.parent
                    / "plots"
                    / path.parent.name
                    / (path.stem + ".html")
                )
                if plot_path.exists():
                    plot_path.unlink()

                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete saved result {file_path}: {e}")
            return False

    # Utility methods for CLI and UI integration
    def get_datasources(self) -> List[str]:
        """Get available datasources"""
        return self.download_service.get_datasources()

    def get_market_configs(self) -> List[str]:
        """Get available market configurations"""
        return self.download_service.get_market_configs()

    def get_available_timeframes(self) -> List[str]:
        """Get available timeframes"""
        return self.download_service.get_available_timeframes()

    def get_market_config_info(self, config: str) -> Optional[Dict[str, Any]]:
        """Get market configuration info"""
        return self.download_service.get_market_config_info(config)

    def calculate_dates_from_timeframe(
        self,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, str]:
        """Calculate dates from timeframe"""
        return self.download_service.calculate_dates_from_timeframe(
            timeframe, start_date, end_date
        )

    def get_timeframe_description(self, timeframe: str) -> str:
        """Get timeframe description"""
        return self.download_service.get_timeframe_description(timeframe)

    def get_default_dates(self) -> Dict[str, str]:
        """Get default date range"""
        return self.download_service.get_default_dates()

    def load_external_method(self, file_path: str) -> str:
        """Load an external regime detection method from a Python file and register it"""
        import importlib.util
        import inspect
        from pathlib import Path

        # Validate file path
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"External method file not found: {file_path}")

        if not path.suffix == ".py":
            raise ValueError(
                f"External method file must be a Python file (.py): {file_path}"
            )

        # Generate a unique module name
        module_name = f"external_regime_method_{path.stem}_{hash(str(path.absolute()))}"

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {file_path}")

        module = importlib.util.module_from_spec(spec)

        # Track registered methods before execution
        existing_methods = set(self.registry.list_regime_methods())

        # Execute the module
        spec.loader.exec_module(module)

        # Find newly registered methods
        new_methods = set(self.registry.list_regime_methods()) - existing_methods

        if not new_methods:
            # Look for regime detector classes in the module
            method_classes = []
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    inspect.isclass(attr)
                    and hasattr(attr, "detect")
                    and hasattr(attr, "get_default_parameters")
                    and hasattr(attr, "get_parameter_info")
                ):
                    method_classes.append((attr_name, attr))

            if len(method_classes) == 1:
                # Single detector class found, register it automatically
                class_name, class_obj = method_classes[0]
                # Use the class name as the method name
                method_name = class_name.lower()
                if method_name.endswith("detector"):
                    method_name = method_name[:-8]  # Remove 'detector' suffix
                if method_name.endswith("regime"):
                    method_name = method_name[:-6]  # Remove 'regime' suffix

                self.registry._regime_methods[method_name] = class_obj
                class_obj._registry_name = method_name

                # Create instance and add to detectors
                self.detectors[method_name] = class_obj()
                return method_name
            else:
                # Multiple or no classes found
                registered_method_name = list(new_methods)[0] if new_methods else None
                if registered_method_name:
                    method_class = self.registry._regime_methods[registered_method_name]
                    self.detectors[registered_method_name] = method_class()
                    return registered_method_name
                else:
                    raise ValueError(
                        f"No regime detection method found in {file_path}. "
                        f"Make sure to use @registry.register_regime_method decorator or implement a BaseRegimeDetector class."
                    )
        else:
            # Method was registered via decorator
            registered_method_name = list(new_methods)[0]
            method_class = self.registry._regime_methods[registered_method_name]
            self.detectors[registered_method_name] = method_class()
            return registered_method_name

    def str2bool(self, v: Union[bool, str]) -> bool:
        """Convert string to boolean (for CLI compatibility)"""
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise ValueError(f"Boolean value expected, got: {v}")
