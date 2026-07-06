#!/usr/bin/env python3
"""
CLI for Market Regime Detector

Provides command-line interface for detecting market regimes using various methods
with data integration and visualization capabilities.

Installed as the ``connors-regime`` console script; also runnable via
``python -m connors_regime.cli``.
"""

import argparse
from pathlib import Path
from typing import Any, Dict

from connors_regime.services.regime_service import RegimeDetectionRequest, RegimeService


def _parse_method_params(param_string: str) -> Dict[str, Any]:
    """Parse method parameter string into dictionary"""
    params = {}
    for pair in param_string.split(";"):
        if ":" in pair:
            key, value = pair.split(":", 1)
            key = key.strip()
            value = value.strip()

            # Handle different value types
            if value.startswith("[") and value.endswith("]"):
                # Parse list [1.0, 2.0]
                list_values = []
                for v in value[1:-1].split(","):
                    v = v.strip()
                    try:
                        if "." in v:
                            list_values.append(float(v))
                        else:
                            list_values.append(int(v))
                    except ValueError:
                        list_values.append(v)  # Keep as string
                params[key] = list_values
            else:
                # Try to convert to appropriate type
                try:
                    if "." in value:
                        params[key] = float(value)
                    elif value.isdigit():
                        params[key] = int(value)
                    elif value.lower() in ["true", "false"]:
                        params[key] = value.lower() == "true"
                    else:
                        params[key] = value
                except ValueError:
                    params[key] = value

    return params


def main() -> None:
    """Main entry point for Market Regime Detector CLI"""

    # Initialize regime service
    regime_service = RegimeService()

    parser = argparse.ArgumentParser(
        prog="connors-regime",
        description="Detect market regimes using various algorithmic methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Basic regime detection with default settings
  connors-regime --ticker AAPL --method composite

  # Composite method with custom parameters
  connors-regime --ticker AAPL --method composite --method-params "confirm_days:10;volatility_window:30"

  # Detection with timespan and plotting
  connors-regime --ticker MSFT --method composite --timespan 2Y --plot --save-plot

  # Using dataset file
  connors-regime --ticker TSLA --method composite --dataset-file data.csv --save-results

  # Different market configurations
  connors-regime --ticker BHP --method composite --market australia --timespan 1Y

Available methods: {', '.join(regime_service.get_available_methods())}
Available datasources: {', '.join(regime_service.get_datasources())}
Available markets: {', '.join(regime_service.get_market_configs())}
Available timespans: {', '.join(regime_service.get_available_timeframes())}
        """,
    )

    # Required arguments (conditional based on flags)
    parser.add_argument(
        "--ticker",
        type=str,
        help="Stock ticker symbol to analyze (e.g., AAPL, MSFT, TSLA)",
    )

    # Create mutually exclusive group for method selection
    method_group = parser.add_mutually_exclusive_group(required=True)

    method_group.add_argument(
        "--method",
        type=str,
        choices=regime_service.get_available_methods(),
        help="Built-in market regime detection method",
    )

    method_group.add_argument(
        "--external-method",
        type=str,
        help="Path to external Python file containing regime detection method (must use @registry.register_regime_method decorator)",
    )

    # Data source options
    parser.add_argument(
        "--datasource",
        type=str,
        default="yfinance",
        choices=regime_service.get_datasources(),
        help="Data source for quotes (default: yfinance)",
    )

    parser.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help="CSV or JSON file path to use as dataset instead of downloading (must contain OHLCV data)",
    )

    # Date range options
    parser.add_argument(
        "--start", type=str, default=None, help="Start date for analysis (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end", type=str, default=None, help="End date for analysis (YYYY-MM-DD)"
    )

    # Get available timeframes from service
    available_timeframes = regime_service.get_available_timeframes()
    parser.add_argument(
        "--timespan",
        choices=available_timeframes,
        help=f"Pre-defined timespan instead of custom dates. Available: {', '.join(available_timeframes)}",
    )

    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"],
        help='Time interval for quotes (default: "1d")',
    )

    # Market configuration
    parser.add_argument(
        "--market",
        type=str,
        default="america",
        choices=regime_service.get_market_configs(),
        help="Market configuration (default: america)",
    )

    # Method parameters
    parser.add_argument(
        "--method-params",
        type=str,
        default=None,
        help='Override method parameters (format: "key1:value1;key2:value2")',
    )

    parser.add_argument(
        "--show-method-params",
        action="store_true",
        help="Show available parameters for the specified method",
    )

    # Output options
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save detection results to file",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and display interactive plot in browser",
    )

    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save plot to file",
    )

    # Information options
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="List all available detection methods and exit",
    )

    parser.add_argument(
        "--list-datasources",
        action="store_true",
        help="List all available datasources and exit",
    )

    parser.add_argument(
        "--list-markets",
        action="store_true",
        help="List all available market configurations and exit",
    )

    parser.add_argument(
        "--list-saved",
        action="store_true",
        help="List saved regime detection results and exit",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Handle information requests
    if args.list_methods:
        print("📋 Available market regime detection methods:")
        methods_info = regime_service.get_all_methods_info()
        for method, info in methods_info.items():
            print(f"\n🔹 {method}:")
            print(
                f"   Description: {info.get('description', 'No description available')}"
            )
            print(f"   Default parameters: {info.get('default_parameters', {})}")
        return

    if args.list_datasources:
        print("📋 Available datasources:")
        for ds in regime_service.get_datasources():
            print(f"  - {ds}")
        return

    if args.list_markets:
        print("🌍 Available market configurations:")
        for market in regime_service.get_market_configs():
            config_info = regime_service.get_market_config_info(market)
            if config_info:
                suffix_info = (
                    f" (suffix: {config_info['yf_ticker_suffix']})"
                    if config_info["yf_ticker_suffix"]
                    else " (no suffix)"
                )
                print(f"  - {market}: {config_info['name']}{suffix_info}")
        return

    if args.list_saved:
        print("💾 Saved regime detection results:")
        saved_results = regime_service.list_saved_results()
        if not saved_results:
            print("  No saved results found.")
            return

        for result in saved_results[:20]:  # Show last 20 results
            print(
                f"  🎯 {result['ticker']} ({result['method']}) - {result['market']} - {result['start_date']} to {result['end_date']}"
            )
            print(f"      File: {result['file_path']}")
            print(f"      Modified: {result['modified']}")

        if len(saved_results) > 20:
            print(f"  ... and {len(saved_results) - 20} more results")
        return

    # Handle show method parameters
    if args.show_method_params:
        method_info = regime_service.get_method_info(args.method)
        if not method_info:
            print(
                f"❌ Method '{args.method}' not found. Available: {regime_service.get_available_methods()}"
            )
            return

        print(f"Method: {args.method}")
        print(
            f"Description: {method_info.get('description', 'No description available')}"
        )
        print("\n📊 Available Parameters:")

        param_info = method_info.get("parameter_info", {})
        if not param_info:
            print("  No configurable parameters for this method.")
        else:
            for param_name, param_details in param_info.items():
                print(f"  🔹 {param_name}:")
                print(f"      Type: {param_details.get('type', 'unknown')}")
                print(f"      Default: {param_details.get('default', 'N/A')}")
                if "min" in param_details and "max" in param_details:
                    print(
                        f"      Range: {param_details['min']} - {param_details['max']}"
                    )
                print(
                    f"      Description: {param_details.get('description', 'No description')}"
                )

        print(f"\nDefault parameters: {method_info.get('default_parameters', {})}")
        return

    # Validate required arguments for actual detection
    if not args.ticker:
        print("❌ Error: --ticker is required for regime detection")
        return

    # Method selection is already enforced by mutually exclusive group

    # Validate dataset file if provided
    if args.dataset_file:
        dataset_path = Path(args.dataset_file)
        if not dataset_path.exists():
            print(f"❌ Dataset file not found: {args.dataset_file}")
            return

        print(f"📁 Using dataset file: {args.dataset_file}")
        print(
            f"⚠️  Note: When using dataset file, start/end dates and datasource are ignored"
        )

    print("=" * 100)
    print("🎯 MARKET REGIME DETECTOR")
    print("=" * 100)

    print(f"📊 Ticker: {args.ticker}")
    print(f"🔍 Method: {args.method if args.method else 'External'}")

    if not args.dataset_file:
        print(f"📡 Data Source: {args.datasource}")
        print(f"⏱️  Interval: {args.interval}")

    # Get market configuration info
    market_config_info = regime_service.get_market_config_info(args.market)
    if market_config_info:
        print(f"🌍 Market: {market_config_info['name']}")
        if market_config_info.get("yf_ticker_suffix"):
            print(f"🏷️  Ticker Suffix: {market_config_info['yf_ticker_suffix']}")

    # Get date range
    if args.timespan and not args.dataset_file:
        date_result = regime_service.calculate_dates_from_timeframe(
            timeframe=args.timespan, start_date=args.start, end_date=args.end
        )
        start_str = date_result["start"]
        end_str = date_result["end"]
        timespan_desc = regime_service.get_timeframe_description(args.timespan)
        print(f"📅 Timespan: {args.timespan} ({timespan_desc})")
        print(f"📅 Analysis Period: {start_str} to {end_str}")
    elif args.start and args.end and not args.dataset_file:
        start_str = args.start
        end_str = args.end
        print(f"📅 Analysis Period: {start_str} to {end_str}")
    elif not args.dataset_file:
        # Use default dates
        default_dates = regime_service.get_default_dates()
        start_str = args.start or default_dates["start"]
        end_str = args.end or default_dates["end"]
        print(f"📅 Analysis Period: {start_str} to {end_str} (default)")

    # Parse method parameters
    method_params = None
    if args.method_params:
        try:
            method_params = _parse_method_params(args.method_params)
            print(f"⚙️  Method parameter overrides: {args.method_params}")
        except Exception as e:
            print(f"❌ Error parsing method parameters: {e}")
            return

    print("=" * 100)

    # Determine which method to use
    if args.external_method:
        try:
            loaded_method_name = regime_service.load_external_method(args.external_method)
            print(f"✅ Loaded external regime detection method: {loaded_method_name}")
            selected_method = loaded_method_name
        except Exception as e:
            print(f"❌ Failed to load external regime detection method: {e}")
            return
    else:
        selected_method = args.method

    # Create detection request
    request = RegimeDetectionRequest(
        ticker=args.ticker,
        method=selected_method,
        parameters=method_params,
        datasource=args.datasource if not args.dataset_file else "file",
        dataset_file=args.dataset_file,
        start=start_str if not args.dataset_file else None,
        end=end_str if not args.dataset_file else None,
        interval=args.interval,
        market_config=args.market,
        timeframe=args.timespan if not args.dataset_file else None,
        save_results=args.save_results,
        save_plot=args.save_plot,
        show_plot=args.plot,
    )

    # Run detection
    try:
        print(f"🔄 Detecting {selected_method.title()} market regimes for {args.ticker}...")

        result = regime_service.detect_regime(request)

        if not result.success:
            print(f"❌ Detection failed: {result.error}")
            return

        regime_result = result.results

        print(f"✅ Detection completed in {regime_result.calculation_time:.2f} seconds")
        print(f"📊 Data points analyzed: {len(regime_result.data)}")
        print(f"🎯 Current regime: {regime_result.current_regime.value.upper()}")
        print(
            f"🔄 Regime transitions detected: {len(regime_result.regime_transitions)}"
        )

        # Display current regime details
        if regime_result.detections:
            current_detection = regime_result.detections[-1]
            print(f"📈 Current regime confidence: {current_detection.confidence:.1%}")

            if current_detection.metadata:
                print("📋 Current market metrics:")
                metadata = current_detection.metadata
                if "return_value" in metadata:
                    print(f"    📈 Rolling Return: {metadata['return_value']:.1%}")
                if "volatility_value" in metadata:
                    print(f"    📊 Volatility: {metadata['volatility_value']:.1%}")
                if "price_vs_sma" in metadata:
                    print(f"    📉 Price vs SMA: {metadata['price_vs_sma']:.1%}")

        # Display regime transitions
        if regime_result.regime_transitions:
            print("\n" + "=" * 80)
            print("🔄 RECENT REGIME TRANSITIONS")
            print("=" * 80)

            # Show last 10 transitions
            recent_transitions = regime_result.regime_transitions[-10:]
            for i, transition in enumerate(recent_transitions, 1):
                date_str = (
                    transition["date"].strftime("%Y-%m-%d")
                    if hasattr(transition["date"], "strftime")
                    else str(transition["date"])
                )
                print(
                    f"{i:2d}. {date_str}: {transition['from_regime'].upper()} → {transition['to_regime'].upper()} (conf: {transition['confidence']:.1%})"
                )

            if len(regime_result.regime_transitions) > 10:
                print(
                    f"    ... and {len(regime_result.regime_transitions) - 10} more transitions"
                )

        # Display regime statistics
        if regime_result.detections:
            print("\n" + "=" * 80)
            print("📊 REGIME DISTRIBUTION")
            print("=" * 80)

            # Count regime occurrences
            regime_counts = {}
            for detection in regime_result.detections:
                regime = detection.regime.value
                regime_counts[regime] = regime_counts.get(regime, 0) + 1

            total_periods = len(regime_result.detections)

            for regime, count in sorted(regime_counts.items()):
                percentage = (count / total_periods) * 100
                print(f"  🔹 {regime.upper()}: {count:4d} periods ({percentage:5.1f}%)")

        # Show file paths if results were saved
        if result.results_path:
            print(f"\n💾 Results saved to: {result.results_path}")

        if result.plot_path:
            print(f"📊 Plot saved to: {result.plot_path}")
            if args.plot:
                print("🌐 Opening plot in browser...")

        print("\n" + "=" * 100)

        # Display method parameters used
        if args.verbose:
            print(f"⚙️  Method parameters used: {regime_result.parameters}")

            print(f"📈 DataFrame columns added:")
            original_cols = ["Open", "High", "Low", "Close", "Volume"]
            new_cols = [
                col for col in regime_result.data.columns if col not in original_cols
            ]
            for col in new_cols:
                print(f"     - {col}")

    except Exception as e:
        print(f"❌ Detection failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return

    print("✨ Market regime detection completed!")


if __name__ == "__main__":
    main()
