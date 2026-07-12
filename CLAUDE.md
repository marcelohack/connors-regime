# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Connors Regime is a Python library for detecting market regimes using various algorithmic methods. It's part of the Connors Trading framework and integrates with connors-datafetch for data sourcing.

## Development Commands

### Setup and Installation
```bash
# Local development installation
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_regime_detector.py

# Run specific test class or method
pytest tests/test_regime_detector.py::TestCompositeRegimeDetector::test_detector_initialization

# Run golden-period validation against real SPY history (downloads data)
RUN_INTEGRATION=1 pytest tests/test_golden_periods.py -v

# Run with coverage
pytest --cov=connors_regime --cov-report=html

# Run with verbose output
pytest -v
```

### Code Quality
```bash
# Format code with black
black connors_regime tests

# Sort imports
isort connors_regime tests

# Lint code
flake8 connors_regime tests

# Type checking
mypy connors_regime
```

## Architecture

### Core Components

**Registry System** (`connors_regime/core/registry.py`)
- Minimal decorator-based registry for regime detection methods
- Enables external custom detector registration via `@registry.register_regime_method("name")`
- Global `registry` instance used throughout the codebase

**Regime Detection** (`connors_regime/core/market_regime.py`)
- `BaseRegimeDetector`: Abstract base class for all detectors
- `CompositeRegimeDetector`: The single built-in method. Classifies two independent axes — trend (`TrendState`: bull/neutral/bear from price vs long SMA + rolling return sign) and volatility (`VolatilityState`: low/normal/high/extreme from the percentile rank of annualized volatility within the asset's own trailing history) — then maps the grid onto `RegimeType` labels. Hysteresis commits a regime change only after `confirm_days` consecutive bars (`crisis_confirm_days` for crisis).
- `RegimeType` enum: bull, bear, sideways, high_volatility, low_volatility, crisis, recovery
- `RegimeMethod` enum: composite (deliberately the only built-in — add new methods only for a specific validated need, and make them pass the golden-period suite first)
- `RegimeResult`: Container with detections, transitions, confidence scores, and enriched DataFrame

**CLI** (`connors_regime/cli.py`)
- Installed as the `connors-regime` console script (also `python -m connors_regime.cli`)
- Argument choices (methods, datasources, markets, timespans) come dynamically from `RegimeService`
- Supports `--method-params "key1:value1;key2:value2"` overrides, `--external-method` loading, dataset files, plotting/saving, and `--list-*` / `--show-method-params` info flags

**Service Layer** (`connors_regime/services/regime_service.py`)
- `RegimeService`: High-level API orchestrating data fetching, detection, and file I/O
- Integrates with connors-datafetch for multi-source data (yfinance, polygon, finnhub, fmp)
- Handles external method loading via `load_external_method(file_path)`
- Saves results to `~/.connors/regime_detections/{method}/{ticker}_{start}_{end}.json`
- Generates interactive Plotly visualizations with OHLC, volume, volatility, and regime panels

### Data Flow

1. **Request** → `RegimeDetectionRequest` with ticker, method, datasource, date range
2. **Data Acquisition** → Downloads via connors-datafetch OR loads from CSV/JSON file
3. **Column Normalization** → Converts lowercase OHLCV to title case (Open, High, Low, Close, Volume)
4. **Feature Calculation** → Adds log_returns, rolling returns, annualized volatility + its percentile rank, trend SMA, drawdown from trailing peak
5. **Detection** → Applies detector logic, generates `RegimeDetection` objects per date
6. **Enrichment** → Adds regime and confidence columns to DataFrame
7. **Results** → Returns `RegimeServiceResult` with plots, JSON results, transition events

### External Method Registration

Custom detectors can be loaded dynamically:
```python
# External detector file must:
# 1. Inherit from BaseRegimeDetector
# 2. Implement detect(), get_default_parameters(), get_parameter_info()
# 3. Use @registry.register_regime_method("name") decorator OR have a single detector class

service.load_external_method("my_detector.py")  # Returns method name
```

Service auto-detects:
- Single detector class → auto-registers with class name
- Decorated class → uses decorator-provided name
- Multiple/no classes → raises error

### File Storage Structure

```
~/.connors/regime_detections/
  {method}/                          # e.g., "composite"
    {ticker}_{market}_{start}_{end}.json
    plots/
      {ticker}_{market}_{start}_{end}.html
```

## Key Implementation Details

### Data Validation
- Requires minimum 30 data points for regime detection
- Validates OHLCV columns (title case: Open, High, Low, Close, Volume)
- Accepts both lowercase and title case on input, normalizes to title case

### Detection Parameters
Composite detector defaults (see `get_parameter_info()` for the full list):
- `return_window`: 60 bars (rolling return, trend axis)
- `volatility_window`: 20 bars (annualized with sqrt(252) — this scaling is load-bearing; a regression test guards it)
- `vol_percentile_window`: 252 bars trailing history for the volatility percentile rank
- `trend_window`: 200-bar SMA; `trend_threshold`: 0.02 distance to call bull/bear
- `vol_low_pct` / `vol_high_pct` / `vol_extreme_pct`: 0.20 / 0.80 / 0.95 percentile buckets
- `crisis_drawdown`: -0.20 from trailing 252-bar peak
- `confirm_days`: 5 (hysteresis); `crisis_confirm_days`: 2

### Regime Classification Priority
1. **Crisis** (highest) - Deep drawdown + elevated volatility, or bear trend + extreme volatility
2. **Recovery** - Strong short-window rebound (return_window/3 bars) while still below trend SMA with remaining drawdown
3. **Bull/Bear** - From the trend axis (volatility state preserved in detection metadata)
4. **High/Low Volatility / Sideways** - Neutral trend labeled by its volatility state

### Confidence Calculation
Confidence (0.1-0.95) is the share of recent raw (pre-hysteresis) labels that agree with the committed regime — an agreement ratio, not a distance-from-threshold heuristic.

### Validation Philosophy
`tests/test_golden_periods.py` asserts the detector reproduces historically known SPY regimes (2017 quiet bull, March 2020 crisis, 2020 recovery, 2022 bear, 2023-24 bull) plus persistence/transition-rate sanity metrics. If a detector change breaks these, fix the detector, not the test.

## Testing

Test files are in `tests/` directory:
- `test_regime_detector.py`: Comprehensive tests for detectors and service
- `test_cli.py`: CLI tests (argument handling, parameter parsing, service integration via mocks)
- `test_golden_periods.py`: Real-SPY validation suite (gated by `RUN_INTEGRATION=1`)
- Uses pytest fixtures for sample OHLCV data generation
- Mocks external dependencies (data downloads)
- Tests detector logic, service operations, file I/O, CLI integration

## Python Version

Requires Python >=3.13

## Dependencies

Core runtime:
- pandas >=2.0.0
- numpy >=1.24.0
- plotly >=5.17.0
- connors-datafetch >=0.1.0

Development:
- pytest >=7.4.0
- black, isort, flake8, mypy
