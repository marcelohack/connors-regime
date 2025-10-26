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
pytest tests/test_regime_detector.py::TestRuleBasedRegimeDetector::test_detector_initialization

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
- `RuleBasedRegimeDetector`: Threshold-based detection using rolling returns and volatility
- `RegimeType` enum: bull, bear, sideways, high_volatility, low_volatility, crisis, recovery
- `RegimeMethod` enum: rule_based (more methods planned: clustering, HMM, regime_switching)
- `RegimeResult`: Container with detections, transitions, confidence scores, and enriched DataFrame

**Service Layer** (`connors_regime/services/regime_service.py`)
- `RegimeService`: High-level API orchestrating data fetching, detection, and file I/O
- Integrates with connors-datafetch for multi-source data (yfinance, polygon, finnhub, fmp)
- Handles external method loading via `load_external_method(file_path)`
- Saves results to `~/.connors/regime_detections/{method}/{ticker}_{market}_{start}_{end}.json`
- Generates interactive Plotly visualizations with OHLC, volume, volatility, and regime panels

### Data Flow

1. **Request** → `RegimeDetectionRequest` with ticker, method, datasource, date range
2. **Data Acquisition** → Downloads via connors-datafetch OR loads from CSV/JSON file
3. **Column Normalization** → Converts lowercase OHLCV to title case (Open, High, Low, Close, Volume)
4. **Feature Calculation** → Adds log_returns, rolling returns/volatility, SMA, RSI, volume metrics
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
  {method}/                          # e.g., "rule_based"
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
Rule-based detector default thresholds:
- `return_window`: 60 days
- `volatility_window`: 20 days
- `bull_return_threshold`: 0.10 (10%)
- `bear_return_threshold`: -0.10 (-10%)
- `high_volatility_threshold`: 0.25 (25%)
- `low_volatility_threshold`: 0.10 (10%)
- `crisis_return_threshold`: -0.20 (-20%)
- `crisis_volatility_threshold`: 0.35 (35%)

### Regime Classification Priority
1. **Crisis** (highest) - Large drawdown + extreme volatility
2. **Recovery** - Positive returns while still below SMA
3. **High/Low Volatility** - Based on volatility thresholds
4. **Bull/Bear/Sideways** - Based on rolling returns

### Confidence Calculation
Confidence (0.1-0.95) based on distance from thresholds:
- Strong signals (far from threshold) → higher confidence
- Weak signals (near threshold) → moderate confidence
- Minimum confidence: 0.1

## Testing

Test files are in `tests/` directory:
- `test_regime_detector.py`: Comprehensive tests for detectors and service
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
