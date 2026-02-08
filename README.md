# connors-regime

> Part of the [Connors Trading System](https://github.com/marcelohack/connors-playground)

## Overview

Market regime detection library using algorithmic methods to classify market conditions. Identifies bull, bear, sideways, high/low volatility, crisis, and recovery regimes with confidence scores and transition tracking.

## Features

- **Rule-Based Detection**: Threshold-based regime classification using rolling returns and volatility
- **7 Regime Types**: Bull, Bear, Sideways, High/Low Volatility, Crisis, Recovery
- **External Methods**: Load custom detection algorithms from external Python files
- **Rich Output**: Confidence scores, transition detection, interactive Plotly visualizations
- **Data Integration**: Works with connors-datafetch or custom DataFrames

## Installation

```bash
pip install git+https://github.com/marcelohack/connors-regime.git@main
```

### Local Development

**Prerequisites**: Python 3.13, [pyenv](https://github.com/pyenv/pyenv) + [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)

```bash
# 1. Create and activate a virtual environment
pyenv virtualenv 3.13 connors-regime
pyenv activate connors-regime

# 2. Install connors packages from local checkouts (not on PyPI)
pip install -e ../core
pip install -e ../datafetch

# 3. Install with dev dependencies
pip install -e ".[dev]"
```

A `.python-version` file is included so pyenv auto-activates when you `cd` into this directory.

## Quick Start

```python
from connors_regime import RegimeService, RegimeDetectionRequest

# Initialize service
service = RegimeService()

# Create detection request
request = RegimeDetectionRequest(
    ticker="AAPL",
    method="rule_based",
    datasource="yfinance",
    start="2023-01-01",
    end="2024-01-01",
    interval="1d",
    market_config="america",
    save_results=True,
    save_plot=True,
)

# Run detection
result = service.detect_regime(request)

# Access results
print(f"Current Regime: {result.results.current_regime.value}")
print(f"Transitions: {len(result.results.regime_transitions)}")
print(f"Confidence: {result.results.detections[-1].confidence:.1%}")
```

## CLI Usage

The regime detection CLI is part of [connors-playground](https://github.com/marcelohack/connors-playground):

```bash
# Basic regime detection
python -m connors.cli.regime_detector --ticker AAPL --method rule_based --timespan 2Y

# With custom thresholds
python -m connors.cli.regime_detector --ticker MSFT --method rule_based \
  --method-params "bull_return_threshold:0.15;volatility_window:30"

# With plotting and saving
python -m connors.cli.regime_detector --ticker NVDA --method rule_based \
  --timespan 1Y --plot --save-results --save-plot

# External detection method
python -m connors.cli.regime_detector --ticker TSLA \
  --external-method ~/.connors/regime_methods/test_regime_method.py \
  --method-params "detection_method:momentum;lookback_period:30" --timespan 6M

# Different markets and data sources
python -m connors.cli.regime_detector --ticker BHP --method rule_based \
  --market australia --datasource yfinance --timespan 1Y

# Using dataset file
python -m connors.cli.regime_detector --ticker CUSTOM --method rule_based \
  --dataset-file my_data.csv --plot

# Show method parameters
python -m connors.cli.regime_detector --method rule_based --show-method-params

# List methods and saved results
python -m connors.cli.regime_detector --list-methods
python -m connors.cli.regime_detector --list-saved
```

## Rule-Based Detection

Classifies regimes based on configurable thresholds:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `return_window` | 60 days | Rolling window for return calculations |
| `volatility_window` | 20 days | Rolling window for volatility |
| `bull_return_threshold` | 0.10 (10%) | Minimum return for bull classification |
| `bear_return_threshold` | -0.10 (-10%) | Maximum return for bear classification |
| `high_volatility_threshold` | 0.25 (25%) | Above this = high volatility regime |
| `low_volatility_threshold` | 0.10 (10%) | Below this = low volatility regime |
| `crisis_return_threshold` | -0.20 | Extreme negative return threshold |
| `crisis_volatility_threshold` | 0.35 | Extreme volatility threshold |

## Custom Detection Methods

```python
from connors_regime.core.registry import registry
from connors_regime.core.market_regime import BaseRegimeDetector, RegimeResult

@registry.register_regime_method("my_custom")
class MyCustomDetector(BaseRegimeDetector):
    def __init__(self):
        super().__init__(method="my_custom")

    def detect(self, data, **params):
        # Your custom detection logic
        return RegimeResult(...)

    def get_default_parameters(self):
        return {"param1": 10, "param2": 0.5}

    def get_parameter_info(self):
        return {
            "param1": {"type": "int", "default": 10, "description": "..."}
        }
```

## Output Format

Results are saved to `~/.connors/regime_detections/{method}/{ticker}_{market}_{start}_{end}.json`

```json
{
  "ticker": "AAPL",
  "method": "rule_based",
  "current_regime": "bull",
  "calculation_time": 0.15,
  "parameters": {},
  "detections": [],
  "regime_transitions": []
}
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=connors_regime
```

## Related Packages

| Package | Description | Links |
|---------|-------------|-------|
| [connors-playground](https://github.com/marcelohack/connors-playground) | CLI + Streamlit UI (integration hub) | [README](https://github.com/marcelohack/connors-playground#readme) |
| [connors-core](https://github.com/marcelohack/connors-core) | Registry, config, indicators, metrics | [README](https://github.com/marcelohack/connors-core#readme) |
| [connors-backtest](https://github.com/marcelohack/connors-backtest) | Backtesting service + built-in strategies | [README](https://github.com/marcelohack/connors-backtest#readme) |
| [connors-strategies](https://github.com/marcelohack/connors-strategies) | Trading strategy collection (private) | — |
| [connors-screener](https://github.com/marcelohack/connors-screener) | Stock screening system | [README](https://github.com/marcelohack/connors-screener#readme) |
| [connors-datafetch](https://github.com/marcelohack/connors-datafetch) | Multi-source data downloader | [README](https://github.com/marcelohack/connors-datafetch#readme) |
| [connors-sr](https://github.com/marcelohack/connors-sr) | Support & Resistance calculator | [README](https://github.com/marcelohack/connors-sr#readme) |
| [connors-bots](https://github.com/marcelohack/connors-bots) | Automated trading bots | [README](https://github.com/marcelohack/connors-bots#readme) |

## License

MIT
