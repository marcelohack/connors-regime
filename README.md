# Connors Regime - Market Regime Detection

A Python library for detecting market regimes using various algorithmic methods. Part of the Connors Trading framework.

## Features

- **Multiple Detection Methods:**
  - Rule-Based Regime Detection (threshold-based)
  - Future: Clustering, HMM, Regime-Switching models

- **Market Regime Types:**
  - Bull / Bear markets
  - Sideways / Ranging markets
  - High / Low volatility regimes
  - Crisis and Recovery periods

- **Flexible Integration:**
  - Programmatic API for Python applications
  - Support for external custom detection methods
  - Integration with connors-datafetch for data sourcing

- **Rich Output:**
  - Regime classifications with confidence scores
  - Transition detection and tracking
  - Interactive visualizations (Plotly)
  - JSON export for results

## Installation

### From GitHub (Development)
```bash
pip install git+https://github.com/marcelohack/connors-regime.git@main
```

### Local Development
```bash
git clone https://github.com/marcelohack/connors-regime.git
cd connors-regime
pip install -e .
```

## Quick Start

### Programmatic Usage

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

### Available Methods

```python
from connors_regime import RegimeService

service = RegimeService()

# List available detection methods
methods = service.get_available_methods()
print(methods)  # ['rule_based']

# Get method details
method_info = service.get_method_info("rule_based")
print(method_info["description"])
print(method_info["default_parameters"])
```

### Custom Parameters

```python
# Override default parameters
request = RegimeDetectionRequest(
    ticker="MSFT",
    method="rule_based",
    parameters={
        "bull_return_threshold": 0.15,  # 15% threshold for bull market
        "volatility_window": 30,        # 30-day volatility window
        "high_volatility_threshold": 0.30,
    },
    timeframe="2Y",  # Last 2 years
)

result = service.detect_regime(request)
```

## External Method Registration

You can create and register custom regime detection methods:

### Create Custom Detector

```python
# my_custom_detector.py
from connors_regime.core.registry import registry
from connors_regime.core.market_regime import BaseRegimeDetector, RegimeResult

@registry.register_regime_method("my_custom")
class MyCustomDetector(BaseRegimeDetector):
    def __init__(self):
        super().__init__(method="my_custom")

    def detect(self, data, **params):
        # Your custom detection logic here
        # Must return RegimeResult object
        pass

    def get_default_parameters(self):
        return {"param1": 10, "param2": 0.5}

    def get_parameter_info(self):
        return {
            "param1": {
                "type": "int",
                "default": 10,
                "description": "My parameter description"
            }
        }
```

### Load and Use Custom Detector

```python
from connors_regime import RegimeService

service = RegimeService()

# Load external method
method_name = service.load_external_method("my_custom_detector.py")

# Use it
request = RegimeDetectionRequest(
    ticker="TSLA",
    method=method_name,  # "my_custom"
    save_results=True,
)

result = service.detect_regime(request)
```

## Regime Detection Methods

### Rule-Based Detection

Classifies regimes based on configurable thresholds:
- **Rolling Returns** - Trend detection (bull/bear/sideways)
- **Rolling Volatility** - Volatility regime (high/low)
- **Price vs MA** - Trend strength
- **Volume Patterns** - Market participation

**Default Parameters:**
- `return_window`: 60 days
- `volatility_window`: 20 days
- `bull_return_threshold`: 0.10 (10%)
- `bear_return_threshold`: -0.10 (-10%)
- `high_volatility_threshold`: 0.25 (25%)
- `low_volatility_threshold`: 0.10 (10%)

## Data Requirements

Input data must be OHLCV format with columns:
- `Open`, `High`, `Low`, `Close`, `Volume`
- DateTime index

Data can be sourced from:
- connors-datafetch integration (yfinance, polygon, finnhub, etc.)
- Local CSV files
- Custom pandas DataFrames

## Output Format

### RegimeResult Object

```python
result.results.ticker              # Ticker symbol
result.results.current_regime      # Current regime (RegimeType enum)
result.results.detections          # List of RegimeDetection objects
result.results.regime_transitions  # List of transition events
result.results.calculation_time    # Processing time
result.results.data                # DataFrame with regime columns added
```

### Saved Results

Results are saved to `~/.connors/regime_detections/{method}/{ticker}_{market}_{start}_{end}.json`

Structure:
```json
{
  "ticker": "AAPL",
  "method": "rule_based",
  "current_regime": "bull",
  "calculation_time": 0.15,
  "parameters": {...},
  "detections": [...],
  "regime_transitions": [...]
}
```

## Requirements

- Python >=3.13
- pandas >=2.0.0
- numpy >=1.24.0
- plotly >=5.17.0
- connors-datafetch >=0.1.0

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Projects

- [connors-datafetch](https://github.com/marcelohack/connors-datafetch) - Multi-source financial data downloader
- [connors-trading](https://github.com/marcelohack/connors) - Main trading framework and playground
