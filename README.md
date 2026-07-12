# connors-regime

> Part of the [Connors Trading System](https://github.com/marcelohack/connors-playground)

## Overview

Market regime detection library. A single built-in `composite` method classifies two independent axes — **trend** (price vs long moving average + rolling return) and **volatility** (percentile rank of annualized volatility within the asset's own history) — and maps their combination onto bull, bear, sideways, high/low volatility, crisis, and recovery regimes. A hysteresis filter commits a regime change only after it persists for several bars, so regimes last weeks, not days.

The detector is validated against historically known regimes on real SPY data (2017 quiet bull, March 2020 crisis, 2020 recovery, 2022 bear, 2023–24 bull) — see `tests/test_golden_periods.py`.

## Features

- **Composite Detection**: trend × volatility grid, self-calibrating volatility thresholds (works unchanged on SPY, TSLA, or BTC), hysteresis against regime flickering
- **7 Regime Types**: Bull, Bear, Sideways, High/Low Volatility, Crisis, Recovery — with both raw axes preserved in detection metadata
- **External Methods**: Load custom detection algorithms from external Python files (for experimentation; the composite method is the single supported detector)
- **Rich Output**: Confidence scores, transition detection, interactive Plotly visualizations
- **Data Integration**: Works with connors-datafetch or custom DataFrames

## Installation

```bash
pip install git+https://github.com/marcelohack/connors-regime.git@main
```

### Local Development

**Prerequisites**: [uv](https://github.com/astral-sh/uv) (will install Python 3.13 if needed).
Sibling repos must be cloned alongside this one: `../core`, `../datafetch` (wired as editable path sources via `[tool.uv.sources]`).

```bash
uv sync --extra dev
```

uv reads `.python-version` to pick the interpreter and creates `.venv/` automatically. Run commands with `uv run <cmd>` (no activation needed), or `source .venv/bin/activate`.

## Quick Start

```python
from connors_regime import RegimeService, RegimeDetectionRequest

# Initialize service
service = RegimeService()

# Create detection request
request = RegimeDetectionRequest(
    ticker="AAPL",
    method="composite",
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

Installing the package provides the `connors-regime` command (also runnable as `python -m connors_regime.cli`):

```bash
# Basic regime detection
connors-regime --ticker AAPL --method composite --timespan 2Y

# With custom parameters
connors-regime --ticker MSFT --method composite \
  --method-params "confirm_days:10;volatility_window:30"

# With plotting and saving
connors-regime --ticker NVDA --method composite \
  --timespan 1Y --plot --save-results --save-plot

# External (experimental) detection method
connors-regime --ticker TSLA \
  --external-method ~/.connors/regime_methods/my_method.py --timespan 6M

# Different markets and data sources
connors-regime --ticker BHP --method composite \
  --market australia --datasource yfinance --timespan 1Y

# Using dataset file
connors-regime --ticker CUSTOM --method composite \
  --dataset-file my_data.csv --plot

# Show method parameters
connors-regime --method composite --show-method-params

# List methods and saved results
connors-regime --list-methods
connors-regime --list-saved
```

## Composite Detection

Two independent axes are classified per bar and combined:

**Trend axis** — `bull` / `neutral` / `bear` from price distance to a long SMA plus the sign of the rolling return.

**Volatility axis** — `low` / `normal` / `high` / `extreme` from the percentile rank of current annualized volatility within the asset's own trailing history. Because thresholds are percentiles, they self-calibrate per asset: no retuning between SPY, TSLA, and BTC.

**Mapping to regimes** — bear trend + extreme vol, or a >20% drawdown with elevated vol → `crisis`; a strong short-window rebound while still below the long SMA → `recovery`; bull/bear trend → `bull`/`bear` (volatility state kept in metadata); neutral trend is labeled by its volatility state (`high_volatility` / `low_volatility` / `sideways`).

**Hysteresis** — a new regime must persist `confirm_days` consecutive bars before it is committed (`crisis_confirm_days` for crisis, so stress is flagged fast). Confidence is the share of recent raw labels agreeing with the committed regime.

Key parameters (see `--show-method-params` for the full list):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `return_window` | 60 bars | Rolling return window for the trend axis |
| `volatility_window` | 20 bars | Window for annualized volatility |
| `vol_percentile_window` | 252 bars | Trailing history for the volatility percentile rank |
| `trend_window` | 200 bars | Long SMA for the trend axis |
| `vol_low_pct` / `vol_high_pct` / `vol_extreme_pct` | 0.20 / 0.80 / 0.95 | Volatility percentile buckets |
| `crisis_drawdown` | -0.20 | Drawdown from trailing peak qualifying as crisis |
| `confirm_days` | 5 | Bars a new regime must persist before committing |

## Validation

Unit tests run on synthetic data with known regimes. The golden-period suite validates against real SPY history — 2017 quiet bull, March 2020 crisis, 2020 recovery, 2022 bear, 2023–24 bull — plus sanity metrics (regimes persist ≥10 bars on average, ~<12 transitions/year, bull regimes have positive mean returns):

```bash
RUN_INTEGRATION=1 uv run pytest tests/test_golden_periods.py -v
```

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

Results are saved to `~/.connors/regime_detections/{method}/{ticker}_{start}_{end}.json`

```json
{
  "ticker": "AAPL",
  "method": "composite",
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
| [connors-playground](https://github.com/marcelohack/connors-playground) | Workspace hub + API token manager | [README](https://github.com/marcelohack/connors-playground#readme) |
| [connors-core](https://github.com/marcelohack/connors-core) | Registry, config, indicators, metrics | [README](https://github.com/marcelohack/connors-core#readme) |
| [connors-backtest](https://github.com/marcelohack/connors-backtest) | Backtesting service + built-in strategies | [README](https://github.com/marcelohack/connors-backtest#readme) |
| [connors-strategies](https://github.com/marcelohack/connors-strategies) | Trading strategy collection (private) | — |
| [connors-screener](https://github.com/marcelohack/connors-screener) | Stock screening system | [README](https://github.com/marcelohack/connors-screener#readme) |
| [connors-datafetch](https://github.com/marcelohack/connors-datafetch) | Multi-source data downloader | [README](https://github.com/marcelohack/connors-datafetch#readme) |
| [connors-sr](https://github.com/marcelohack/connors-sr) | Support & Resistance calculator | [README](https://github.com/marcelohack/connors-sr#readme) |
| [connors-bots](https://github.com/marcelohack/connors-bots) | Automated trading bots | [README](https://github.com/marcelohack/connors-bots#readme) |

## License

MIT
