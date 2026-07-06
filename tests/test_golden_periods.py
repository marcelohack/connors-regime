"""
Golden-period validation: the detector must reproduce historically
known market regimes on real SPY data.

These tests download data from Yahoo Finance, so they are opt-in:

    RUN_INTEGRATION=1 uv run pytest tests/test_golden_periods.py -v

If the detector fails any of these, its labels cannot be trusted on
unseen data — fix the detector, not the test.
"""

import os

import numpy as np
import pandas as pd
import pytest

from connors_regime import CompositeRegimeDetector, RegimeType

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION"),
    reason="integration test; set RUN_INTEGRATION=1 to run",
)


@pytest.fixture(scope="module")
def spy_result():
    """Detect regimes on SPY 2015-2024 (fetched once per test run)"""
    from connors_datafetch.services.datafetch_service import DataFetchService

    result = DataFetchService().download_data(
        datasource="yfinance",
        ticker="SPY",
        start="2015-01-01",
        end="2024-12-31",
        interval="1d",
        market="america",
    )
    assert result.success, f"SPY download failed: {result.error}"

    data = result.data.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    data.ticker = "SPY"

    detection = CompositeRegimeDetector().detect(data)
    assert detection.success, detection.error
    return detection


def regime_share(result, start: str, end: str) -> "dict[RegimeType, float]":
    """Fraction of detections per regime within a date window"""
    window = [
        d
        for d in result.detections
        if pd.Timestamp(start) <= d.date <= pd.Timestamp(end)
    ]
    assert window, f"no detections between {start} and {end}"
    return {
        regime: sum(1 for d in window if d.regime == regime) / len(window)
        for regime in RegimeType
    }


class TestGoldenPeriods:
    def test_2017_quiet_bull(self, spy_result):
        """2017: steady low-volatility bull market, no crisis"""
        share = regime_share(spy_result, "2017-01-01", "2017-12-31")
        assert share[RegimeType.BULL] > 0.5
        assert share[RegimeType.CRISIS] == 0.0
        assert share[RegimeType.BEAR] < 0.05

    def test_march_2020_crisis(self, spy_result):
        """COVID crash: fastest 30% drawdown in history"""
        share = regime_share(spy_result, "2020-03-01", "2020-04-30")
        assert share[RegimeType.CRISIS] > 0.5

    def test_2020_recovery(self, spy_result):
        """Post-COVID rebound while still below the pre-crash trend"""
        share = regime_share(spy_result, "2020-05-01", "2020-08-31")
        assert (share[RegimeType.RECOVERY] + share[RegimeType.BULL]) > 0.5
        assert share[RegimeType.BEAR] < 0.1

    def test_2022_bear(self, spy_result):
        """2022: sustained bear market (rate-hike drawdown)"""
        share = regime_share(spy_result, "2022-04-01", "2022-12-31")
        bearish = (
            share[RegimeType.BEAR]
            + share[RegimeType.CRISIS]
            + share[RegimeType.HIGH_VOLATILITY]
            + share[RegimeType.RECOVERY]
        )
        assert bearish > 0.5
        assert share[RegimeType.BULL] < 0.15

    def test_2023_2024_bull(self, spy_result):
        """2023-2024: sustained bull market"""
        share = regime_share(spy_result, "2023-06-01", "2024-12-31")
        assert share[RegimeType.BULL] > 0.5
        assert share[RegimeType.CRISIS] == 0.0


class TestSanityMetrics:
    def test_regimes_persist(self, spy_result):
        """A regime should last weeks, not days"""
        regimes = [d.regime for d in spy_result.detections]
        durations = []
        run = 1
        for prev, cur in zip(regimes, regimes[1:]):
            if cur == prev:
                run += 1
            else:
                durations.append(run)
                run = 1
        durations.append(run)

        assert np.mean(durations) >= 10

    def test_transition_count_reasonable(self, spy_result):
        """~10 years of data should not produce hundreds of transitions"""
        n_years = len(spy_result.detections) / 252
        transitions_per_year = len(spy_result.regime_transitions) / n_years
        assert transitions_per_year < 12

    def test_regime_return_profiles(self, spy_result):
        """Bull regimes should have positive mean returns; crisis
        regimes should be the most volatile"""
        df = spy_result.data.dropna(subset=["regime", "log_returns"])
        stats = df.groupby("regime")["log_returns"].agg(["mean", "std"])

        if "bull" in stats.index:
            assert stats.loc["bull", "mean"] > 0
        if "crisis" in stats.index:
            assert stats.loc["crisis", "std"] == stats["std"].max()
