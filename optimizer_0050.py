#!/usr/bin/env python3
"""Standalone 0050 representative-universe Swing parameter optimizer.

The strategy enters a trend-confirmed volume breakout, exits on SMA defence
failure, and settles any position after at most 20 trading days.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd

from twbacktest_candlestick.indicators import add_indicators
from twbacktest_candlestick.providers import PriceRequest, ProviderRouter, TwseDailyProvider, YFinanceProvider


LOG = logging.getLogger(__name__)
# Representative large-cap 0050-style pool.  Override with --tickers when needed.
DEFAULT_0050_TICKERS = (
    "2330", "2317", "2454", "2308", "2382", "2303", "2327", "2383", "2412", "3034",
    "2881", "2882", "2891", "2884", "2886", "1301", "1303", "2002", "2207", "2357",
)
HOLDING_DAYS = 20


@dataclass(frozen=True)
class TickerStats:
    trades: int
    win_rate: float | None
    avg_roi: float | None
    mdd: float


def network_router() -> ProviderRouter:
    return ProviderRouter([YFinanceProvider(), TwseDailyProvider()], cache=None)


def parse_csv_values(value: str, cast):
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def backtest_20_day_swing(data: pd.DataFrame, breakout_days: int, volume_multiplier: float) -> TickerStats:
    """Run a cash-or-long breakout strategy, with a maximum 20-trading-day settlement cycle."""
    frame = data.sort_values("Date").reset_index(drop=True)
    prior_high = frame["Close"].shift(1).rolling(breakout_days, min_periods=breakout_days).max()
    cash, shares, entry_value, held_days = 1.0, 0.0, 0.0, 0
    trade_returns: list[float] = []
    equity: list[float] = []
    for index, row in enumerate(frame.itertuples(index=False)):
        if shares:
            held_days += 1
            stop = pd.notna(row.SMA20) and (row.Close < row.SMA20 or row.Close < row.SMA60)
            settle = held_days >= HOLDING_DAYS or index == len(frame) - 1
            if stop or settle:
                cash = shares * row.Close
                trade_returns.append(cash / entry_value - 1)
                shares, entry_value, held_days = 0.0, 0.0, 0
        elif all(pd.notna(value) for value in (row.SMA20, row.SMA60, row.VMA20, prior_high.iloc[index])):
            trend_up = row.Close > row.SMA60 and row.SMA20 > row.SMA60
            breakout = row.Close > prior_high.iloc[index] and row.Volume > volume_multiplier * row.VMA20
            if trend_up and breakout:
                shares, entry_value, cash = cash / row.Close, cash, 0.0
        equity.append(cash if not shares else shares * row.Close)
    equity_series = pd.Series(equity, dtype=float)
    mdd = float(equity_series.div(equity_series.cummax()).sub(1).min())
    if not trade_returns:
        return TickerStats(0, None, None, mdd)
    returns = pd.Series(trade_returns)
    return TickerStats(len(returns), float((returns > 0).mean()), float(returns.mean()), mdd)


def fetch_universe(tickers: list[str], start: str, end: str | None) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Fetch once per ticker so grid parameters reuse identical OHLCV input."""
    actual_start = pd.Timestamp(start).normalize()
    warmup_start = (actual_start - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    router, universe, failures = network_router(), {}, []
    for ticker in tickers:
        request = PriceRequest(ticker, warmup_start, end)
        try:
            indicators = add_indicators(router.fetch(request))
            filtered = indicators.loc[indicators["Date"] >= actual_start].copy()
            if filtered.empty:
                raise ValueError("no data after requested start")
            universe[request.local_ticker] = filtered
        except Exception as exc:
            failures.append(f"{ticker}: {exc}")
            LOG.warning("Skipped %s", failures[-1])
    return universe, failures


def optimize(universe: dict[str, pd.DataFrame], breakout_days: list[int], multipliers: list[float]) -> pd.DataFrame:
    records: list[dict] = []
    for days, multiplier in product(breakout_days, multipliers):
        stats = [backtest_20_day_swing(data, days, multiplier) for data in universe.values()]
        traded = [item for item in stats if item.trades]
        profitable = [item for item in traded if item.avg_roi is not None and item.avg_roi > 0]
        records.append({
            "BreakoutDays": days, "VolumeMultiplier": multiplier, "HoldingDays": HOLDING_DAYS,
            "StocksScanned": len(stats), "StocksTraded": len(traded),
            "ProfitableShare": len(profitable) / len(traded) if traded else 0.0,
            "AvgROI": sum(item.avg_roi for item in traded if item.avg_roi is not None) / len(traded) if traded else float("nan"),
            "AvgWinRate": sum(item.win_rate for item in traded if item.win_rate is not None) / len(traded) if traded else float("nan"),
            "AvgMDD": sum(item.mdd for item in stats) / len(stats) if stats else float("nan"),
            "TotalTrades": sum(item.trades for item in stats),
        })
    ranking = pd.DataFrame(records)
    if ranking.empty:
        return ranking
    return ranking.sort_values(["ProfitableShare", "AvgROI", "AvgMDD", "AvgWinRate"], ascending=[False, False, False, False], na_position="last").reset_index(drop=True)


def print_ranking(ranking: pd.DataFrame) -> None:
    if ranking.empty:
        print("No successful ticker data was available for optimization.")
        return
    print("\n0050 Representative-Universe Grid Search (20 Trading-Day Settlement)")
    print(f"{'Rank':>4} {'Breakout':>10} {'Volume':>8} {'Profitable':>12} {'Avg ROI':>10} {'Avg MDD':>10} {'Avg Win':>10} {'Trades':>8}")
    for rank, row in enumerate(ranking.itertuples(index=False), start=1):
        print(f"{rank:>4} {row.BreakoutDays:>8} days {row.VolumeMultiplier:>8.2f} {row.ProfitableShare:>11.2%} {row.AvgROI:>10.2%} {row.AvgMDD:>10.2%} {row.AvgWinRate:>10.2%} {row.TotalTrades:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description="0050 representative-universe Swing grid optimizer")
    parser.add_argument("--start", default="2020-01-01", help="Backtest start date")
    parser.add_argument("--end", help="Backtest end date")
    parser.add_argument("--tickers", default=",".join(DEFAULT_0050_TICKERS), help="Comma-separated representative universe")
    parser.add_argument("--breakout-days", default="10,20,30", help="Comma-separated breakout lookbacks")
    parser.add_argument("--volume-multipliers", default="1.2,1.3,1.5", help="Comma-separated VMA20 multipliers")
    parser.add_argument("--output-csv", default="output/optimizer_0050_grid_summary.csv", help="CSV summary output")
    args = parser.parse_args()
    days, multipliers = parse_csv_values(args.breakout_days, int), parse_csv_values(args.volume_multipliers, float)
    if not days or min(days) < 2 or not multipliers or min(multipliers) <= 0:
        parser.error("breakout days must be >= 2 and volume multipliers must be positive")
    tickers = list(dict.fromkeys(value.strip().upper().split(".")[0] for value in args.tickers.split(",") if value.strip()))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    universe, failures = fetch_universe(tickers, args.start, args.end)
    ranking = optimize(universe, days, multipliers)
    print_ranking(ranking)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\nCSV summary: {output}")
    if failures:
        print(f"Skipped {len(failures)} ticker(s); see warning logs for details.")


if __name__ == "__main__":
    main()
