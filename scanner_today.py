#!/usr/bin/env python3
"""Standalone latest-candle Swing signal scanner.

Uses the existing twbacktest_candlestick Provider and indicator modules without
changing any project source files.  Data is fetched from yfinance/TWSE only.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from twbacktest_candlestick.indicators import add_indicators
from twbacktest_candlestick.providers import PriceRequest, ProviderRouter, TwseDailyProvider, YFinanceProvider


LOG = logging.getLogger(__name__)
DEFAULT_WATCHLIST = ("2330", "2383", "2454", "2317", "2308", "2382", "2327", "2303", "2412", "2881")


def parse_watchlist(value: str | None, file_path: str | None) -> list[str]:
    """Load a comma list or TXT/CSV watchlist; use the built-in list if neither is given."""
    if file_path:
        path = Path(file_path)
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, dtype=str)
            column = next((c for c in frame.columns if str(c).strip().lower() in {"ticker", "symbol", "code", "代號", "證券代號"}), frame.columns[0])
            value = ",".join(frame[column].dropna().astype(str))
        else:
            value = ",".join(re.split(r"[\s,;]+", path.read_text(encoding="utf-8-sig")))
    values = value.split(",") if value else DEFAULT_WATCHLIST
    return list(dict.fromkeys(item.strip().upper().split(".")[0] for item in values if item.strip()))


def network_router() -> ProviderRouter:
    """No local CSV and no persistent cache: every scan uses live network sources."""
    return ProviderRouter([YFinanceProvider(), TwseDailyProvider()], cache=None)


def classify_latest(ticker: str, data: pd.DataFrame, volume_multiplier: float) -> dict:
    """Evaluate the final available candle using the Swing entry/exit rules."""
    frame = data.sort_values("Date").reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("at least two trading days are required")
    latest, previous = frame.iloc[-1], frame.iloc[-2]
    required = ("SMA20", "SMA60", "VMA20")
    category, reason = "Hold", "Insufficient indicator history"
    if not any(pd.isna(latest[column]) for column in required):
        trend_up = latest["Close"] > latest["SMA60"] and latest["SMA20"] > latest["SMA60"]
        prior_high = frame["Close"].shift(1).rolling(20, min_periods=20).max().iloc[-1]
        breakout = bool(
            trend_up and pd.notna(prior_high) and latest["Close"] > prior_high
            and latest["Volume"] > volume_multiplier * latest["VMA20"]
        )
        bounce = bool(
            trend_up and pd.notna(previous["SMA20"]) and previous["Close"] <= previous["SMA20"]
            and latest["Close"] > latest["SMA20"] and latest["Close"] > previous["Close"]
            and latest["Volume"] < latest["VMA20"]
        )
        if latest["Close"] < latest["SMA20"] or latest["Close"] < latest["SMA60"]:
            category, reason = "Sell", "Close below SMA20/SMA60"
        elif breakout:
            category, reason = "Buy", "Volume breakout"
        elif bounce:
            category, reason = "Buy", "SMA20 pullback bounce"
        else:
            reason = "No Swing trigger"
    return {"Ticker": ticker, "Date": latest["Date"], "Close": latest["Close"], "Signal": category, "Reason": reason}


def scan(tickers: list[str], end_date: str, lookback_days: int, volume_multiplier: float) -> tuple[pd.DataFrame, list[str]]:
    start = (pd.Timestamp(end_date).normalize() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    router, records, failures = network_router(), [], []
    for ticker in tickers:
        request = PriceRequest(ticker, start, end_date)
        try:
            records.append(classify_latest(request.local_ticker, add_indicators(router.fetch(request)), volume_multiplier))
        except Exception as exc:
            failures.append(f"{request.local_ticker}: {exc}")
            LOG.warning("Skipped %s", failures[-1])
    return pd.DataFrame(records, columns=["Ticker", "Date", "Close", "Signal", "Reason"]), failures


def print_report(results: pd.DataFrame, failures: list[str]) -> None:
    for heading, category in (("🚀 【今日符合買進訊號的股票】", "Buy"), ("🔻 【今日符合賣出／出場訊號的股票】", "Sell"), ("💤 【無訊號觀望中】", "Hold")):
        print(f"\n{heading}")
        rows = results.loc[results["Signal"].eq(category)] if not results.empty else results
        if rows.empty:
            print("None")
            continue
        print(f"{'Ticker':<8} {'Date':<12} {'Close':>12}  Reason")
        for row in rows.itertuples(index=False):
            print(f"{row.Ticker:<8} {row.Date:%Y-%m-%d} {row.Close:>12,.2f}  {row.Reason}")
    if failures:
        print(f"\nSkipped {len(failures)} ticker(s); see warning logs for details.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Latest-candle Swing signal scanner")
    parser.add_argument("--watchlist", help="Comma-separated tickers")
    parser.add_argument("--watchlist-file", help="TXT or CSV ticker list")
    parser.add_argument("--output-csv", help="Optional CSV report path")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="Latest date to request")
    parser.add_argument("--lookback-days", type=int, default=400, help="Calendar days of history to request")
    parser.add_argument("--volume-multiplier", type=float, default=1.3, help="VMA20 breakout multiplier")
    args = parser.parse_args()
    if args.lookback_days < 100:
        parser.error("--lookback-days must be at least 100")
    if args.volume_multiplier <= 0:
        parser.error("--volume-multiplier must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    results, failures = scan(parse_watchlist(args.watchlist, args.watchlist_file), args.end_date, args.lookback_days, args.volume_multiplier)
    print_report(results, failures)
    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output, index=False, encoding="utf-8-sig")
        print(f"\nCSV report: {output}")


if __name__ == "__main__":
    main()
