"""Daily latest-candle signal scanner using the same rules as ``run_swing``."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .batch_scanner import DEFAULT_TICKERS, build_network_router, parse_tickers
from .indicators import add_indicators
from .providers import PriceRequest


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class DailySignal:
    ticker: str
    date: pd.Timestamp
    close: float
    category: str  # buy, sell, hold
    reason: str


def load_watchlist(inline_tickers: str | None = None, watchlist_file: str | Path | None = None) -> list[str]:
    """Load normalized tickers from a comma list, a text/CSV file, or the default watchlist."""
    if watchlist_file is None:
        return parse_tickers(inline_tickers) if inline_tickers else list(DEFAULT_TICKERS)
    path = Path(watchlist_file)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str)
        column = next((name for name in frame.columns if str(name).strip().lower() in {"ticker", "symbol", "code", "代號", "證券代號"}), frame.columns[0])
        return parse_tickers(",".join(frame[column].dropna().astype(str)))
    text = path.read_text(encoding="utf-8-sig")
    return parse_tickers(",".join(part for part in re.split(r"[\s,;]+", text) if part))


def evaluate_latest(ticker: str, data: pd.DataFrame, volume_multiplier: float = 1.3) -> DailySignal:
    """Classify the final candle with the same entry/exit conditions used by ``run_swing``."""
    frame = data.sort_values("Date").reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("at least two trading days are required")
    latest, previous = frame.iloc[-1], frame.iloc[-2]
    required = ("SMA20", "SMA60", "VMA20")
    if any(pd.isna(latest[column]) for column in required):
        return DailySignal(ticker, latest["Date"], float(latest["Close"]), "hold", "Insufficient indicator history")

    trend_up = latest["Close"] > latest["SMA60"] and latest["SMA20"] > latest["SMA60"]
    previous_high = frame["Close"].shift(1).rolling(20, min_periods=20).max().iloc[-1]
    breakout = bool(
        trend_up and pd.notna(previous_high) and latest["Close"] > previous_high
        and latest["Volume"] > volume_multiplier * latest["VMA20"]
    )
    pullback_bounce = bool(
        trend_up and pd.notna(previous["SMA20"]) and previous["Close"] <= previous["SMA20"]
        and latest["Close"] > latest["SMA20"] and latest["Close"] > previous["Close"]
        and latest["Volume"] < latest["VMA20"]
    )
    if latest["Close"] < latest["SMA20"] or latest["Close"] < latest["SMA60"]:
        return DailySignal(ticker, latest["Date"], float(latest["Close"]), "sell", "Close below SMA20/SMA60")
    if breakout:
        return DailySignal(ticker, latest["Date"], float(latest["Close"]), "buy", "Volume breakout")
    if pullback_bounce:
        return DailySignal(ticker, latest["Date"], float(latest["Close"]), "buy", "SMA20 pullback bounce")
    return DailySignal(ticker, latest["Date"], float(latest["Close"]), "hold", "No Swing trigger")


def scan_today(
    tickers: Iterable[str], end_date: str, lookback_days: int = 400, provider_names: str = "yfinance,twse",
    volume_multiplier: float = 1.3,
) -> tuple[list[DailySignal], list[str]]:
    """Fetch enough recent OHLCV history and classify each ticker's latest available candle."""
    if lookback_days < 100:
        raise ValueError("lookback_days must be at least 100 for SMA60 and VMA20")
    end = pd.Timestamp(end_date).normalize()
    start = (end - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    router = build_network_router(provider_names)
    signals: list[DailySignal] = []
    failures: list[str] = []
    for ticker in tickers:
        request = PriceRequest(ticker, start, end.strftime("%Y-%m-%d"))
        try:
            data = add_indicators(router.fetch(request))
            signals.append(evaluate_latest(request.local_ticker, data, volume_multiplier))
        except Exception as exc:  # A single failed fetch must not stop the daily report.
            message = f"{request.local_ticker}: {exc}"
            failures.append(message)
            LOG.warning("Skipped %s", message)
    return signals, failures


def print_daily_report(signals: Iterable[DailySignal], failures: Iterable[str] = ()) -> None:
    """Print grouped Buy/Sell/Hold results for an at-a-glance daily review."""
    groups = {"buy": [], "sell": [], "hold": []}
    for signal in signals:
        groups[signal.category].append(signal)
    sections = (
        ("🚀 【今日符合買進訊號的股票】", groups["buy"]),
        ("🔻 【今日符合賣出／出場訊號的股票】", groups["sell"]),
        ("💤 【無訊號觀望中】", groups["hold"]),
    )
    for heading, rows in sections:
        print(f"\n{heading}")
        if not rows:
            print("None")
            continue
        print(f"{'Ticker':<8} {'Date':<12} {'Close':>12}  Reason")
        for signal in rows:
            print(f"{signal.ticker:<8} {signal.date:%Y-%m-%d} {signal.close:>12,.2f}  {signal.reason}")
    failed = list(failures)
    if failed:
        print(f"\nSkipped {len(failed)} ticker(s); see warning logs for details.")
