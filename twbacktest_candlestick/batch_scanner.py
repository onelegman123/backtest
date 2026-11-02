"""Network-only batch scanner for ranking Swing Strategy results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from .backtest import SwingConfig, run_swing
from .indicators import add_indicators
from .providers import DataProvider, PriceRequest, ProviderRouter, TwseDailyProvider, YFinanceProvider


LOG = logging.getLogger(__name__)
DEFAULT_TICKERS = (
    "2330", "2317", "2454", "2308", "2303", "2382", "2327", "2383",
    "2412", "3034", "2881", "2882", "2891", "1301", "1303",
)
RANKING_COLUMNS = ["Rank", "Ticker", "ROI", "MDD", "WinRate", "Trades", "FinalValue"]


def parse_tickers(value: str | None) -> list[str]:
    """Return normalized unique tickers, falling back to a compact popular-stock pool."""
    source = value.split(",") if value else DEFAULT_TICKERS
    return list(dict.fromkeys(ticker.strip().upper().split(".")[0] for ticker in source if ticker.strip()))


def build_network_router(provider_names: str) -> ProviderRouter:
    """Create a router that never uses local CSV files or persistent price cache."""
    candidates: dict[str, DataProvider] = {"yfinance": YFinanceProvider(), "twse": TwseDailyProvider()}
    selected = [name.strip().lower() for name in provider_names.split(",") if name.strip()]
    if not selected or any(name not in candidates for name in selected):
        raise ValueError("Batch scan providers must be yfinance and/or twse")
    return ProviderRouter([candidates[name] for name in selected], cache=None)


def _filter_window(data: pd.DataFrame, request: PriceRequest) -> pd.DataFrame:
    mask = data["Date"] >= request.start_date
    if request.end_date is not None:
        mask &= data["Date"] <= request.end_date
    return data.loc[mask].copy()


def scan_tickers(
    tickers: Iterable[str], start: str, end: str | None, swing_config: SwingConfig,
    provider_names: str = "yfinance,twse",
) -> tuple[pd.DataFrame, list[str]]:
    """Backtest each ticker independently and return ROI-ranked records plus failures."""
    router = build_network_router(provider_names)
    records: list[dict] = []
    failures: list[str] = []
    for ticker in tickers:
        request = PriceRequest(ticker, start, end)
        try:
            data = _filter_window(add_indicators(router.fetch(request)), request)
            if data.empty:
                raise ValueError("no data in requested date range")
            result = run_swing(data, swing_config)
            records.append({
                "Ticker": request.local_ticker, "ROI": result.roi, "MDD": result.mdd,
                "WinRate": result.win_rate, "Trades": result.trade_count, "FinalValue": result.final_value,
            })
            LOG.info("Scanned %s: ROI %.2f%%", request.local_ticker, result.roi * 100)
        except Exception as exc:  # One unavailable symbol must not stop the scan.
            message = f"{request.local_ticker}: {exc}"
            failures.append(message)
            LOG.warning("Skipped %s", message)

    if not records:
        return pd.DataFrame(columns=RANKING_COLUMNS), failures
    ranking = pd.DataFrame(records).sort_values("ROI", ascending=False, kind="stable").reset_index(drop=True)
    ranking.insert(0, "Rank", ranking.index + 1)
    return ranking.loc[:, RANKING_COLUMNS], failures


def print_ranking(ranking: pd.DataFrame, top: int) -> None:
    """Print a dependency-free, fixed-width ranking table."""
    selected = ranking.head(top)
    if selected.empty:
        print("No stocks were successfully scanned.")
        return
    print(f"\nTop {len(selected)} Swing Strategy Ranking")
    print(f"{'Rank':>4} {'Ticker':<8} {'ROI':>10} {'MDD':>10} {'Win Rate':>10} {'Trades':>8} {'Final Value':>16}")
    for row in selected.itertuples(index=False):
        win_rate = f"{row.WinRate:.2%}" if pd.notna(row.WinRate) else "N/A"
        print(
            f"{row.Rank:>4} {row.Ticker:<8} {row.ROI:>10.2%} {row.MDD:>10.2%} "
            f"{win_rate:>10} {row.Trades:>8} {row.FinalValue:>16,.0f}"
        )


def save_ranking(ranking: pd.DataFrame, output_path: str | Path, top: int) -> Path:
    """Write only the requested top rows as a UTF-8 BOM CSV for spreadsheet compatibility."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranking.head(top).to_csv(path, index=False, encoding="utf-8-sig")
    return path
