"""命令列進入點：組裝設定、Provider、指標、策略與輸出。"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from .backtest import BacktestConfig, BacktestResult, SwingConfig, run_dca, run_swing
from .charts import plot_backtest
from .indicators import add_indicators
from .providers import PriceRequest, build_default_router


NETWORK_PROVIDERS = {"yfinance", "twse"}
LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="台股定期定額與智慧抄底回測")
    parser.add_argument("--ticker", required=True, help="例如 2330、2317 或 2454.TW")
    parser.add_argument("--start", "--start-date", dest="start", default="2020-01-01", help="回測起日 YYYY-MM-DD")
    parser.add_argument("--end", "--end-date", dest="end", help="回測迄日 YYYY-MM-DD（包含該日）")
    parser.add_argument("--cache-dir", default=".cache/twbacktest", help="舊版快取資料夾；每次執行會清除該 ticker 快取")
    parser.add_argument("--providers", default="yfinance,twse", help="網路 Provider 順序：yfinance,twse")
    parser.add_argument("--monthly-amount", type=float, default=10_000, help="每月注資金額")
    parser.add_argument("--buy-fraction", type=float, default=0.8, help="訊號時動用現金比例")
    parser.add_argument("--rsi-threshold", type=float, default=35, help="RSI 抄底門檻")
    parser.add_argument("--deviation-threshold", type=float, default=-0.08, help="相對季線乖離率門檻")
    parser.add_argument("--swing-initial-cash", type=float, default=100_000, help="波段策略初始資金")
    parser.add_argument("--swing-position-fraction", type=float, default=1.0, help="每次進場動用可用現金比例")
    parser.add_argument("--swing-volume-multiplier", type=float, default=1.3, help="帶量突破所需 VMA20 倍數")
    parser.add_argument("--output-dir", default="output/modular_backtest", help="輸出資料夾")
    parser.add_argument("--no-chart", action="store_true", help="只執行回測，不產生 PNG")
    parser.add_argument("--no-open-chart", action="store_true", help="產生 PNG，但不自動用系統圖片程式開啟")
    return parser.parse_args()


def print_result(result: BacktestResult) -> None:
    print(f"\n【{result.name}】")
    print(f"總投入本金：{result.contributed:,.0f} 元")
    print(f"最終累積股數：{result.shares:,.4f} 股")
    print(f"最終現金餘額：{result.cash:,.0f} 元")
    print(f"最終資產總市值：{result.final_value:,.0f} 元")
    win_rate = f"{result.win_rate:.2%}" if result.win_rate is not None else "N/A（DCA 無平倉交易）"
    print(f"ROI：{result.roi:.2%}；MDD：{result.mdd:.2%}")
    print(f"交易次數：{result.trade_count}；勝率：{win_rate}")


def print_comparison(swing: BacktestResult, dca: BacktestResult) -> None:
    """以固定欄位列出兩個策略的核心績效，方便終端機或 CI 收集。"""
    print("\n【績效對照】")
    print(f"{'Strategy':<18} {'ROI':>10} {'MDD':>10} {'Win Rate':>12} {'Trades':>8} {'Final Value':>16}")
    for result in (swing, dca):
        win_rate = f"{result.win_rate:.2%}" if result.win_rate is not None else "N/A"
        print(
            f"{result.name:<18} {result.roi:>10.2%} {result.mdd:>10.2%} "
            f"{win_rate:>12} {result.trade_count:>8} {result.final_value:>16,.0f}"
        )


def filter_backtest_window(data: pd.DataFrame, request: PriceRequest) -> pd.DataFrame:
    """Apply the inclusive CLI date window after indicators have been calculated."""
    start_date = request.start_date
    end_date = request.end_date
    mask = data["Date"] >= start_date
    if end_date is not None:
        mask &= data["Date"] <= end_date
    filtered = data.loc[mask].copy()
    if filtered.empty:
        end_text = end_date.strftime("%Y-%m-%d") if end_date is not None else "latest"
        raise ValueError(f"No data available between {start_date:%Y-%m-%d} and {end_text}")
    return filtered


def clear_ticker_artifacts(ticker: str, output_dir: Path, cache_dir: Path) -> None:
    """Remove only generated files for one validated ticker before a fresh network run."""
    if re.fullmatch(r"[A-Z0-9_-]+", ticker) is None:
        raise ValueError("ticker may only contain letters, numbers, underscores, or hyphens")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob(f"{ticker}_*"):
        if path.is_file() or path.is_symlink():
            path.unlink()
            LOG.info("Removed previous output: %s", path)
    cache_file = cache_dir / f"{ticker}.csv"
    if cache_file.is_file() or cache_file.is_symlink():
        cache_file.unlink()
        LOG.info("Removed previous ticker cache: %s", cache_file)


def build_network_router(cache_dir: Path, provider_names: str):
    """Build a network-only router; local snapshots and cached prices are never read or written."""
    selected = [name.strip().lower() for name in provider_names.split(",") if name.strip()]
    if not selected or any(name not in NETWORK_PROVIDERS for name in selected):
        raise ValueError("網路模式僅允許 providers=yfinance,twse")
    router = build_default_router(None, cache_dir, selected)
    router.cache = None  # Do not persist a network response for a future run.
    return router


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    request = PriceRequest(args.ticker, args.start, args.end)
    output = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    clear_ticker_artifacts(request.local_ticker, output, cache_dir)
    router = build_network_router(cache_dir, args.providers)
    # Calculate indicators first, then use one explicit, inclusive window for backtesting and charts.
    raw_data = router.fetch(request)
    LOG.info(
        "Fetched range %s to %s for requested range %s to %s",
        raw_data["Date"].min().date(), raw_data["Date"].max().date(), request.start_date.date(),
        request.end_date.date() if request.end_date is not None else "latest",
    )
    data = filter_backtest_window(add_indicators(raw_data), request)
    dca_config = BacktestConfig(args.monthly_amount, args.buy_fraction, args.rsi_threshold, args.deviation_threshold)
    swing_config = SwingConfig(args.swing_initial_cash, args.swing_position_fraction, args.swing_volume_multiplier)
    swing, dca = run_swing(data, swing_config), run_dca(data, dca_config)
    # 圖內維持英文，避免 Linux 伺服器只有 DejaVu Sans 時出現中文字缺字。
    title = f"{request.local_ticker} Swing vs DCA Backtest"
    swing.trades.to_csv(output / f"{request.local_ticker}_swing_trades.csv", index=False, encoding="utf-8-sig")
    dca.trades.to_csv(output / f"{request.local_ticker}_dca_trades.csv", index=False, encoding="utf-8-sig")
    if not args.no_chart:
        plot_backtest(
            data, swing, dca, title, output / f"{request.local_ticker}_backtest.png",
            open_after_save=not args.no_open_chart, start_date=request.start_date, end_date=request.end_date,
        )
    print(f"資料來源：{data['Provider'].iloc[0]}；區間：{data['Date'].min():%Y-%m-%d} ～ {data['Date'].max():%Y-%m-%d}")
    print_result(swing)
    print_result(dca)
    print_comparison(swing, dca)
    print(f"\nSwing 相對 DCA ROI 差異：{swing.roi - dca.roi:+.2%}")


if __name__ == "__main__":
    main()
