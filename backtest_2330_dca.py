#!/usr/bin/env python3
"""台積電（2330）定期定額／季線 RSI 加碼回測。

範例：
    python3 backtest_2330_dca.py
    python3 backtest_2330_dca.py --csv output/tw50_prices_indicators.csv --amount 10000

假設：以收盤價成交、可買零股、不計手續費、交易稅、滑價與股利；僅作研究用途。
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"ticker", "date", "close", "sma_60", "rsi_14"}
MAX_SCHEDULE_OFFSET_DAYS = 7


def nearest_sixth_trading_day(month_data: pd.DataFrame) -> pd.Timestamp | None:
    """回傳該月距離 6 號最近的交易日；不完整月份則略過。"""
    first_date = month_data["date"].iloc[0]
    target = pd.Timestamp(year=first_date.year, month=first_date.month, day=6)
    distance = (month_data["date"] - target).abs()
    if distance.min() > pd.Timedelta(days=MAX_SCHEDULE_OFFSET_DAYS):
        return None
    candidates = month_data.loc[distance.eq(distance.min()), "date"]
    on_or_after = candidates[candidates.ge(target)]
    return on_or_after.min() if not on_or_after.empty else candidates.max()


def select_purchase_dates(stock: pd.DataFrame) -> set[pd.Timestamp]:
    """每一個有資料的月份選出一個固定扣款日。"""
    dates: set[pd.Timestamp] = set()
    for _, month_data in stock.groupby(stock["date"].dt.to_period("M"), sort=True):
        purchase_date = nearest_sixth_trading_day(month_data)
        if purchase_date is not None:
            dates.add(purchase_date)
    return dates


def run_backtest(stock: pd.DataFrame, base_amount: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """執行定期定額與 RSI／季線條件加碼，回傳每日資產、交易及績效。"""
    if base_amount <= 0:
        raise ValueError("每期扣款金額必須大於 0")

    daily = stock.copy().sort_values("date").reset_index(drop=True)
    purchase_dates = select_purchase_dates(daily)
    daily["buy_amount"] = 0.0
    daily["buy_shares"] = 0.0
    daily["smart_add"] = False

    for index, row in daily.iterrows():
        if row["date"] not in purchase_dates:
            continue
        smart_add = bool(
            pd.notna(row["sma_60"])
            and pd.notna(row["rsi_14"])
            and row["close"] < row["sma_60"]
            and row["rsi_14"] < 35
        )
        amount = base_amount * (2 if smart_add else 1)
        daily.loc[index, "buy_amount"] = amount
        daily.loc[index, "buy_shares"] = amount / row["close"]
        daily.loc[index, "smart_add"] = smart_add

    daily["cumulative_shares"] = daily["buy_shares"].cumsum()
    daily["invested_capital"] = daily["buy_amount"].cumsum()
    daily["asset_value"] = daily["cumulative_shares"] * daily["close"]

    invested = daily.loc[daily["invested_capital"] > 0, "asset_value"]
    drawdown = invested.div(invested.cummax()).sub(1)
    transactions = daily.loc[daily["buy_amount"] > 0].copy()
    final = daily.iloc[-1]
    metrics = {
        "total_invested": float(final["invested_capital"]),
        "cumulative_shares": float(final["cumulative_shares"]),
        "final_market_value": float(final["asset_value"]),
        "roi": float(final["asset_value"] / final["invested_capital"] - 1),
        "mdd": float(drawdown.min()),
    }
    return daily, transactions, metrics


def configure_matplotlib_chinese_font(plt) -> str | None:
    """自動選用已安裝的中文字型，並避免座標軸負號顯示成方塊。"""
    from matplotlib import font_manager

    preferred_fonts = [
        "Microsoft JhengHei", "Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei",
        "Noto Sans CJK TC", "Noto Sans CJK SC", "Source Han Sans TW", "PingFang TC",
        "Heiti TC", "Arial Unicode MS",
    ]
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next((font for font in preferred_fonts if font in installed_fonts), None)
    if selected_font:
        plt.rcParams["font.family"] = [selected_font]
    else:
        warnings.warn(
            "找不到常見中文字型；請安裝 Noto Sans CJK TC、Microsoft JhengHei 或 SimHei。",
            stacklevel=2,
        )
    plt.rcParams["axes.unicode_minus"] = False
    return selected_font


def plot_backtest(daily: pd.DataFrame, transactions: pd.DataFrame, metrics: dict[str, float], output_path: Path) -> None:
    """繪製主圖（股價、季線、買點）與副圖（市值、累積本金）。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "無法載入 matplotlib；請在專案環境執行 `python -m pip install -U -r requirements.txt`。"
        ) from exc
    configure_matplotlib_chinese_font(plt)

    fig, (price_axis, asset_axis) = plt.subplots(
        nrows=2, ncols=1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    fig.suptitle(
        "2330 定期定額與智慧加碼回測（含 ROI 與 MDD 數據）\n"
        f"ROI {metrics['roi']:.2%}｜MDD {metrics['mdd']:.2%}",
        fontsize=16,
        fontweight="bold",
    )

    price_axis.plot(daily["date"], daily["close"], color="#263238", linewidth=1.35, label="收盤價")
    price_axis.plot(daily["date"], daily["sma_60"], color="#7e57c2", linewidth=1.1, alpha=0.85, label="60 日均線（季線）")
    regular = transactions.loc[~transactions["smart_add"]]
    boosted = transactions.loc[transactions["smart_add"]]
    price_axis.scatter(regular["date"], regular["close"], color="#2e7d32", marker="^", s=62, zorder=3, label="固定扣款")
    price_axis.scatter(boosted["date"], boosted["close"], color="#d32f2f", marker="^", s=82, zorder=4, label="智慧加碼（2 倍）")
    price_axis.set_ylabel("2330 收盤價（TWD）")
    price_axis.grid(alpha=0.25)
    price_axis.legend(loc="upper left", ncols=2, frameon=True)

    asset_axis.plot(daily["date"], daily["asset_value"], color="#1565c0", linewidth=1.7, label="資產市值")
    asset_axis.plot(daily["date"], daily["invested_capital"], color="#757575", linestyle="--", linewidth=1.25, label="累積投入本金")
    asset_axis.fill_between(daily["date"], daily["invested_capital"], daily["asset_value"],
                            color="#42a5f5", alpha=0.12)
    asset_axis.set_xlabel("日期")
    asset_axis.set_ylabel("金額（TWD）")
    asset_axis.grid(alpha=0.25)
    asset_axis.legend(loc="upper left", frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2330 定期定額與智慧加碼回測")
    parser.add_argument("--csv", default="output/tw50_prices_indicators.csv", help="歷史資料 CSV 路徑")
    parser.add_argument("--ticker", default="2330", help="股票代號，預設 2330")
    parser.add_argument("--amount", type=float, default=10_000, help="每月基礎扣款金額，預設 10000")
    parser.add_argument("--output-dir", default="output/backtest", help="交易明細與圖表輸出資料夾")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.csv)
    raw = pd.read_csv(source, dtype={"ticker": str})
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要欄位：{', '.join(sorted(missing))}")

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["ticker"] = raw["ticker"].str.strip().str.split(".").str[0]
    stock = raw.loc[raw["ticker"].eq(str(args.ticker)), list(REQUIRED_COLUMNS)].copy()
    stock["close"] = pd.to_numeric(stock["close"], errors="coerce")
    stock["sma_60"] = pd.to_numeric(stock["sma_60"], errors="coerce")
    stock["rsi_14"] = pd.to_numeric(stock["rsi_14"], errors="coerce")
    stock = stock.dropna(subset=["date", "close"])
    if stock.empty:
        raise ValueError(f"CSV 找不到 {args.ticker} 的有效價格資料")

    daily, transactions, metrics = run_backtest(stock, args.amount)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(output_dir / f"{args.ticker}_dca_transactions.csv", index=False, encoding="utf-8-sig")
    plot_backtest(daily, transactions, metrics, output_dir / f"{args.ticker}_dca_backtest.png")

    print(f"股票：{args.ticker}；資料區間：{daily['date'].min():%Y-%m-%d} ～ {daily['date'].max():%Y-%m-%d}")
    print(f"扣款次數：{len(transactions)}（智慧加碼 {int(transactions['smart_add'].sum())} 次）")
    print(f"總投入本金：{metrics['total_invested']:,.0f} 元")
    print(f"累積股數：{metrics['cumulative_shares']:,.4f} 股")
    print(f"最終市值：{metrics['final_market_value']:,.0f} 元")
    print(f"總報酬率（ROI）：{metrics['roi']:.2%}")
    print(f"最大回檔（MDD）：{metrics['mdd']:.2%}")


if __name__ == "__main__":
    main()
