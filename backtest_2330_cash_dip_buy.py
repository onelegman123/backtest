#!/usr/bin/env python3
"""台股現金累積與動態抄底策略，並與每月定期定額比較。

範例：
    python3 backtest_2330_cash_dip_buy.py --ticker 2317 --start 2020-01-01 --end 2025-12-31
    python3 backtest_2330_cash_dip_buy.py --ticker 2454.TW --start 2016-01-01

假設：以收盤價成交、可買零股，未計入手續費、交易稅、滑價與股利。
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TRADING_DAYS_AFTER_MONTH_START = 7


@dataclass
class StrategyResult:
    name: str
    daily: pd.DataFrame
    transactions: pd.DataFrame
    total_contributed: float
    final_shares: float
    final_cash: float
    final_value: float
    roi: float
    mdd: float


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """使用 Wilder 平滑法計算 RSI。"""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def normalize_ticker(ticker: str) -> tuple[str, str]:
    """回傳純數字／原始代號，以及供 yfinance 使用的完整代號。"""
    requested = ticker.strip().upper()
    if not requested:
        raise ValueError("股票代號不可為空白")
    local_code = requested.split(".")[0]
    yf_symbol = requested if "." in requested else f"{requested}.TW"
    return local_code, yf_symbol


def get_yfinance_name(yf_symbol: str) -> str:
    """盡力取得股票名稱；資料服務無名稱時保留空字串。"""
    try:
        import yfinance as yf

        info = yf.Ticker(yf_symbol).get_info()
        return str(info.get("longName") or info.get("shortName") or "").strip()
    except Exception:
        return ""


def load_price_data(csv_path: str | None, ticker: str, start: str, end: str | None) -> tuple[pd.DataFrame, str, str, str]:
    """優先從本機 CSV 篩選；找不到目標股票才以 yfinance 下載。"""
    local_code, yf_symbol = normalize_ticker(ticker)
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end) if end else None
    if end_date is not None and end_date < start_date:
        raise ValueError("--end 不可早於 --start")
    data = pd.DataFrame()
    stock_name = ""

    if csv_path and Path(csv_path).is_file():
        raw = pd.read_csv(csv_path, dtype={"ticker": str})
        required = {"date", "close", "ticker"}
        missing = required.difference(raw.columns)
        if missing:
            raise ValueError(f"CSV 缺少必要欄位：{', '.join(sorted(missing))}")
        raw["ticker"] = raw["ticker"].str.strip().str.upper().str.split(".").str[0]
        data = raw.loc[raw["ticker"].eq(local_code)].copy()
        if not data.empty and "twse_name" in data.columns:
            names = data["twse_name"].dropna().astype(str).str.strip()
            stock_name = next((name for name in names if name and name.lower() != "nan"), "")

    if data.empty:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("未安裝 yfinance；請執行 `python -m pip install -r requirements.txt`。") from exc
        # yfinance 的 end 為不包含該日；使用者輸入的 --end 則設計為包含該日。
        yf_end = (end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end_date else None
        downloaded = yf.download(
            yf_symbol, start=start_date.strftime("%Y-%m-%d"), end=yf_end,
            auto_adjust=True, progress=False, threads=False,
        )
        if downloaded.empty:
            raise ValueError(f"yfinance 未取得 {yf_symbol} 歷史資料")
        if isinstance(downloaded.columns, pd.MultiIndex):
            downloaded.columns = downloaded.columns.get_level_values(0)
        data = downloaded.reset_index().rename(columns={"Date": "date", "Close": "close"})
        stock_name = get_yfinance_name(yf_symbol)

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    data = data.loc[data["date"].ge(start_date)].copy()
    if end_date is not None:
        data = data.loc[data["date"].le(end_date)].copy()
    if data.empty:
        raise ValueError(f"指定期間內找不到 {local_code} 的有效收盤價資料")

    # 無論來源是否已有技術指標，一律由 close 重算，避免不同來源的計算定義不同。
    data["sma_60"] = data["close"].rolling(60, min_periods=60).mean()
    data["rsi_14"] = calculate_rsi(data["close"])
    data["deviation_60"] = data["close"] / data["sma_60"] - 1
    return data.reset_index(drop=True), local_code, yf_symbol, stock_name


def monthly_injection_dates(data: pd.DataFrame) -> set[pd.Timestamp]:
    """每月選擇 1 號起第一個交易日；略過資料從月中才開始的不完整月份。"""
    result: set[pd.Timestamp] = set()
    for _, month_data in data.groupby(data["date"].dt.to_period("M"), sort=True):
        first_date = month_data["date"].iloc[0]
        month_start = first_date.replace(day=1)
        if first_date - month_start <= pd.Timedelta(days=TRADING_DAYS_AFTER_MONTH_START):
            result.add(first_date)
    return result


def summarize_strategy(
    name: str,
    daily: pd.DataFrame,
    transactions: pd.DataFrame,
) -> StrategyResult:
    final = daily.iloc[-1]
    active_values = daily.loc[daily["contributed"] > 0, "portfolio_value"]
    drawdown = active_values.div(active_values.cummax()).sub(1)
    total_contributed = float(final["contributed"])
    final_value = float(final["portfolio_value"])
    return StrategyResult(
        name=name,
        daily=daily,
        transactions=transactions,
        total_contributed=total_contributed,
        final_shares=float(final["shares"]),
        final_cash=float(final["cash"]),
        final_value=final_value,
        roi=final_value / total_contributed - 1,
        mdd=float(drawdown.min()),
    )


def run_cash_dip_buy(data: pd.DataFrame, monthly_amount: float, buy_fraction: float) -> StrategyResult:
    """每月注資至現金池，在 RSI 或季線乖離觸發的新訊號時買入可用現金比例。"""
    daily = data.copy()
    injection_dates = monthly_injection_dates(daily)
    cash = shares = contributed = 0.0
    previous_signal = False
    records: list[dict[str, object]] = []
    transactions: list[dict[str, object]] = []

    for row in daily.itertuples(index=False):
        injected = monthly_amount if row.date in injection_dates else 0.0
        cash += injected
        contributed += injected
        rsi_signal = pd.notna(row.rsi_14) and row.rsi_14 < 35
        deviation_signal = pd.notna(row.deviation_60) and row.deviation_60 < -0.08
        signal = bool(rsi_signal or deviation_signal)
        is_new_signal = signal and not previous_signal
        buy_amount = cash * buy_fraction if is_new_signal else 0.0
        if buy_amount:
            shares += buy_amount / row.close
            cash -= buy_amount
            transactions.append({
                "date": row.date,
                "close": row.close,
                "rsi_14": row.rsi_14,
                "deviation_60": row.deviation_60,
                "buy_amount": buy_amount,
                "buy_shares": buy_amount / row.close,
                "rsi_signal": rsi_signal,
                "deviation_signal": deviation_signal,
            })
        previous_signal = signal
        records.append({
            "date": row.date,
            "close": row.close,
            "sma_60": row.sma_60,
            "rsi_14": row.rsi_14,
            "deviation_60": row.deviation_60,
            "injected": injected,
            "buy_amount": buy_amount,
            "shares": shares,
            "cash": cash,
            "contributed": contributed,
            "portfolio_value": shares * row.close + cash,
        })

    return summarize_strategy(
        "智慧抄底",
        pd.DataFrame(records),
        pd.DataFrame(transactions),
    )


def run_dca(data: pd.DataFrame, monthly_amount: float) -> StrategyResult:
    """同一注資日將固定金額全數買進，作為傳統 DCA 對照組。"""
    daily = data.copy()
    injection_dates = monthly_injection_dates(daily)
    shares = contributed = 0.0
    records: list[dict[str, object]] = []
    transactions: list[dict[str, object]] = []

    for row in daily.itertuples(index=False):
        buy_amount = monthly_amount if row.date in injection_dates else 0.0
        contributed += buy_amount
        buy_shares = buy_amount / row.close if buy_amount else 0.0
        shares += buy_shares
        if buy_amount:
            transactions.append({
                "date": row.date,
                "close": row.close,
                "buy_amount": buy_amount,
                "buy_shares": buy_shares,
            })
        records.append({
            "date": row.date,
            "close": row.close,
            "injected": buy_amount,
            "buy_amount": buy_amount,
            "shares": shares,
            "cash": 0.0,
            "contributed": contributed,
            "portfolio_value": shares * row.close,
        })

    return summarize_strategy(
        "傳統 DCA",
        pd.DataFrame(records),
        pd.DataFrame(transactions),
    )


def configure_chinese_font(plt) -> str | None:
    """自動選擇可用中文字型，並確保負號可正常顯示。"""
    from matplotlib import font_manager

    preferred = [
        "Microsoft JhengHei", "SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK TC",
        "Noto Sans CJK SC", "Microsoft YaHei", "Source Han Sans TW", "PingFang TC",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((font for font in preferred if font in installed), None)
    # 同時保留候選清單，以符合 Windows、Linux 與 macOS 的字型回退行為。
    plt.rcParams["font.sans-serif"] = ([selected] if selected else []) + preferred
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    if selected is None:
        warnings.warn("找不到中文字型；請安裝 Noto Sans CJK TC、Microsoft JhengHei 或 SimHei。")
    return selected


def plot_result(data: pd.DataFrame, smart: StrategyResult, display_name: str, output_path: Path) -> None:
    """上圖為股價／季線／紅色抄底買點，下圖為智慧策略現金池。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("無法載入 matplotlib；請執行 `python -m pip install -U -r requirements.txt`。") from exc

    configure_chinese_font(plt)
    fig, (price_ax, cash_ax) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]},
    )
    fig.suptitle(f"{display_name} 定期定額與智慧抄底回測", fontsize=16, fontweight="bold")

    price_ax.plot(data["date"], data["close"], color="#263238", linewidth=1.3, label=f"{display_name} 收盤價")
    price_ax.plot(data["date"], data["sma_60"], color="#7e57c2", linewidth=1.1, label="60 日均線（季線）")
    if not smart.transactions.empty:
        price_ax.scatter(
            smart.transactions["date"], smart.transactions["close"], color="#d32f2f",
            marker="v", s=88, zorder=5, label="智慧抄底買入點（紅色箭頭）",
        )
    price_ax.set_ylabel("股價（TWD）")
    price_ax.grid(alpha=0.25)
    price_ax.legend(loc="upper left", frameon=True)

    cash_ax.plot(smart.daily["date"], smart.daily["cash"], color="#1565c0", linewidth=1.6, label="智慧策略現金池餘額")
    cash_ax.fill_between(smart.daily["date"], 0, smart.daily["cash"], color="#42a5f5", alpha=0.16)
    cash_ax.set_xlabel("日期")
    cash_ax.set_ylabel("現金（TWD）")
    cash_ax.grid(alpha=0.25)
    cash_ax.legend(loc="upper left", frameon=True)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_metrics(result: StrategyResult) -> None:
    print(f"\n【{result.name}】")
    print(f"總投入本金：{result.total_contributed:,.0f} 元")
    print(f"最終累積股數：{result.final_shares:,.4f} 股")
    print(f"最終現金餘額：{result.final_cash:,.0f} 元")
    print(f"最終資產總市值：{result.final_value:,.0f} 元")
    print(f"總報酬率（ROI）：{result.roi:.2%}")
    print(f"最大回檔（MDD）：{result.mdd:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="台股定期定額與現金累積智慧抄底回測")
    parser.add_argument("--ticker", help="股票代號，例如 2317、2454 或 2454.TW；未填時改為互動式輸入")
    parser.add_argument("--csv", default="output/tw50_prices_indicators.csv", help="本機歷史資料 CSV；找不到目標代號時自動下載")
    parser.add_argument("--start", default="2016-01-01", help="回測起日 YYYY-MM-DD，預設 2016-01-01")
    parser.add_argument("--end", help="回測迄日 YYYY-MM-DD（包含該日），預設今天")
    parser.add_argument("--monthly-amount", type=float, default=10_000, help="每月注資金額，預設 10000")
    parser.add_argument("--buy-fraction", type=float, default=0.8, help="訊號觸發時動用現金池比例，預設 0.8")
    parser.add_argument("--output-dir", default="output/cash_dip_buy", help="輸出資料夾")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ticker = args.ticker or input("請輸入股票代號（例如 2317 或 2454.TW）：").strip()
    if args.monthly_amount <= 0:
        raise ValueError("--monthly-amount 必須大於 0")
    if not 0 < args.buy_fraction <= 1:
        raise ValueError("--buy-fraction 必須介於 0 與 1 之間")

    data, local_code, yf_symbol, stock_name = load_price_data(args.csv, ticker, args.start, args.end)
    display_name = " ".join(part for part in [local_code, stock_name] if part) or yf_symbol
    smart = run_cash_dip_buy(data, args.monthly_amount, args.buy_fraction)
    dca = run_dca(data, args.monthly_amount)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smart.transactions.to_csv(output_dir / f"{local_code}_smart_dip_transactions.csv", index=False, encoding="utf-8-sig")
    dca.transactions.to_csv(output_dir / f"{local_code}_dca_transactions.csv", index=False, encoding="utf-8-sig")
    plot_result(data, smart, display_name, output_dir / f"{local_code}_cash_dip_buy_backtest.png")

    print(f"股票：{display_name}（yfinance: {yf_symbol}）")
    print(f"資料區間：{data['date'].min():%Y-%m-%d} ～ {data['date'].max():%Y-%m-%d}")
    print_metrics(smart)
    print_metrics(dca)
    print(f"\n智慧抄底相對 DCA 的最終市值差額：{smart.final_value - dca.final_value:,.0f} 元")
    print(f"智慧抄底相對 DCA 的 ROI 差異：{smart.roi - dca.roi:+.2%}")


if __name__ == "__main__":
    main()
