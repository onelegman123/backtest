"""可選的 matplotlib 呈現層；回測核心不依賴繪圖。"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from .backtest import BacktestResult


def configure_chart_font(plt) -> None:
    """使用 matplotlib 內建 DejaVu Sans，避免伺服器缺少中文字型。"""
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def open_image_safely(output_path: Path) -> None:
    """依作業系統嘗試開圖；無 GUI 或失敗時不影響回測結果。"""
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(str(output_path))  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.Popen(["open", str(output_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif system == "Linux":
            if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                print("偵測到無圖形介面的 Linux 環境，略過自動開啟圖片。")
                return
            if shutil.which("xdg-open") is None:
                print("系統未安裝 xdg-open，略過自動開啟圖片。")
                return
            subprocess.Popen(["xdg-open", str(output_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print(f"不支援自動開圖的作業系統：{system}")
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"無法自動開啟圖表（不影響回測）：{exc}")


def _limit_to_chart_window(
    frame: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return only rows inside the inclusive window without changing the caller's frame."""
    dates = pd.to_datetime(frame["Date"])
    return frame.loc[dates.between(start_date, end_date)].copy()


def plot_backtest(
    data: pd.DataFrame, swing: BacktestResult, dca: BacktestResult, title: str,
    output_path: str | Path, open_after_save: bool = True,
    start_date: str | pd.Timestamp | None = None, end_date: str | pd.Timestamp | None = None,
) -> None:
    """上方呈現均線與波段買賣點；下方比較 Swing 和 DCA 資產曲線。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("未安裝 matplotlib，請執行 `python -m pip install -r requirements.txt`") from exc
    configure_chart_font(plt)
    chart_start = pd.to_datetime(start_date) if start_date is not None else pd.to_datetime(data["Date"]).min()
    chart_end = pd.to_datetime(end_date) if end_date is not None else pd.to_datetime(data["Date"]).max()
    if chart_start > chart_end:
        raise ValueError("start_date must not be after end_date")
    price_data = _limit_to_chart_window(data, chart_start, chart_end)
    swing_daily = _limit_to_chart_window(swing.daily, chart_start, chart_end)
    dca_daily = _limit_to_chart_window(dca.daily, chart_start, chart_end)
    if price_data.empty or swing_daily.empty or dca_daily.empty:
        raise ValueError("No chart data is available in the requested date range")

    fig, (price_ax, cash_ax) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    fig.suptitle(title, fontsize=16, fontweight="bold")
    price_ax.plot(price_data["Date"], price_data["Close"], color="#263238", label="Close Price")
    price_ax.plot(price_data["Date"], price_data["SMA20"], color="#fb8c00", label="20-Day SMA")
    price_ax.plot(price_data["Date"], price_data["SMA60"], color="#7e57c2", label="60-Day SMA")
    if not swing.trades.empty:
        chart_trades = _limit_to_chart_window(swing.trades, chart_start, chart_end)
        buys = chart_trades.loc[chart_trades["Action"].eq("Buy")]
        sells = chart_trades.loc[chart_trades["Action"].eq("Sell")]
        if not buys.empty:
            price_ax.scatter(buys["Date"], buys["Close"], color="#2e7d32", marker="^", s=80, zorder=5, label="Buy")
        if not sells.empty:
            price_ax.scatter(sells["Date"], sells["Close"], color="#c62828", marker="v", s=80, zorder=5, label="Sell")
    price_ax.set_ylabel("Price (TWD)")
    price_ax.grid(alpha=0.25)
    price_ax.legend(loc="upper left")
    cash_ax.plot(swing_daily["Date"], swing_daily["PortfolioValue"], color="#1565c0", label="Swing Strategy")
    cash_ax.plot(dca_daily["Date"], dca_daily["PortfolioValue"], color="#ef6c00", label="DCA Strategy")
    cash_ax.fill_between(swing_daily["Date"], 0, swing_daily["Cash"], color="#42a5f5", alpha=0.12, label="Swing Cash")
    cash_ax.set_xlabel("Date")
    cash_ax.set_ylabel("Portfolio Value (TWD)")
    cash_ax.grid(alpha=0.25)
    cash_ax.legend(loc="upper left")
    price_ax.set_xlim(chart_start, chart_end)
    cash_ax.set_xlim(chart_start, chart_end)
    resolved_path = Path(output_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(resolved_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"回測圖表已成功產出：{resolved_path}")
    if open_after_save:
        open_image_safely(resolved_path)
