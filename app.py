"""Streamlit Web UI for the twbacktest_candlestick toolkit.

Run locally with: streamlit run app.py
"""

from __future__ import annotations

import io
from dataclasses import asdict
from datetime import date, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle

from optimizer_0050 import DEFAULT_0050_TICKERS, fetch_universe, optimize
from twbacktest_candlestick.backtest import BacktestConfig, BacktestResult, SwingConfig, run_dca, run_swing
from twbacktest_candlestick.daily_scanner import load_watchlist, scan_today
from twbacktest_candlestick.indicators import add_indicators, sma
from twbacktest_candlestick.providers import PriceRequest, ProviderRouter, TwseDailyProvider, YFinanceProvider


st.set_page_config(page_title="Taiwan Stock Backtest", page_icon="📈", layout="wide")


def make_router(order: str) -> ProviderRouter:
    providers = {"yfinance": YFinanceProvider(), "twse": TwseDailyProvider()}
    return ProviderRouter([providers[name] for name in order.split(",")], cache=None)


def fetch_indicators(
    ticker: str, start: date, end: date, fast_period: int, slow_period: int, volume_period: int, provider_order: str,
) -> pd.DataFrame:
    """Fetch a warm-up period, then limit data to the selected visual backtest range."""
    warmup_start = (pd.Timestamp(start) - pd.Timedelta(days=max(150, slow_period * 3))).strftime("%Y-%m-%d")
    request = PriceRequest(ticker, warmup_start, end.isoformat())
    raw = make_router(provider_order).fetch(request)
    data = add_indicators(raw, sma_period=slow_period)
    # Reuse the existing SMA helper while allowing the UI to vary fast MA and volume MA.
    data["SMA20"] = sma(data["Close"], fast_period)
    data["VMA20"] = sma(data["Volume"], volume_period)
    data["SMA60"] = sma(data["Close"], slow_period)
    data["Deviation60"] = data["Close"] / data["SMA60"] - 1
    return data.loc[data["Date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()


def run_configurable_swing(
    data: pd.DataFrame, initial_cash: float, position_fraction: float, volume_multiplier: float,
    breakout_days: int,
) -> BacktestResult:
    """UI-only configurable form of the existing cash-or-long Swing strategy."""
    frame = data.sort_values("Date").reset_index(drop=True)
    prior_high = frame["Close"].shift(1).rolling(breakout_days, min_periods=breakout_days).max()
    cash, shares, entry_cost = initial_cash, 0.0, None
    closed, wins = 0, 0
    records: list[dict] = []
    trades: list[dict] = []
    for index, row in enumerate(frame.itertuples(index=False)):
        previous = frame.iloc[index - 1] if index else None
        ready = all(pd.notna(value) for value in (row.SMA20, row.SMA60, row.VMA20))
        trend_up = ready and row.Close > row.SMA60 and row.SMA20 > row.SMA60
        breakout = bool(
            trend_up and pd.notna(prior_high.iloc[index]) and row.Close > prior_high.iloc[index]
            and row.Volume > volume_multiplier * row.VMA20
        )
        bounce = bool(
            trend_up and previous is not None and pd.notna(previous["SMA20"])
            and previous["Close"] <= previous["SMA20"] and row.Close > row.SMA20
            and row.Close > previous["Close"] and row.Volume < row.VMA20
        )
        buy_amount = sell_amount = 0.0
        if shares and ready and (row.Close < row.SMA20 or row.Close < row.SMA60):
            sell_amount = shares * row.Close
            pnl = sell_amount - float(entry_cost)
            cash += sell_amount
            closed, wins = closed + 1, wins + int(pnl > 0)
            trades.append({"Date": row.Date, "Action": "Sell", "Close": row.Close, "Amount": sell_amount, "Shares": shares, "PnL": pnl, "Reason": "Close below SMA"})
            shares, entry_cost = 0.0, None
        elif not shares and (breakout or bounce):
            buy_amount = cash * position_fraction
            shares, cash, entry_cost = buy_amount / row.Close, cash - buy_amount, buy_amount
            trades.append({"Date": row.Date, "Action": "Buy", "Close": row.Close, "Amount": buy_amount, "Shares": shares, "PnL": None, "Reason": "Volume breakout" if breakout else "SMA pullback bounce"})
        records.append({"Date": row.Date, "Close": row.Close, "Cash": cash, "Shares": shares, "Contributed": initial_cash, "BuyAmount": buy_amount, "SellAmount": sell_amount, "PortfolioValue": cash + shares * row.Close})
    daily, trade_frame = pd.DataFrame(records), pd.DataFrame(trades)
    last = daily.iloc[-1]
    mdd = float(daily["PortfolioValue"].div(daily["PortfolioValue"].cummax()).sub(1).min())
    return BacktestResult(
        name="Swing Strategy", daily=daily, trades=trade_frame, contributed=initial_cash,
        shares=float(last["Shares"]), cash=float(last["Cash"]), final_value=float(last["PortfolioValue"]),
        roi=float(last["PortfolioValue"] / initial_cash - 1), mdd=mdd, trade_count=len(trade_frame),
        win_rate=wins / closed if closed else None,
    )


def build_figure(data: pd.DataFrame, swing: BacktestResult, dca: BacktestResult, fast_period: int, slow_period: int):
    """Create an in-memory candlestick comparison figure for ``st.pyplot``."""
    fig, (price_ax, value_ax) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    dates = mdates.date2num(pd.to_datetime(data["Date"]).dt.to_pydatetime())
    for number, row in zip(dates, data.itertuples(index=False)):
        color = "#d32f2f" if row.Close >= row.Open else "#2e7d32"
        price_ax.vlines(number, row.Low, row.High, color=color, linewidth=0.8, zorder=1)
        height = abs(row.Close - row.Open)
        if height:
            price_ax.add_patch(Rectangle((number - 0.31, min(row.Open, row.Close)), 0.62, height, facecolor=color, edgecolor=color, zorder=2))
        else:
            price_ax.hlines(row.Close, number - 0.31, number + 0.31, color=color, linewidth=1.2)
    price_ax.plot(data["Date"], data["SMA20"], color="#fb8c00", label=f"{fast_period}-Day SMA")
    price_ax.plot(data["Date"], data["SMA60"], color="#7e57c2", label=f"{slow_period}-Day SMA")
    if not swing.trades.empty:
        buys = swing.trades.loc[swing.trades["Action"].eq("Buy")]
        sells = swing.trades.loc[swing.trades["Action"].eq("Sell")]
        if not buys.empty:
            price_ax.scatter(buys["Date"], buys["Close"], marker="^", color="#2e7d32", s=72, label="Swing Buy", zorder=5)
        if not sells.empty:
            price_ax.scatter(sells["Date"], sells["Close"], marker="v", color="#c62828", s=72, label="Swing Sell", zorder=5)
    if not dca.trades.empty:
        price_ax.scatter(dca.trades["Date"], dca.trades["Close"], marker="s", color="#ef6c00", edgecolors="white", linewidths=0.7, s=48, label="DCA Buy", zorder=4)
    price_ax.set_title("Candlestick Chart", loc="left")
    price_ax.set_ylabel("Price (TWD)")
    price_ax.legend(loc="upper left")
    price_ax.grid(alpha=0.2)
    value_ax.plot(swing.daily["Date"], swing.daily["PortfolioValue"], color="#1565c0", label="Swing Strategy")
    value_ax.plot(dca.daily["Date"], dca.daily["PortfolioValue"], color="#ef6c00", label="DCA Strategy")
    value_ax.fill_between(swing.daily["Date"], 0, swing.daily["Cash"], color="#42a5f5", alpha=0.12, label="Swing Cash")
    value_ax.set_ylabel("Portfolio Value (TWD)")
    value_ax.set_xlabel("Date")
    value_ax.legend(loc="upper left")
    value_ax.grid(alpha=0.2)
    price_ax.set_xlim(data["Date"].min(), data["Date"].max())
    fig.tight_layout()
    return fig


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


st.title("📈 Taiwan Stock Strategy Workbench")
st.caption("Network-only data · Swing Trading vs. DCA · Signal scanning · 0050 parameter optimization")

with st.sidebar:
    st.header("Backtest Parameters")
    ticker = st.text_input("Ticker", value="2330").strip().upper()
    if st.checkbox("Choose from 0050 representative list"):
        ticker = st.selectbox("0050 ticker", DEFAULT_0050_TICKERS, index=0)
    default_end = date.today()
    selected_range = st.date_input("Backtest range", value=(default_end - timedelta(days=730), default_end))
    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
    else:
        start_date = end_date = selected_range
    monthly_amount = st.number_input("DCA monthly amount (TWD)", min_value=1_000.0, value=10_000.0, step=1_000.0)
    initial_cash = st.number_input("Swing initial cash (TWD)", min_value=1_000.0, value=100_000.0, step=10_000.0)
    position_fraction = st.slider("Swing position fraction", 0.1, 1.0, 1.0, 0.1)
    st.divider()
    st.subheader("Strategy Parameters")
    fast_period = st.number_input("Fast SMA days", min_value=5, max_value=100, value=20)
    slow_period = st.number_input("Slow SMA days", min_value=20, max_value=250, value=60)
    volume_period = st.number_input("Volume MA days", min_value=5, max_value=100, value=20)
    breakout_days = st.number_input("Breakout lookback days", min_value=5, max_value=100, value=20)
    volume_multiplier = st.slider("Volume multiplier", 1.0, 3.0, 1.3, 0.1)
    provider_order = st.selectbox("Network provider order", ("yfinance,twse", "twse,yfinance"))

tab_backtest, tab_signals, tab_optimizer = st.tabs(["Single Stock Backtest", "Today Signals", "0050 Optimizer"])

with tab_backtest:
    st.subheader("Swing Strategy vs. DCA")
    if st.button("Run backtest", type="primary", use_container_width=True):
        try:
            with st.spinner("Fetching market data and running backtest..."):
                data = fetch_indicators(ticker, start_date, end_date, int(fast_period), int(slow_period), int(volume_period), provider_order)
                if data.empty:
                    raise ValueError("No data is available in the selected date range.")
                # The default controls call the package's original engine directly.  The local
                # configurable variant is used only when a UI-specific MA/breakout value changes.
                if (fast_period, slow_period, volume_period, breakout_days) == (20, 60, 20, 20):
                    swing = run_swing(data, SwingConfig(initial_cash, position_fraction, volume_multiplier))
                else:
                    swing = run_configurable_swing(data, initial_cash, position_fraction, volume_multiplier, int(breakout_days))
                dca = run_dca(data, BacktestConfig(monthly_amount=monthly_amount))
                st.session_state["backtest"] = (data, swing, dca, int(fast_period), int(slow_period), ticker)
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
    if result := st.session_state.get("backtest"):
        data, swing, dca, fast, slow, result_ticker = result
        columns = st.columns(6)
        metrics = (("Swing ROI", swing.roi), ("Swing MDD", swing.mdd), ("Swing Value", swing.final_value), ("DCA ROI", dca.roi), ("DCA MDD", dca.mdd), ("DCA Value", dca.final_value))
        for column, (label, value) in zip(columns, metrics):
            column.metric(label, f"{value:.2%}" if "ROI" in label or "MDD" in label else f"{value:,.0f}")
        figure = build_figure(data, swing, dca, fast, slow)
        st.pyplot(figure, clear_figure=True, use_container_width=True)
        st.dataframe(swing.trades, use_container_width=True)
        st.download_button("Download Swing trades CSV", csv_bytes(swing.trades), f"{result_ticker}_swing_trades.csv", "text/csv")
        st.download_button("Download DCA trades CSV", csv_bytes(dca.trades), f"{result_ticker}_dca_trades.csv", "text/csv")

with tab_signals:
    st.subheader("Latest Candle Signal Scanner")
    watchlist_text = st.text_area("Watchlist (comma-separated)", value="2330,2383,2454,2317")
    scan_end = st.date_input("Latest date to scan", value=end_date, key="scan_end")
    scan_lookback = st.number_input("Signal scan lookback days", min_value=100, value=400, step=10)
    if st.button("Scan latest signals", use_container_width=True):
        try:
            with st.spinner("Fetching latest OHLCV data..."):
                signals, failures = scan_today(load_watchlist(watchlist_text), scan_end.isoformat(), int(scan_lookback), provider_order, volume_multiplier)
                signal_frame = pd.DataFrame([asdict(signal) for signal in signals])
                st.session_state["signals"] = (signal_frame, failures)
        except Exception as exc:
            st.error(f"Signal scan failed: {exc}")
    if scan_result := st.session_state.get("signals"):
        signal_frame, failures = scan_result
        for label, category in (("🚀 Buy Trigger", "buy"), ("🔻 Sell Trigger", "sell"), ("💤 Hold", "hold")):
            st.markdown(f"#### {label}")
            st.dataframe(signal_frame.loc[signal_frame["category"].eq(category)], use_container_width=True)
        if failures:
            st.warning(f"Skipped {len(failures)} ticker(s): " + " | ".join(failures[:3]))
        st.download_button("Download signal report CSV", csv_bytes(signal_frame), "today_signals.csv", "text/csv")

with tab_optimizer:
    st.subheader("0050 Representative-Universe Grid Search")
    optimizer_tickers = st.text_area("Optimization universe", value=",".join(DEFAULT_0050_TICKERS), key="optimizer_tickers")
    grid_left, grid_right = st.columns(2)
    with grid_left:
        grid_breakouts = st.multiselect("Breakout days", (10, 20, 30, 60), default=(10, 20, 30))
    with grid_right:
        grid_volumes = st.multiselect("Volume multipliers", (1.1, 1.2, 1.3, 1.5, 2.0), default=(1.2, 1.3, 1.5))
    if st.button("Run 0050 optimization", use_container_width=True):
        try:
            tickers = [value.strip().upper().split(".")[0] for value in optimizer_tickers.split(",") if value.strip()]
            if not tickers or not grid_breakouts or not grid_volumes:
                raise ValueError("Select at least one ticker, breakout day, and volume multiplier.")
            with st.spinner("Fetching data and evaluating grid combinations..."):
                universe, failures = fetch_universe(tickers, start_date.isoformat(), end_date.isoformat())
                ranking = optimize(universe, list(grid_breakouts), list(grid_volumes))
                st.session_state["optimizer"] = (ranking, failures)
        except Exception as exc:
            st.error(f"Optimization failed: {exc}")
    if optimizer_result := st.session_state.get("optimizer"):
        ranking, failures = optimizer_result
        st.dataframe(ranking, use_container_width=True)
        if not ranking.empty:
            st.success("Top row maximizes profitable-stock share first, then return/risk metrics.")
            st.download_button("Download optimizer CSV", csv_bytes(ranking), "0050_optimizer_ranking.csv", "text/csv")
        if failures:
            st.warning(f"Skipped {len(failures)} ticker(s): " + " | ".join(failures[:3]))
