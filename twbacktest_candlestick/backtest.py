"""策略核心：不關心資料來源，只接受標準 OHLCV 與指標欄位。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    monthly_amount: float = 10_000
    buy_fraction: float = 0.8
    rsi_threshold: float = 35
    deviation_threshold: float = -0.08
    max_month_start_gap_days: int = 7


@dataclass(frozen=True)
class SwingConfig:
    """波段策略資金與訊號設定。初始資金不會在回測中追加。"""

    initial_cash: float = 100_000
    position_fraction: float = 1.0
    volume_multiplier: float = 1.3


@dataclass
class BacktestResult:
    name: str
    daily: pd.DataFrame
    trades: pd.DataFrame
    contributed: float
    shares: float
    cash: float
    final_value: float
    roi: float
    mdd: float
    trade_count: int
    win_rate: float | None


def _injection_dates(data: pd.DataFrame, config: BacktestConfig) -> set[pd.Timestamp]:
    dates: set[pd.Timestamp] = set()
    for _, month in data.groupby(data["Date"].dt.to_period("M"), sort=True):
        first = month["Date"].iloc[0]
        if first - first.replace(day=1) <= pd.Timedelta(days=config.max_month_start_gap_days):
            dates.add(first)
    return dates


def _result(
    name: str, records: list[dict], trades: list[dict], win_rate: float | None = None,
) -> BacktestResult:
    daily, transaction_frame = pd.DataFrame(records), pd.DataFrame(trades)
    last = daily.iloc[-1]
    active = daily.loc[daily["Contributed"] > 0, "PortfolioValue"]
    mdd = active.div(active.cummax()).sub(1).min()
    return BacktestResult(
        name=name, daily=daily, trades=transaction_frame, contributed=float(last["Contributed"]),
        shares=float(last["Shares"]), cash=float(last["Cash"]), final_value=float(last["PortfolioValue"]),
        roi=float(last["PortfolioValue"] / last["Contributed"] - 1), mdd=float(mdd),
        trade_count=len(transaction_frame), win_rate=win_rate,
    )


def run_smart_dip(data: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """每月累積現金；RSI 或乖離率新觸發時，買入現金池比例。"""
    if not 0 < config.buy_fraction <= 1 or config.monthly_amount <= 0:
        raise ValueError("monthly_amount 必須大於 0，buy_fraction 必須介於 0 與 1")
    required = {"Date", "Close", "RSI14", "Deviation60"}
    if missing := required.difference(data.columns):
        raise ValueError(f"策略缺少欄位：{sorted(missing)}")
    injections, cash, shares, contributed, previous_signal = _injection_dates(data, config), 0.0, 0.0, 0.0, False
    records: list[dict] = []
    trades: list[dict] = []
    for row in data.sort_values("Date").itertuples(index=False):
        injected = config.monthly_amount if row.Date in injections else 0.0
        cash, contributed = cash + injected, contributed + injected
        rsi_signal = pd.notna(row.RSI14) and row.RSI14 < config.rsi_threshold
        deviation_signal = pd.notna(row.Deviation60) and row.Deviation60 < config.deviation_threshold
        signal = bool(rsi_signal or deviation_signal)
        amount = cash * config.buy_fraction if signal and not previous_signal else 0.0
        if amount:
            bought = amount / row.Close
            shares, cash = shares + bought, cash - amount
            trades.append({"Date": row.Date, "Close": row.Close, "BuyAmount": amount, "Shares": bought, "RSI14": row.RSI14, "Deviation60": row.Deviation60})
        previous_signal = signal
        records.append({"Date": row.Date, "Close": row.Close, "Cash": cash, "Shares": shares, "Contributed": contributed, "BuyAmount": amount, "PortfolioValue": shares * row.Close + cash})
    return _result("智慧抄底", records, trades)


def run_dca(data: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """對照組：同樣月度注資日直接全額買入。"""
    injections, shares, contributed = _injection_dates(data, config), 0.0, 0.0
    records: list[dict] = []
    trades: list[dict] = []
    for row in data.sort_values("Date").itertuples(index=False):
        amount = config.monthly_amount if row.Date in injections else 0.0
        bought = amount / row.Close if amount else 0.0
        shares, contributed = shares + bought, contributed + amount
        if amount:
            trades.append({"Date": row.Date, "Close": row.Close, "BuyAmount": amount, "Shares": bought})
        records.append({"Date": row.Date, "Close": row.Close, "Cash": 0.0, "Shares": shares, "Contributed": contributed, "BuyAmount": amount, "PortfolioValue": shares * row.Close})
    return _result("傳統 DCA", records, trades)


def run_swing(data: pd.DataFrame, config: SwingConfig) -> BacktestResult:
    """以均線趨勢和量能進出場的全現金／持股波段策略。

    進場需處於多頭排列（收盤高於 SMA60、SMA20 高於 SMA60），再符合下列任一條件：
    - 收盤突破前 20 個交易日的最高收盤，且成交量高於 VMA20 的指定倍數；
    - 前一日回測 SMA20 以下，當日站回 SMA20、收盤上漲且量縮。
    持倉後，收盤跌破 SMA20 或 SMA60 即全數賣出。
    """
    if config.initial_cash <= 0:
        raise ValueError("initial_cash 必須大於 0")
    if not 0 < config.position_fraction <= 1:
        raise ValueError("position_fraction 必須介於 0 與 1")
    if config.volume_multiplier <= 0:
        raise ValueError("volume_multiplier 必須大於 0")
    required = {"Date", "Close", "Volume", "SMA20", "SMA60", "VMA20"}
    if missing := required.difference(data.columns):
        raise ValueError(f"策略缺少欄位：{sorted(missing)}")

    cash, shares = float(config.initial_cash), 0.0
    entry_cost: float | None = None
    wins, closed_trades = 0, 0
    records: list[dict] = []
    trades: list[dict] = []
    frame = data.sort_values("Date").reset_index(drop=True)
    prior_high = frame["Close"].shift(1).rolling(20, min_periods=20).max()

    for index, row in enumerate(frame.itertuples(index=False)):
        previous = frame.iloc[index - 1] if index else None
        indicators_ready = all(pd.notna(value) for value in (row.SMA20, row.SMA60, row.VMA20))
        trend_up = indicators_ready and row.Close > row.SMA60 and row.SMA20 > row.SMA60
        breakout = (
            trend_up and pd.notna(prior_high.iloc[index]) and row.Close > prior_high.iloc[index]
            and row.Volume > config.volume_multiplier * row.VMA20
        )
        pullback_bounce = bool(
            trend_up and previous is not None and pd.notna(previous["SMA20"])
            and previous["Close"] <= previous["SMA20"] and row.Close > row.SMA20
            and row.Close > previous["Close"] and row.Volume < row.VMA20
        )
        exit_signal = shares > 0 and indicators_ready and (row.Close < row.SMA20 or row.Close < row.SMA60)
        buy_amount = sell_amount = 0.0

        if exit_signal:
            sell_amount = shares * row.Close
            pnl = sell_amount - (entry_cost or 0.0)
            closed_trades += 1
            wins += pnl > 0
            cash += sell_amount
            trades.append({"Date": row.Date, "Action": "Sell", "Close": row.Close, "Amount": sell_amount, "Shares": shares, "PnL": pnl, "Reason": "Close below SMA20/SMA60"})
            shares, entry_cost = 0.0, None
        elif shares == 0 and (breakout or pullback_bounce):
            buy_amount = cash * config.position_fraction
            shares = buy_amount / row.Close
            cash -= buy_amount
            entry_cost = buy_amount
            reason = "Volume breakout" if breakout else "SMA20 pullback bounce"
            trades.append({"Date": row.Date, "Action": "Buy", "Close": row.Close, "Amount": buy_amount, "Shares": shares, "PnL": None, "Reason": reason})

        records.append({
            "Date": row.Date, "Close": row.Close, "Cash": cash, "Shares": shares,
            "Contributed": config.initial_cash, "BuyAmount": buy_amount, "SellAmount": sell_amount,
            "PortfolioValue": shares * row.Close + cash,
        })
    win_rate = wins / closed_trades if closed_trades else None
    return _result("Swing Strategy", records, trades, win_rate)
