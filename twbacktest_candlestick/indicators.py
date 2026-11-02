"""純 Pandas 技術指標，避免策略內散落重複計算邏輯。"""

from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, period: int) -> pd.Series:
    """簡單移動平均。"""
    return close.rolling(period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI。"""
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """回傳 MACD 線、訊號線與柱狀圖。"""
    macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"MACD": macd_line, "MACDSignal": signal_line, "MACDHist": macd_line - signal_line})


def add_indicators(data: pd.DataFrame, sma_period: int = 60, rsi_period: int = 14) -> pd.DataFrame:
    """不改動原始資料，回傳含趨勢、量能與動能指標的新 DataFrame。"""
    required = {"Close", "Volume"}
    if missing := required.difference(data.columns):
        raise ValueError(f"技術指標缺少欄位：{sorted(missing)}")
    result = data.copy()
    result["SMA20"] = sma(result["Close"], 20)
    result["SMA60"] = sma(result["Close"], sma_period)
    result["VMA20"] = sma(result["Volume"], 20)
    result["RSI14"] = rsi(result["Close"], rsi_period)
    result["Deviation60"] = result["Close"] / result["SMA60"] - 1
    return result.join(macd(result["Close"]))
