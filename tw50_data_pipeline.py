#!/usr/bin/env python3
"""
臺灣 50（0050）資料抓取、法人籌碼與技術指標管線。

安裝：
    python -m pip install -U pandas numpy yfinance requests

執行範例：
    python tw50_data_pipeline.py --period 1y --output-dir output
    python tw50_data_pipeline.py --period 2y --csv-only
    python tw50_data_pipeline.py --constituents-file my_constituents.csv
    python tw50_data_pipeline.py --constituents-file my_constituents.txt

輸出：
    output/tw50_prices_indicators.csv       每檔、每日 OHLCV 與技術指標
    output/tw50_latest_snapshot.csv         每檔最新一列，適合選股
    output/tw50_constituents.csv            本次實際使用的成分股清單
    output/tw50_data.sqlite                  上述資料的 SQLite 資料庫（除 --csv-only）

資料來源與限制：
* 成分股優先由 Yahoo Finance 的 0050.TW ETF 公開持股資料動態取得，接著嘗試元大
  投信的公開持股頁面。若兩個來源都無法解析出完整 50 檔，會使用程式內建的 0050
  成分股快照，確保後續流程不會中斷。快照會隨指數定期調整而過期，正式使用時仍應
  優先指定自己的清單。
* 可用 --constituents-file 指定 CSV（含 ticker 欄）或 TXT（每行一個代號）清單。
* OHLCV 與技術指標使用 Yahoo Finance 日線資料，非即時報價。
* 法人資料使用證交所 T86 盤後公開資料。該 API 公布買進、賣出、淨買賣「股數」，
  未公布外資/投信各自的成交金額；本程式以最近收盤價乘股數換算，欄位名稱皆以
  estimated_amount_twd 結尾，不應視為證交所官方成交金額。
* 本程式為資料工程範例，不構成投資建議。
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOG = logging.getLogger("tw50")
T = TypeVar("T")
TWSE_T86_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"
YUANTA_0050_HOLDINGS_URL = "https://www.yuantaetfs.com/product/detail/0050/ratio"
MIN_CONSTITUENT_COUNT = 50

# 0050 成分股快照（2026-10-30）。這不是即時的指數定義；僅在所有動態來源失敗時
# 使用，以避免研究管線因網站版型、限流或 API 回傳 top-10 而中斷。
DEFAULT_0050_TICKERS = [
    "2330", "2454", "2317", "2308", "3711", "1303", "2383", "2881", "2891", "3037",
    "2303", "2882", "2887", "3017", "2382", "2885", "2345", "2327", "2884", "2360",
    "2886", "2357", "2412", "2890", "3443", "2883", "2408", "6669", "3231", "3653",
    "2059", "2344", "3008", "2892", "2301", "2880", "1216", "4958", "6446", "2368",
    "2395", "5880", "8046", "2449", "2603", "3665", "7769", "4904", "3045", "6505",
]


def retry_call(attempts: int = 4, initial_wait: float = 1.5) -> Callable:
    """為 yfinance 等非 requests 呼叫加入指數退避重試。"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapped(*args, **kwargs) -> T:
            last_error: Exception | None = None
            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # 網路、限流與來源暫時異常都可重試
                    last_error = exc
                    if attempt == attempts - 1:
                        break
                    wait = initial_wait * (2 ** attempt)
                    LOG.warning("%s 失敗（第 %d/%d 次）：%s；%.1f 秒後重試",
                                func.__name__, attempt + 1, attempts, exc, wait)
                    time.sleep(wait)
            raise RuntimeError(f"{func.__name__} 在 {attempts} 次嘗試後仍失敗") from last_error
        return wrapped
    return decorator


def make_http_session() -> requests.Session:
    """建立會自動重試 429 與暫時性 5xx 錯誤的 HTTP 連線。"""
    retry = Retry(
        total=4, backoff_factor=1.2, status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",), raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "tw50-research-data-pipeline/1.0"})
    return session


def normalize_ticker(value: object) -> str | None:
    """把 2330、2330.TW、2330.TWO 統一成純數字台股代號。"""
    text = str(value).strip().upper().split(".")[0]
    return text if text.isdigit() and 4 <= len(text) <= 6 else None


def validate_constituents(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """正規化並確認來源確實提供完整 50 檔，而非只提供 top holdings。"""
    if "ticker" not in frame.columns:
        raise ValueError(f"{source} 沒有 ticker 欄位")
    result = frame.copy()
    result["ticker"] = result["ticker"].map(normalize_ticker)
    result = result.dropna(subset=["ticker"]).drop_duplicates("ticker").head(MIN_CONSTITUENT_COUNT)
    if len(result) < MIN_CONSTITUENT_COUNT:
        raise ValueError(f"{source} 僅解析出 {len(result)} 檔，需至少 {MIN_CONSTITUENT_COUNT} 檔")
    result["source"] = source
    return result.reset_index(drop=True)


@retry_call()
def get_constituents_from_0050() -> pd.DataFrame:
    """透過 yfinance 讀取 0050 的公開持股；僅接受完整 50 檔結果。"""
    holdings = yf.Ticker("0050.TW").funds_data.top_holdings.copy()
    if holdings.empty:
        raise ValueError("0050.TW 沒有回傳 holdings")

    # yfinance 版本間可能將標的代號放在 index 或欄位，兩者皆處理。
    frame = holdings.reset_index()
    ticker_col = next((c for c in frame.columns if str(c).lower() in {"symbol", "holding", "index"}), None)
    if ticker_col is None:
        ticker_col = frame.columns[0]
    name_col = next((c for c in frame.columns if "name" in str(c).lower()), None)
    weight_col = next((c for c in frame.columns if "weight" in str(c).lower()), None)
    result = pd.DataFrame({"ticker": frame[ticker_col].map(normalize_ticker)})
    result["name"] = frame[name_col] if name_col else pd.NA
    result["weight"] = frame[weight_col] if weight_col else pd.NA
    return validate_constituents(result, "Yahoo Finance 0050.TW funds_data.top_holdings")


def get_constituents_from_yuanta(session: requests.Session) -> pd.DataFrame:
    """網頁備援：從元大投信 0050 公開持股頁解析代號。"""
    response = session.get(YUANTA_0050_HOLDINGS_URL, timeout=30)
    response.raise_for_status()
    try:
        tables = pd.read_html(response.text)
    except ValueError as exc:
        raise ValueError("元大投信持股頁沒有可解析的表格") from exc
    candidates: list[object] = []
    for table in tables:
        candidates.extend(table.astype(str).to_numpy().ravel().tolist())
    return validate_constituents(pd.DataFrame({"ticker": candidates}), "Yuanta 0050 holdings page")


def get_default_constituents() -> pd.DataFrame:
    """回傳內建完整快照；最後一道防線，不讓資料管線中斷。"""
    return validate_constituents(
        pd.DataFrame({"ticker": DEFAULT_0050_TICKERS}),
        "built-in 0050 constituent snapshot (2026-10-30)",
    )


def read_constituents_file(file_path: str) -> pd.DataFrame:
    """讀取使用者 CSV（ticker 欄）或 TXT（每行一個代號）清單。"""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到成分股檔案：{file_path}")
    if path.suffix.lower() == ".txt":
        supplied = pd.DataFrame({"ticker": path.read_text(encoding="utf-8-sig").splitlines()})
    else:
        supplied = pd.read_csv(path, dtype=str)
        if "ticker" not in supplied.columns:
            raise ValueError("--constituents-file CSV 必須包含 ticker 欄位")
    return validate_constituents(supplied, f"user file: {file_path}")


def load_constituents(constituents_file: str | None) -> pd.DataFrame:
    """依序讀本機檔、yfinance、官方網頁備援，最後使用內建 50 檔快照。"""
    if constituents_file:
        return read_constituents_file(constituents_file)

    sources: list[tuple[str, Callable[[], pd.DataFrame]]] = [
        ("yfinance", get_constituents_from_0050),
        ("元大投信網頁備援", lambda: get_constituents_from_yuanta(make_http_session())),
    ]
    for source_name, fetch in sources:
        try:
            result = fetch()
            LOG.info("成分股取得成功：%s", source_name)
            return result
        except Exception as exc:
            LOG.warning("成分股來源 %s 失敗：%s；改試下一個來源", source_name, exc)
    LOG.warning("所有動態成分股來源皆失敗，改用內建 0050 完整快照")
    return get_default_constituents()


@retry_call()
def get_twse_institutional(session: requests.Session, trading_date: date) -> pd.DataFrame:
    """抓取 T86 三大法人資料，並只保留外資與投信欄位。"""
    params = {"date": trading_date.strftime("%Y%m%d"), "selectType": "ALL", "response": "json"}
    response = session.get(TWSE_T86_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", [])
    fields = payload.get("fields", [])
    if not rows or not fields:
        raise ValueError(f"{trading_date} 沒有 T86 資料（可能不是交易日）")
    raw = pd.DataFrame(rows, columns=fields)
    raw.columns = [str(c).replace(" ", "") for c in raw.columns]

    # 不依賴固定欄位順序，從中英文欄名辨識 T86 欄位。
    def find_col(*keyword_groups: tuple[str, ...]) -> str:
        """每個群組至少命中一個詞；兼容 T86 欄位名稱的小幅改版。"""
        for col in raw.columns:
            if all(any(word in col for word in group) for group in keyword_groups):
                return col
        raise KeyError(f"找不到 T86 欄位：{keyword_groups}；實際欄位={list(raw.columns)}")

    code_col = find_col(("證券代號",))
    name_col = find_col(("證券名稱",))
    # 現行 T86 常用「外資及陸資」，舊資料有時會寫成「外陸資」。
    foreign_buy = find_col(("外陸資", "外資及陸資"), ("買進",), ("股數",))
    foreign_sell = find_col(("外陸資", "外資及陸資"), ("賣出",), ("股數",))
    foreign_net = find_col(("外陸資", "外資及陸資"), ("買賣超",), ("股數",))
    trust_buy = find_col(("投信",), ("買進",), ("股數",))
    trust_sell = find_col(("投信",), ("賣出",), ("股數",))
    trust_net = find_col(("投信",), ("買賣超",), ("股數",))

    picked = raw[[code_col, name_col, foreign_buy, foreign_sell, foreign_net,
                  trust_buy, trust_sell, trust_net]].copy()
    picked.columns = ["ticker", "twse_name", "foreign_buy_shares", "foreign_sell_shares",
                      "foreign_net_shares", "trust_buy_shares", "trust_sell_shares",
                      "trust_net_shares"]
    picked["ticker"] = picked["ticker"].map(normalize_ticker)
    for col in picked.columns[2:]:
        picked[col] = pd.to_numeric(picked[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    picked["institution_date"] = pd.Timestamp(trading_date)
    return picked.dropna(subset=["ticker"])


def get_latest_institutional(session: requests.Session) -> pd.DataFrame:
    """週末、國定假日或盤後未更新時，往前尋找最近 10 個日曆日資料。"""
    for delta in range(10):
        candidate = date.today() - timedelta(days=delta)
        try:
            data = get_twse_institutional(session, candidate)
            LOG.info("法人資料日期：%s", candidate)
            return data
        except Exception as exc:
            LOG.info("%s 無法人資料：%s", candidate, exc)
    raise RuntimeError("往前 10 日仍無法取得 T86 法人資料")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算 SMA/EMA、Wilder RSI、KD(J) 與 MACD；輸入必須為單一股票日期序列。"""
    result = df.sort_values("date").copy()
    close, high, low = result["close"], result["high"], result["low"]
    for window in (5, 20, 60):
        result[f"sma_{window}"] = close.rolling(window, min_periods=window).mean()
        result[f"ema_{window}"] = close.ewm(span=window, adjust=False, min_periods=window).mean()

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    result["rsi_14"] = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))

    low_9 = low.rolling(9, min_periods=9).min()
    high_9 = high.rolling(9, min_periods=9).max()
    rsv = 100 * (close - low_9) / (high_9 - low_9).replace(0, np.nan)
    result["kd_k_9_3"] = rsv.ewm(alpha=1 / 3, adjust=False, min_periods=1).mean()
    result["kd_d_9_3"] = result["kd_k_9_3"].ewm(alpha=1 / 3, adjust=False, min_periods=1).mean()
    result["kd_j_9_3"] = 3 * result["kd_k_9_3"] - 2 * result["kd_d_9_3"]

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["macd_dif_12_26"] = ema_12 - ema_26
    result["macd_dea_9"] = result["macd_dif_12_26"].ewm(span=9, adjust=False, min_periods=9).mean()
    result["macd_hist"] = result["macd_dif_12_26"] - result["macd_dea_9"]
    return result


@retry_call()
def download_one_stock(ticker: str, period: str) -> pd.DataFrame:
    """下載單一標的，逐檔下載可避免 yfinance 多標的欄位 MultiIndex 的版本差異。"""
    history = yf.Ticker(f"{ticker}.TW").history(period=period, auto_adjust=False, actions=False)
    if history.empty:
        raise ValueError("Yahoo Finance 未回傳日線")
    history = history.reset_index()
    date_col = next(c for c in history.columns if str(c).lower() in {"date", "datetime"})
    result = history.rename(columns={date_col: "date", "Open": "open", "High": "high", "Low": "low",
                                     "Close": "close", "Volume": "volume"})
    result = result[["date", "open", "high", "low", "close", "volume"]]
    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None).dt.normalize()
    result.insert(0, "ticker", ticker)
    return add_indicators(result)


def download_all_stocks(tickers: list[str], period: str, workers: int) -> pd.DataFrame:
    """小量平行下載，個別失敗不會令整個批次中斷。"""
    complete: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one_stock, ticker, period): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                data = future.result()
                complete.append(data)
                LOG.info("%-6s 完成 %d 筆日線", ticker, len(data))
            except Exception as exc:
                LOG.error("%-6s 下載失敗：%s", ticker, exc)
    if not complete:
        raise RuntimeError("所有股票日線下載失敗，請檢查網路或稍後重試")
    return pd.concat(complete, ignore_index=True).sort_values(["ticker", "date"])


def merge_institutional(prices: pd.DataFrame, constituents: pd.DataFrame, institutional: pd.DataFrame) -> pd.DataFrame:
    """加入名稱、權重與最近一日法人資料；金額以最新收盤價估算。"""
    meta = constituents.drop(columns=["source"], errors="ignore")
    result = prices.merge(meta, on="ticker", how="left")
    result = result.merge(institutional, on="ticker", how="left")
    shares = ["foreign_buy_shares", "foreign_sell_shares", "foreign_net_shares",
              "trust_buy_shares", "trust_sell_shares", "trust_net_shares"]
    for col in shares:
        result[f"{col.replace('_shares', '_lots')}"] = result[col] / 1_000
        result[f"{col.replace('_shares', '_estimated_amount_twd')}"] = result[col] * result["close"]
    return result


def export_data(data: pd.DataFrame, constituents: pd.DataFrame, output_dir: Path, csv_only: bool) -> None:
    """輸出完整歷史、最新快照與可供回測讀取的 SQLite 表。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "tw50_prices_indicators.csv", index=False, encoding="utf-8-sig")
    constituents.to_csv(output_dir / "tw50_constituents.csv", index=False, encoding="utf-8-sig")
    latest = data.sort_values("date").groupby("ticker", as_index=False).tail(1)
    latest.to_csv(output_dir / "tw50_latest_snapshot.csv", index=False, encoding="utf-8-sig")
    if not csv_only:
        db_path = output_dir / "tw50_data.sqlite"
        with sqlite3.connect(db_path) as conn:
            data.to_sql("daily_prices_indicators", conn, if_exists="replace", index=False)
            latest.to_sql("latest_snapshot", conn, if_exists="replace", index=False)
            constituents.to_sql("constituents", conn, if_exists="replace", index=False)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_ticker_date ON daily_prices_indicators(ticker, date)")
    LOG.info("輸出完成：%s（%d 筆歷史資料、%d 檔股票）", output_dir, len(data), data["ticker"].nunique())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抓取臺灣50資料並計算技術指標")
    parser.add_argument("--period", default="1y", help="Yahoo Finance 期間，例如 6mo、1y、2y；預設 1y")
    parser.add_argument("--output-dir", default="output", help="輸出資料夾；預設 output")
    parser.add_argument("--constituents-file", help="覆蓋動態清單的 CSV（ticker 欄）或 TXT（每行一個代號）")
    parser.add_argument("--workers", type=int, default=3, help="同時下載數，預設 3（過大可能觸發限流）")
    parser.add_argument("--csv-only", action="store_true", help="不建立 SQLite，只輸出 CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.workers < 1 or args.workers > 8:
        raise ValueError("--workers 請設在 1 到 8，避免來源限流")
    started = datetime.now()
    constituents = load_constituents(args.constituents_file)
    LOG.info("取得 %d 檔成分股；來源：%s", len(constituents), constituents["source"].iloc[0])
    prices = download_all_stocks(constituents["ticker"].tolist(), args.period, args.workers)
    institutional = get_latest_institutional(make_http_session())
    merged = merge_institutional(prices, constituents, institutional)
    export_data(merged, constituents, Path(args.output_dir), args.csv_only)
    LOG.info("總耗時：%s", datetime.now() - started)


if __name__ == "__main__":
    main()
