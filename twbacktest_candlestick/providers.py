"""資料 Provider 與統一 OHLCV 資料契約。

所有 Provider 的 fetch() 都必須輸出下列欄位：
Date、Open、High、Low、Close、Volume、Ticker、Name、Provider。
這讓策略程式不需知道資料實際來自 CSV、Yahoo 或 TWSE。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .exceptions import AllProvidersFailedError, ProviderError


LOG = logging.getLogger(__name__)
STANDARD_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Name", "Provider"]


@dataclass(frozen=True)
class PriceRequest:
    """請求價格資料所需的最小設定。end 為包含該日。"""

    ticker: str
    start: str
    end: str | None = None

    @property
    def local_ticker(self) -> str:
        return self.ticker.strip().upper().split(".")[0]

    @property
    def yahoo_ticker(self) -> str:
        ticker = self.ticker.strip().upper()
        return ticker if "." in ticker else f"{ticker}.TW"

    @property
    def start_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.start).normalize()

    @property
    def end_date(self) -> pd.Timestamp | None:
        return pd.Timestamp(self.end).normalize() if self.end else None


def _to_number(series: pd.Series) -> pd.Series:
    """清理 TWSE 的逗號、-- 與正負號字串。"""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("--", "", regex=False),
        errors="coerce",
    )


def standardize_ohlcv(frame: pd.DataFrame, request: PriceRequest, provider: str, name: str = "") -> pd.DataFrame:
    """轉換常見中英文欄名為標準 OHLCV 契約並進行完整性驗證。"""
    if frame.empty:
        raise ProviderError(f"{provider} 回傳空資料")

    aliases = {
        "date": "Date", "日期": "Date", "open": "Open", "開盤價": "Open",
        "high": "High", "最高價": "High", "low": "Low", "最低價": "Low",
        "close": "Close", "收盤價": "Close", "volume": "Volume", "成交股數": "Volume",
        "成交量": "Volume", "ticker": "Ticker", "證券代號": "Ticker",
        "name": "Name", "twse_name": "Name", "證券名稱": "Name",
    }
    renamed = frame.rename(columns={column: aliases.get(str(column).strip().lower(), aliases.get(str(column).strip(), column)) for column in frame.columns}).copy()
    if "Date" not in renamed or "Close" not in renamed:
        raise ProviderError(f"{provider} 缺少 Date 或 Close 欄位；實際欄位={list(frame.columns)}")

    renamed["Date"] = pd.to_datetime(renamed["Date"], errors="coerce")
    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column not in renamed:
            renamed[column] = renamed["Close"] if column != "Volume" else 0
        renamed[column] = _to_number(renamed[column])
    renamed["Ticker"] = request.local_ticker
    renamed["Name"] = renamed.get("Name", pd.Series(name, index=renamed.index)).fillna(name).astype(str)
    renamed["Provider"] = provider
    result = renamed.loc[:, STANDARD_COLUMNS].dropna(subset=["Date", "Close"])
    result = result.loc[result["Date"].ge(request.start_date)]
    if request.end_date is not None:
        result = result.loc[result["Date"].le(request.end_date)]
    result = result.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    if result.empty:
        raise ProviderError(f"{provider} 在指定區間沒有 {request.local_ticker} 的有效資料")
    return result


class DataProvider(ABC):
    """可替換的資料提供者介面。新增來源只需實作此類別。"""

    name: str

    @abstractmethod
    def fetch(self, request: PriceRequest) -> pd.DataFrame:
        """回傳符合 standardize_ohlcv 契約的 DataFrame，失敗時拋 ProviderError。"""


class LocalCsvProvider(DataProvider):
    name = "local_csv"

    def __init__(self, csv_path: str | Path | None) -> None:
        self.csv_path = Path(csv_path) if csv_path else None

    def fetch(self, request: PriceRequest) -> pd.DataFrame:
        if self.csv_path is None or not self.csv_path.is_file():
            raise ProviderError("本機 CSV 不存在")
        try:
            raw = pd.read_csv(self.csv_path, dtype={"ticker": str})
        except (OSError, pd.errors.ParserError) as exc:
            raise ProviderError(f"無法讀取本機 CSV：{exc}") from exc
        ticker_column = next((c for c in raw.columns if str(c).lower() in {"ticker", "證券代號"}), None)
        if ticker_column is None:
            raise ProviderError("本機 CSV 缺少 ticker／證券代號欄位")
        codes = raw[ticker_column].astype(str).str.strip().str.upper().str.split(".").str[0]
        selected = raw.loc[codes.eq(request.local_ticker)].copy()
        if selected.empty:
            raise ProviderError(f"本機 CSV 沒有 {request.local_ticker}")
        return standardize_ohlcv(selected, request, self.name)


class YFinanceProvider(DataProvider):
    name = "yfinance"

    def fetch(self, request: PriceRequest) -> pd.DataFrame:
        try:
            import yfinance as yf

            # yfinance 的 end 不包含該日，轉成使用者較直覺的包含式區間。
            end = (request.end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if request.end_date else None
            raw = yf.download(
                request.yahoo_ticker, start=request.start_date.strftime("%Y-%m-%d"), end=end,
                auto_adjust=True, progress=False, threads=False,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.reset_index()
            stock_name = ""
            try:
                info = yf.Ticker(request.yahoo_ticker).get_info()
                stock_name = str(info.get("longName") or info.get("shortName") or "")
            except Exception:  # 名稱為附加資訊，不能使價格下載失敗。
                LOG.info("yfinance 無法取得 %s 的名稱", request.yahoo_ticker)
            return standardize_ohlcv(raw, request, self.name, stock_name)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"yfinance {request.yahoo_ticker} 下載失敗：{exc}") from exc


class TwseDailyProvider(DataProvider):
    """TWSE STOCK_DAY 月資料 API Provider；僅適用四碼上市股票。"""

    name = "twse"
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"

    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout

    @staticmethod
    def _parse_roc_date(value: object) -> pd.Timestamp | pd.NaT:
        text = str(value).strip()
        try:
            year, month, day = text.split("/")
            return pd.Timestamp(year=int(year) + 1911, month=int(month), day=int(day))
        except (TypeError, ValueError):
            return pd.NaT

    @staticmethod
    def _field(frame: pd.DataFrame, keyword: str) -> str:
        column = next((c for c in frame.columns if keyword in str(c)), None)
        if column is None:
            raise ProviderError(f"TWSE 回應缺少 {keyword} 欄位")
        return column

    def fetch(self, request: PriceRequest) -> pd.DataFrame:
        if not request.local_ticker.isdigit() or len(request.local_ticker) != 4:
            raise ProviderError("TWSE STOCK_DAY 僅支援四碼上市股票")
        end = request.end_date or pd.Timestamp.today().normalize()
        months = pd.period_range(request.start_date, end, freq="M")
        session = requests.Session()
        session.headers.update({"User-Agent": "twbacktest/1.0"})
        chunks: list[pd.DataFrame] = []
        errors: list[str] = []
        for month in months:
            params = {"response": "json", "date": month.start_time.strftime("%Y%m%d"), "stockNo": request.local_ticker}
            try:
                response = session.get(self.url, params=params, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                fields, rows = payload.get("fields", []), payload.get("data", [])
                if not fields or not rows:
                    errors.append(f"{month}: 無資料")
                    continue
                raw = pd.DataFrame(rows, columns=fields)
                normalized = pd.DataFrame({
                    "Date": raw[self._field(raw, "日期")].map(self._parse_roc_date),
                    "Open": raw[self._field(raw, "開盤價")],
                    "High": raw[self._field(raw, "最高價")],
                    "Low": raw[self._field(raw, "最低價")],
                    "Close": raw[self._field(raw, "收盤價")],
                    "Volume": raw[self._field(raw, "成交股數")],
                    # STOCK_DAY 的欄位通常沒有證券名稱；名稱缺失不應讓價格備援失效。
                    "Name": request.local_ticker,
                })
                chunks.append(normalized)
            except (requests.RequestException, ValueError, ProviderError) as exc:
                errors.append(f"{month}: {exc}")
        if not chunks:
            raise ProviderError(f"TWSE 無法取得資料；{'；'.join(errors[:3])}")
        return standardize_ohlcv(pd.concat(chunks, ignore_index=True), request, self.name)


class CacheProvider(DataProvider):
    """簡單 CSV 快取。快取會合併新資料，網路來源中斷時可作最後備援。"""

    name = "cache"

    def __init__(self, cache_dir: str | Path = ".cache/twbacktest") -> None:
        self.cache_dir = Path(cache_dir)

    def _path(self, request: PriceRequest) -> Path:
        return self.cache_dir / f"{request.local_ticker}.csv"

    def fetch(self, request: PriceRequest) -> pd.DataFrame:
        path = self._path(request)
        if not path.is_file():
            raise ProviderError("尚無本機快取")
        try:
            return standardize_ohlcv(pd.read_csv(path), request, self.name)
        except (OSError, pd.errors.ParserError, ProviderError) as exc:
            raise ProviderError(f"快取無法使用：{exc}") from exc

    def save(self, request: PriceRequest, data: pd.DataFrame) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._path(request)
            old = pd.read_csv(path) if path.exists() else pd.DataFrame()
            merged = data.copy() if old.empty else pd.concat([old, data], ignore_index=True)
            merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
            merged = merged.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates("Date", keep="last")
            merged.to_csv(path, index=False, encoding="utf-8-sig")
        except (OSError, pd.errors.ParserError) as exc:
            LOG.warning("寫入快取失敗：%s", exc)


class ProviderRouter:
    """依序嘗試 Provider；單一來源失敗不會中斷整個資料流程。"""

    def __init__(self, providers: Iterable[DataProvider], cache: CacheProvider | None = None) -> None:
        self.providers = list(providers)
        self.cache = cache

    def fetch(self, request: PriceRequest) -> pd.DataFrame:
        failures: list[str] = []
        for provider in self.providers:
            try:
                data = provider.fetch(request)
                # A non-empty response can still be only a recent partial history.  Accepting it
                # would silently make a chart ignore the requested start date.
                first_date = pd.Timestamp(data["Date"].min()).normalize()
                allowed_gap = pd.Timedelta(days=7)
                if first_date > request.start_date + allowed_gap:
                    raise ProviderError(
                        f"資料起日 {first_date:%Y-%m-%d} 未涵蓋要求起日 {request.start_date:%Y-%m-%d}"
                    )
                LOG.info("資料來源成功：%s（%d 筆）", provider.name, len(data))
                if self.cache is not None and provider.name != self.cache.name:
                    self.cache.save(request, data)
                return data
            except ProviderError as exc:
                failures.append(f"{provider.name}: {exc}")
                LOG.warning("資料來源失敗，切換備援：%s", failures[-1])
        raise AllProvidersFailedError("所有資料來源皆失敗：\n- " + "\n- ".join(failures))


def build_default_router(local_csv: str | Path | None, cache_dir: str | Path, provider_order: Iterable[str]) -> ProviderRouter:
    """依 CLI 指定順序組裝 Provider，未知名稱直接拒絕以避免設定拼字錯誤。"""
    cache = CacheProvider(cache_dir)
    candidates: dict[str, DataProvider] = {
        "local": LocalCsvProvider(local_csv), "yfinance": YFinanceProvider(),
        "twse": TwseDailyProvider(), "cache": cache,
    }
    providers: list[DataProvider] = []
    for name in provider_order:
        provider = candidates.get(name.strip().lower())
        if provider is None:
            raise ValueError(f"未知 Provider：{name}；可用值為 local,yfinance,twse,cache")
        providers.append(provider)
    return ProviderRouter(providers, cache)
