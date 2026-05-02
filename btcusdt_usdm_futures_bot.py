from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import io
import json
import logging
import os
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import websockets


logger = logging.getLogger("btcusdt_usdm_futures")


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class RuntimeConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    testnet: bool = True
    recv_window_ms: int = 5_000
    dry_run: bool = True

    @property
    def rest_base_url(self) -> str:
        return "https://demo-fapi.binance.com" if self.testnet else "https://fapi.binance.com"

    @property
    def market_ws_url(self) -> str:
        symbol = self.symbol.lower()
        return (
            "wss://fstream.binance.com/market/stream?"
            f"streams={symbol}@kline_{self.interval}/{symbol}@markPrice@1s"
        )


@dataclass(frozen=True)
class StrategyConfig:
    ema_fast: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    volume_sma: int = 20
    breakout_lookback: int = 30
    pullback_lookback: int = 8
    min_volume_ratio: float = 1.10
    max_volume_ratio: float = 6.0
    min_atr_pct: float = 0.0015
    max_atr_pct: float = 0.06
    min_adx: float = 18.0
    min_breakout_atr: float = 0.10
    max_close_to_ema_fast_atr: float = 2.8
    min_ema_slow_slope_pct: float = 0.0002
    long_rsi_min: float = 52.0
    long_rsi_max: float = 68.0
    short_rsi_min: float = 28.0
    short_rsi_max: float = 48.0
    cooldown_bars: int = 6
    allow_short: bool = True


@dataclass(frozen=True)
class FuturesRiskConfig:
    initial_equity: float = 10_000.0
    leverage: int = 2
    risk_per_trade: float = 0.005
    max_notional_fraction: float = 0.75
    maintenance_margin_rate: float = 0.005
    liquidation_buffer_pct: float = 0.01
    stop_atr_multiple: float = 2.0
    take_profit_atr_multiple: float = 3.6
    trailing_stop: bool = True
    trailing_atr_multiple: float = 2.8
    breakeven_after_atr: float = 1.4
    max_bars_in_trade: int = 96
    taker_fee_rate: float = 0.0004
    slippage_rate: float = 0.0003


@dataclass(frozen=True)
class BacktestConfig:
    strategy: StrategyConfig = StrategyConfig()
    risk: FuturesRiskConfig = FuturesRiskConfig()


@dataclass(frozen=True)
class SymbolFilters:
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal


@dataclass
class Position:
    symbol: str
    side: int  # 1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float
    quantity: float
    stop_price: float
    take_profit_price: float
    atr_at_entry: float
    entry_fee: float
    margin_used: float
    highest_price: float
    lowest_price: float
    bars_held: int = 0

    @property
    def notional_at_entry(self) -> float:
        return self.entry_price * self.quantity


BINANCE_DATA_BASE_URL = "https://data.binance.vision/data"
REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


# =============================================================================
# Binance REST client
# =============================================================================


class BinanceFuturesClient:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.api_key = os.getenv("BINANCE_FUTURES_API_KEY", "")
        self.api_secret = os.getenv("BINANCE_FUTURES_API_SECRET", "")

    def public_get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        return self._request("GET", path, params=params, signed=False)

    def signed_get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        return self._request("GET", path, params=params, signed=True)

    def signed_post(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        return self._request("POST", path, params=params, signed=True)

    def user_stream_post(self, path: str) -> Any:
        return self._request("POST", path, params=None, signed=False, api_key_required=True)

    def user_stream_put(self, path: str, params: dict[str, Any]) -> Any:
        return self._request("PUT", path, params=params, signed=False, api_key_required=True)

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict[str, Any]],
        signed: bool,
        api_key_required: bool = False,
    ) -> Any:
        params = dict(params or {})
        headers = {"User-Agent": "btcusdt-usdm-futures-bot/1.0"}

        if signed or api_key_required:
            if not self.api_key:
                raise RuntimeError("Missing BINANCE_FUTURES_API_KEY")
            headers["X-MBX-APIKEY"] = self.api_key

        if signed:
            if not self.api_secret:
                raise RuntimeError("Missing BINANCE_FUTURES_API_SECRET")
            params.setdefault("recvWindow", self.config.recv_window_ms)
            params["timestamp"] = int(time.time() * 1000)
            query = urlencode(params)
            signature = hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
            params["signature"] = signature

        body = None
        query = urlencode(params)
        url = f"{self.config.rest_base_url}{path}"
        if method == "GET" and query:
            url = f"{url}?{query}"
        elif method in {"POST", "PUT", "DELETE"} and query:
            body = query.encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"

        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=30) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload) if payload else {}
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Binance HTTP {exc.code} {method} {path}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Network error calling Binance {method} {path}: {exc}") from exc

    def exchange_info(self) -> dict[str, Any]:
        return self.public_get("/fapi/v1/exchangeInfo")

    def mark_price(self, symbol: str) -> dict[str, Any]:
        return self.public_get("/fapi/v1/premiumIndex", {"symbol": symbol})

    def account(self) -> dict[str, Any]:
        return self.signed_get("/fapi/v2/account")

    def position_risk(self, symbol: str) -> list[dict[str, Any]]:
        return self.signed_get("/fapi/v3/positionRisk", {"symbol": symbol})

    def change_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        return self.signed_post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    def test_order(self, params: dict[str, Any]) -> Any:
        return self.signed_post("/fapi/v1/order/test", params)

    def new_order(self, params: dict[str, Any]) -> Any:
        if self.config.dry_run:
            raise RuntimeError("Refusing live order because dry_run=True")
        return self.signed_post("/fapi/v1/order", params)

    def start_listen_key(self) -> str:
        response = self.user_stream_post("/fapi/v1/listenKey")
        return str(response["listenKey"])

    def keepalive_listen_key(self, listen_key: str) -> Any:
        return self.user_stream_put("/fapi/v1/listenKey", {"listenKey": listen_key})


def extract_symbol_filters(exchange_info: dict[str, Any], symbol: str) -> SymbolFilters:
    symbol_info = next(item for item in exchange_info["symbols"] if item["symbol"] == symbol)
    filters = {item["filterType"]: item for item in symbol_info["filters"]}
    return SymbolFilters(
        tick_size=Decimal(filters["PRICE_FILTER"]["tickSize"]),
        step_size=Decimal(filters["LOT_SIZE"]["stepSize"]),
        min_qty=Decimal(filters["LOT_SIZE"]["minQty"]),
        min_notional=Decimal(filters["MIN_NOTIONAL"]["notional"]),
    )


def quantize_down(value: Decimal, step: Decimal) -> Decimal:
    if step == 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


# =============================================================================
# Historical futures data
# =============================================================================


def download_usdm_klines(
    symbol: str,
    interval: str,
    start_month: str,
    end_month: str,
    data_dir: Path,
) -> Path:
    symbol = symbol.upper()
    frames: list[pd.DataFrame] = []
    for month in pd.period_range(start=start_month, end=end_month, freq="M"):
        url = (
            f"{BINANCE_DATA_BASE_URL}/futures/um/monthly/klines/"
            f"{symbol}/{interval}/{symbol}-{interval}-{month.strftime('%Y-%m')}.zip"
        )
        try:
            logger.info("Downloading %s", url)
            frames.append(download_kline_zip(url))
        except HTTPError as exc:
            if exc.code == 404:
                logger.warning("Missing futures kline archive, skipping: %s", url)
                continue
            raise

    if not frames:
        raise RuntimeError(f"No USD-M futures klines downloaded for {symbol} {interval}")

    out = pd.concat(frames, ignore_index=True)[REQUIRED_COLUMNS]
    out["timestamp"] = parse_timestamp_column(out["timestamp"])
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")

    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{symbol}_USDM_{interval}.csv"
    out.to_csv(path, index=False)
    logger.info("Saved %s rows to %s", len(out), path)
    return path


def download_kline_zip(url: str) -> pd.DataFrame:
    request = Request(url, headers={"User-Agent": "btcusdt-usdm-futures-bot/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found in {url}")
        with archive.open(csv_names[0]) as csv_file:
            raw = pd.read_csv(csv_file, header=None)
    raw = raw.iloc[:, : len(BINANCE_KLINE_COLUMNS)].copy()
    raw.columns = BINANCE_KLINE_COLUMNS
    return raw.rename(columns={"open_time": "timestamp"})[REQUIRED_COLUMNS]


def download_funding_rates(
    client: BinanceFuturesClient,
    symbol: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    data_dir: Path,
) -> Path:
    symbol = symbol.upper()
    cursor_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    rows: list[dict[str, Any]] = []

    while cursor_ms <= end_ms:
        batch = client.public_get(
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": cursor_ms, "endTime": end_ms, "limit": 1000},
        )
        if not batch:
            break
        rows.extend(batch)
        last_time = int(batch[-1]["fundingTime"])
        cursor_ms = last_time + 1
        if len(batch) < 1000:
            break

    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{symbol}_funding.csv"
    if rows:
        df = pd.DataFrame(rows)
        df["fundingTime"] = pd.to_datetime(pd.to_numeric(df["fundingTime"]), unit="ms", utc=True)
        df = df.rename(columns={"fundingTime": "timestamp", "fundingRate": "funding_rate", "markPrice": "mark_price"})
        df[["timestamp", "funding_rate", "mark_price"]].to_csv(path, index=False)
    else:
        pd.DataFrame(columns=["timestamp", "funding_rate", "mark_price"]).to_csv(path, index=False)
    logger.info("Saved funding history to %s", path)
    return path


def parse_timestamp_column(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.9:
        parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
        microseconds = numeric >= 100_000_000_000_000
        milliseconds = (numeric >= 10_000_000_000) & ~microseconds
        seconds = numeric < 10_000_000_000
        parsed.loc[microseconds] = pd.to_datetime(numeric.loc[microseconds], unit="us", utc=True, errors="coerce")
        parsed.loc[milliseconds] = pd.to_datetime(numeric.loc[milliseconds], unit="ms", utc=True, errors="coerce")
        parsed.loc[seconds] = pd.to_datetime(numeric.loc[seconds], unit="s", utc=True, errors="coerce")
        return parsed
    return pd.to_datetime(series, utc=True, errors="coerce")


def load_ohlcv_csv(path: Path, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(col).strip().lower() for col in df.columns]
    missing = set(REQUIRED_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[REQUIRED_COLUMNS].copy()
    df["timestamp"] = parse_timestamp_column(df["timestamp"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["symbol"] = symbol.upper()
    df = df.dropna().sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    if df.empty:
        raise ValueError(f"{path} produced empty OHLCV data")
    return df


def load_funding_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["timestamp", "funding_rate", "mark_price"]).set_index("timestamp")
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "mark_price"]).set_index("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df["mark_price"] = pd.to_numeric(df.get("mark_price"), errors="coerce")
    return df.dropna(subset=["timestamp", "funding_rate"]).set_index("timestamp").sort_index()


def default_end_month() -> str:
    return (pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M") - 1).strftime("%Y-%m")


# =============================================================================
# Indicators and signals
# =============================================================================


def add_indicators(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=config.ema_fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=config.ema_slow, adjust=False).mean()
    out["ema_slow_slope_pct"] = out["ema_slow"].pct_change(config.pullback_lookback)
    out["rsi"] = rsi(out["close"], config.rsi_period)
    out["atr"] = atr(out, config.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["adx"] = adx(out, config.adx_period)
    out["volume_sma"] = out["volume"].rolling(config.volume_sma).mean()
    out["volume_ratio"] = out["volume"] / out["volume_sma"]
    out["close_to_ema_fast_atr"] = (out["close"] - out["ema_fast"]).abs() / out["atr"]
    out["prev_breakout_high"] = out["high"].rolling(config.breakout_lookback).max().shift(1)
    out["prev_breakout_low"] = out["low"].rolling(config.breakout_lookback).min().shift(1)
    out["breakout_distance_atr"] = (out["close"] - out["prev_breakout_high"]) / out["atr"]
    out["breakdown_distance_atr"] = (out["prev_breakout_low"] - out["close"]) / out["atr"]
    out["recent_pullback_long"] = out["low"].rolling(config.pullback_lookback).min().shift(1) <= out["ema_fast"].shift(1)
    out["recent_pullback_short"] = out["high"].rolling(config.pullback_lookback).max().shift(1) >= out["ema_fast"].shift(1)
    return out


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(100)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [df["high"] - df["low"], (df["high"] - previous_close).abs(), (df["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / tr.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean().replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / tr.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean().replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_signals(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    volume_ok = out["volume_ratio"].between(config.min_volume_ratio, config.max_volume_ratio)
    volatility_ok = out["atr_pct"].between(config.min_atr_pct, config.max_atr_pct)
    trend_strength_ok = out["adx"] >= config.min_adx
    not_overextended = out["close_to_ema_fast_atr"] <= config.max_close_to_ema_fast_atr

    long_setup = (
        (out["ema_fast"] > out["ema_slow"])
        & (out["close"] > out["ema_fast"])
        & (out["ema_slow_slope_pct"] > config.min_ema_slow_slope_pct)
        & out["rsi"].between(config.long_rsi_min, config.long_rsi_max)
        & (out["breakout_distance_atr"] >= config.min_breakout_atr)
        & out["recent_pullback_long"]
        & volume_ok
        & volatility_ok
        & trend_strength_ok
        & not_overextended
    )
    short_setup = (
        (out["ema_fast"] < out["ema_slow"])
        & (out["close"] < out["ema_fast"])
        & (out["ema_slow_slope_pct"] < -config.min_ema_slow_slope_pct)
        & out["rsi"].between(config.short_rsi_min, config.short_rsi_max)
        & (out["breakdown_distance_atr"] >= config.min_breakout_atr)
        & out["recent_pullback_short"]
        & volume_ok
        & volatility_ok
        & trend_strength_ok
        & not_overextended
    )

    out["raw_signal"] = 0
    out.loc[long_setup, "raw_signal"] = 1
    if config.allow_short:
        out.loc[short_setup, "raw_signal"] = -1
    out["entry_signal"] = out["raw_signal"].shift(1).fillna(0).astype(int)
    return out


# =============================================================================
# Futures backtester
# =============================================================================


def run_futures_backtest(
    df: pd.DataFrame,
    funding: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = add_signals(add_indicators(df, config.strategy), config.strategy)
    risk = config.risk
    equity = risk.initial_equity
    position: Optional[Position] = None
    cooldown_remaining = 0
    records: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    last_timestamp: Optional[pd.Timestamp] = None

    for timestamp, row in data.iterrows():
        if not is_tradeable_row(row):
            records.append(build_equity_record(timestamp, row, equity, position, 0.0))
            continue

        funding_pnl = apply_funding_between(funding, last_timestamp, timestamp, position, float(row["close"]))
        equity += funding_pnl
        last_timestamp = timestamp

        had_position = position is not None
        if position is not None:
            position.bars_held += 1
            update_position_extremes(position, row)
            exit_price, exit_reason = evaluate_exit(position, row, risk)
            if exit_price is None:
                exit_price, exit_reason = evaluate_bar_close_exit(position, row, risk)
            if exit_price is not None:
                equity, trade = close_position(position, timestamp, exit_price, exit_reason, equity, risk, funding_pnl)
                trades.append(trade)
                position = None
                cooldown_remaining = config.strategy.cooldown_bars
            else:
                update_trailing_stop(position, row, risk)

        can_enter = position is None and not had_position and cooldown_remaining <= 0
        if can_enter and int(row["entry_signal"]) != 0:
            position = open_futures_position(row, timestamp, equity, risk)
            if position is not None:
                equity -= position.entry_fee
        elif cooldown_remaining > 0:
            cooldown_remaining -= 1

        records.append(build_equity_record(timestamp, row, equity, position, funding_pnl))

    if position is not None:
        final_row = data.iloc[-1]
        equity, trade = close_position(position, data.index[-1], float(final_row["close"]), "end_of_data", equity, risk, 0.0)
        trades.append(trade)
        records[-1] = build_equity_record(data.index[-1], final_row, equity, None, 0.0)

    return pd.DataFrame(records).set_index("timestamp"), pd.DataFrame(trades)


def is_tradeable_row(row: pd.Series) -> bool:
    cols = ["open", "high", "low", "close", "atr", "ema_fast", "rsi", "entry_signal"]
    return bool(pd.notna(row[cols]).all() and float(row["atr"]) > 0)


def open_futures_position(row: pd.Series, timestamp: pd.Timestamp, equity: float, risk: FuturesRiskConfig) -> Optional[Position]:
    side = int(row["entry_signal"])
    atr_value = float(row["atr"])
    entry_price = apply_slippage(float(row["open"]), side, risk.slippage_rate)
    stop_distance = risk.stop_atr_multiple * atr_value
    if stop_distance <= 0:
        return None

    risk_capital = equity * risk.risk_per_trade
    qty_by_risk = risk_capital / stop_distance
    qty_by_notional_cap = (equity * risk.max_notional_fraction * risk.leverage) / entry_price
    quantity = max(0.0, min(qty_by_risk, qty_by_notional_cap))
    if quantity <= 0:
        return None

    notional = quantity * entry_price
    margin_used = notional / risk.leverage
    if margin_used > equity * risk.max_notional_fraction:
        return None

    if side == 1:
        stop_price = entry_price - stop_distance
        take_profit_price = entry_price + risk.take_profit_atr_multiple * atr_value
    else:
        stop_price = entry_price + stop_distance
        take_profit_price = entry_price - risk.take_profit_atr_multiple * atr_value

    liq_price = estimate_liquidation_price(entry_price, side, risk.leverage, risk.maintenance_margin_rate)
    if side == 1 and stop_price <= liq_price * (1 + risk.liquidation_buffer_pct):
        logger.info("Skipped long: stop too close to estimated liquidation price")
        return None
    if side == -1 and stop_price >= liq_price * (1 - risk.liquidation_buffer_pct):
        logger.info("Skipped short: stop too close to estimated liquidation price")
        return None

    entry_fee = notional * risk.taker_fee_rate
    return Position(
        symbol=str(row["symbol"]),
        side=side,
        entry_time=timestamp,
        entry_price=entry_price,
        quantity=quantity,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
        atr_at_entry=atr_value,
        entry_fee=entry_fee,
        margin_used=margin_used,
        highest_price=entry_price,
        lowest_price=entry_price,
    )


def estimate_liquidation_price(entry_price: float, side: int, leverage: int, maintenance_margin_rate: float) -> float:
    # Conservative approximation. Real Binance liquidation depends on brackets,
    # margin mode, wallet balance, maintenance amount and fees.
    if side == 1:
        return entry_price * (1 - (1 / leverage) + maintenance_margin_rate)
    return entry_price * (1 + (1 / leverage) - maintenance_margin_rate)


def apply_funding_between(
    funding: pd.DataFrame,
    previous_time: Optional[pd.Timestamp],
    current_time: pd.Timestamp,
    position: Optional[Position],
    mark_price: float,
) -> float:
    if position is None or funding.empty or previous_time is None:
        return 0.0
    window = funding[(funding.index > previous_time) & (funding.index <= current_time)]
    if window.empty:
        return 0.0
    pnl = 0.0
    for _, row in window.iterrows():
        notional = position.quantity * float(row.get("mark_price", mark_price) or mark_price)
        pnl += -position.side * notional * float(row["funding_rate"])
    return pnl


def update_position_extremes(position: Position, row: pd.Series) -> None:
    position.highest_price = max(position.highest_price, float(row["high"]))
    position.lowest_price = min(position.lowest_price, float(row["low"]))


def evaluate_exit(position: Position, row: pd.Series, risk: FuturesRiskConfig) -> tuple[Optional[float], Optional[str]]:
    high = float(row["high"])
    low = float(row["low"])
    if position.side == 1:
        if low <= position.stop_price:
            return apply_slippage(position.stop_price, -1, risk.slippage_rate), "stop_loss"
        if high >= position.take_profit_price:
            return apply_slippage(position.take_profit_price, -1, risk.slippage_rate), "take_profit"
    else:
        if high >= position.stop_price:
            return apply_slippage(position.stop_price, 1, risk.slippage_rate), "stop_loss"
        if low <= position.take_profit_price:
            return apply_slippage(position.take_profit_price, 1, risk.slippage_rate), "take_profit"
    return None, None


def evaluate_bar_close_exit(
    position: Position,
    row: pd.Series,
    risk: FuturesRiskConfig,
) -> tuple[Optional[float], Optional[str]]:
    close = float(row["close"])
    ema_fast = float(row["ema_fast"])
    rsi_value = float(row["rsi"])
    if position.bars_held >= risk.max_bars_in_trade:
        return apply_slippage(close, -position.side, risk.slippage_rate), "time_stop"
    if position.side == 1 and close < ema_fast and rsi_value < 48:
        return apply_slippage(close, -1, risk.slippage_rate), "trend_invalid"
    if position.side == -1 and close > ema_fast and rsi_value > 52:
        return apply_slippage(close, 1, risk.slippage_rate), "trend_invalid"
    return None, None


def update_trailing_stop(position: Position, row: pd.Series, risk: FuturesRiskConfig) -> None:
    if not risk.trailing_stop:
        return
    atr_value = float(row["atr"])
    if position.side == 1:
        candidate = float(row["close"]) - risk.trailing_atr_multiple * atr_value
        if position.highest_price >= position.entry_price + risk.breakeven_after_atr * position.atr_at_entry:
            candidate = max(candidate, position.entry_price)
        position.stop_price = max(position.stop_price, candidate)
    else:
        candidate = float(row["close"]) + risk.trailing_atr_multiple * atr_value
        if position.lowest_price <= position.entry_price - risk.breakeven_after_atr * position.atr_at_entry:
            candidate = min(candidate, position.entry_price)
        position.stop_price = min(position.stop_price, candidate)


def close_position(
    position: Position,
    timestamp: pd.Timestamp,
    exit_price: float,
    reason: str,
    equity: float,
    risk: FuturesRiskConfig,
    funding_pnl: float,
) -> tuple[float, dict[str, Any]]:
    gross_pnl = (exit_price - position.entry_price) * position.quantity * position.side
    exit_fee = abs(position.quantity * exit_price) * risk.taker_fee_rate
    net_pnl = gross_pnl - position.entry_fee - exit_fee
    equity += net_pnl
    trade = {
        "symbol": position.symbol,
        "side": "long" if position.side == 1 else "short",
        "entry_time": position.entry_time,
        "exit_time": timestamp,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "quantity": position.quantity,
        "notional": position.notional_at_entry,
        "margin_used": position.margin_used,
        "leverage": risk.leverage,
        "gross_pnl": gross_pnl,
        "fees": position.entry_fee + exit_fee,
        "last_bar_funding_pnl": funding_pnl,
        "net_pnl": net_pnl,
        "r_multiple": net_pnl / (position.atr_at_entry * position.quantity * risk.stop_atr_multiple),
        "bars_held": position.bars_held,
        "exit_reason": reason,
    }
    return equity, trade


def build_equity_record(timestamp: pd.Timestamp, row: pd.Series, equity: float, position: Optional[Position], funding_pnl: float) -> dict[str, Any]:
    unrealized = 0.0
    margin_used = 0.0
    if position is not None:
        mark = float(row["close"])
        unrealized = (mark - position.entry_price) * position.quantity * position.side
        margin_used = position.margin_used
    return {
        "timestamp": timestamp,
        "symbol": str(row["symbol"]),
        "equity": equity + unrealized,
        "realized_equity": equity,
        "unrealized_pnl": unrealized,
        "funding_pnl": funding_pnl,
        "margin_used": margin_used,
        "close": float(row["close"]),
        "in_position": int(position is not None),
    }


def apply_slippage(price: float, trade_direction: int, slippage_rate: float) -> float:
    return price * (1 + slippage_rate * trade_direction)


def calculate_metrics(equity: pd.DataFrame, trades: pd.DataFrame, initial_equity: float) -> dict[str, float]:
    final_equity = float(equity["equity"].iloc[-1])
    total_return = final_equity / initial_equity - 1
    returns = equity["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    dd = equity["equity"] / equity["equity"].cummax() - 1
    max_drawdown = abs(float(dd.min()))
    trade_count = int(len(trades))
    wins = trades.loc[trades["net_pnl"] > 0, "net_pnl"] if not trades.empty else pd.Series(dtype=float)
    losses = trades.loc[trades["net_pnl"] < 0, "net_pnl"] if not trades.empty else pd.Series(dtype=float)
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    win_rate = len(wins) / trade_count if trade_count else 0.0
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss if trade_count else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(len(returns))) if not returns.empty and returns.std(ddof=0) > 0 else 0.0
    return {
        "final_equity": final_equity,
        "total_return_pct": total_return * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "trade_count": float(trade_count),
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "win_rate_pct": win_rate * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "exposure_time_pct": float(equity["in_position"].mean() * 100),
        "sharpe_simplified": sharpe,
        "avg_trade_duration_bars": float(trades["bars_held"].mean()) if not trades.empty else 0.0,
    }


# =============================================================================
# Paper/testnet/live preparation
# =============================================================================


async def paper_stream(config: RuntimeConfig, strategy_config: StrategyConfig) -> None:
    logger.info("Starting paper stream for %s on %s", config.symbol, config.market_ws_url)
    candles: list[dict[str, Any]] = []
    latest_mark_price: Optional[float] = None

    while True:
        try:
            async with websockets.connect(config.market_ws_url, ping_interval=180, ping_timeout=600) as websocket:
                async for raw in websocket:
                    payload = json.loads(raw)
                    stream = payload.get("stream", "")
                    data = payload.get("data", payload)
                    if stream.endswith("@markPrice@1s") or data.get("e") == "markPriceUpdate":
                        latest_mark_price = float(data["p"])
                        continue
                    candle = parse_kline_stream(data)
                    if candle is None or not candle["is_closed"]:
                        continue
                    candles.append(candle)
                    candles = candles[-500:]
                    signal = evaluate_latest_signal(candles, config.symbol, strategy_config)
                    logger.info(
                        "closed_kline symbol=%s close=%s mark=%s signal=%s",
                        config.symbol,
                        candle["close"],
                        latest_mark_price,
                        signal,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("paper stream disconnected/error: %s", exc)
            await asyncio.sleep(5)


def parse_kline_stream(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    if data.get("e") != "kline":
        return None
    k = data["k"]
    return {
        "timestamp": pd.to_datetime(int(k["T"]), unit="ms", utc=True),
        "open": float(k["o"]),
        "high": float(k["h"]),
        "low": float(k["l"]),
        "close": float(k["c"]),
        "volume": float(k["v"]),
        "is_closed": bool(k["x"]),
    }


def evaluate_latest_signal(candles: list[dict[str, Any]], symbol: str, strategy_config: StrategyConfig) -> int:
    df = pd.DataFrame(candles)
    if len(df) < max(strategy_config.ema_slow, strategy_config.breakout_lookback) + 5:
        return 0
    df["symbol"] = symbol
    df = df.set_index("timestamp")
    enriched = add_signals(add_indicators(df, strategy_config), strategy_config)
    return int(enriched["entry_signal"].iloc[-1])


def build_market_order_params(symbol: str, side: int, quantity: Decimal, reduce_only: bool = False) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "side": "BUY" if side == 1 else "SELL",
        "type": "MARKET",
        "quantity": format(quantity, "f"),
        "reduceOnly": "true" if reduce_only else "false",
        "newClientOrderId": f"usdm-{int(time.time() * 1000)}",
    }


# =============================================================================
# CLI
# =============================================================================


def parse_symbol_paths(values: Iterable[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --csv value '{value}'. Use SYMBOL=path")
        symbol, path = value.split("=", 1)
        mapping[symbol.upper()] = Path(path)
    return mapping


def print_metrics(metrics: dict[str, float]) -> None:
    print("\n=== USD-M FUTURES BACKTEST SUMMARY ===")
    for key, value in metrics.items():
        if np.isinf(value):
            printable = "inf"
        elif key.endswith("_pct"):
            printable = f"{value:,.2f}%"
        else:
            printable = f"{value:,.4f}"
        print(f"{key:26s}: {printable}")


def run_backtest_mode(args: argparse.Namespace) -> None:
    runtime = RuntimeConfig(symbol=args.symbol, interval=args.interval, testnet=args.testnet)
    client = BinanceFuturesClient(runtime)
    data_dir = Path(args.data_dir)

    if args.download:
        kline_path = download_usdm_klines(args.symbol, args.interval, args.start, args.end or default_end_month(), data_dir)
        df_for_range = load_ohlcv_csv(kline_path, args.symbol)
        funding_path = download_funding_rates(client, args.symbol, df_for_range.index[0], df_for_range.index[-1], data_dir)
    else:
        if not args.csv:
            raise SystemExit("Use --download or provide --csv")
        kline_path = Path(args.csv)
        funding_path = Path(args.funding_csv) if args.funding_csv else None

    data = load_ohlcv_csv(kline_path, args.symbol)
    funding = load_funding_csv(funding_path)
    config = BacktestConfig(
        strategy=StrategyConfig(allow_short=not args.long_only),
        risk=FuturesRiskConfig(
            initial_equity=args.initial_equity,
            leverage=args.leverage,
            risk_per_trade=args.risk_per_trade,
            max_notional_fraction=args.max_notional_fraction,
        ),
    )
    equity, trades = run_futures_backtest(data, funding, config)
    metrics = calculate_metrics(equity, trades, config.risk.initial_equity)
    print_metrics(metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    equity.to_csv(output_dir / f"{args.symbol.lower()}_usdm_equity.csv")
    trades.to_csv(output_dir / f"{args.symbol.lower()}_usdm_trades.csv", index=False)
    if not trades.empty:
        print("\nExit reason breakdown:")
        print(trades.groupby("exit_reason").agg(trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"), avg_pnl=("net_pnl", "mean")).to_string())


def run_test_order_mode(args: argparse.Namespace) -> None:
    runtime = RuntimeConfig(symbol=args.symbol, interval=args.interval, testnet=args.testnet, dry_run=True)
    client = BinanceFuturesClient(runtime)
    filters = extract_symbol_filters(client.exchange_info(), args.symbol)
    mark = Decimal(client.mark_price(args.symbol)["markPrice"])
    notional = Decimal(str(args.test_order_notional))
    qty = quantize_down(notional / mark, filters.step_size)
    if qty < filters.min_qty or qty * mark < filters.min_notional:
        raise SystemExit(f"Quantity too small after filters: qty={qty}, notional={qty * mark}")
    response = client.test_order(build_market_order_params(args.symbol, 1, qty))
    print("Test order accepted by Binance Futures API.")
    print(response)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BTCUSDT Binance USD-M perpetual futures research/testnet bot")
    sub = parser.add_subparsers(dest="mode", required=True)

    backtest = sub.add_parser("backtest")
    backtest.add_argument("--symbol", default="BTCUSDT")
    backtest.add_argument("--interval", default="1h")
    backtest.add_argument("--download", action="store_true")
    backtest.add_argument("--start", default="2024-01")
    backtest.add_argument("--end", default=None)
    backtest.add_argument("--data-dir", default="data_usdm")
    backtest.add_argument("--csv", default=None)
    backtest.add_argument("--funding-csv", default=None)
    backtest.add_argument("--output-dir", default="usdm_backtest_output")
    backtest.add_argument("--initial-equity", type=float, default=10_000.0)
    backtest.add_argument("--leverage", type=int, default=2)
    backtest.add_argument("--risk-per-trade", type=float, default=0.005)
    backtest.add_argument("--max-notional-fraction", type=float, default=0.75)
    backtest.add_argument("--long-only", action="store_true")
    backtest.add_argument("--testnet", action="store_true")

    paper = sub.add_parser("paper-stream")
    paper.add_argument("--symbol", default="BTCUSDT")
    paper.add_argument("--interval", default="1h")
    paper.add_argument("--testnet", action="store_true")

    test_order = sub.add_parser("test-order")
    test_order.add_argument("--symbol", default="BTCUSDT")
    test_order.add_argument("--interval", default="1h")
    test_order.add_argument("--testnet", action="store_true", default=True)
    test_order.add_argument("--test-order-notional", type=float, default=25.0)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = build_arg_parser().parse_args()
    if args.mode == "backtest":
        run_backtest_mode(args)
    elif args.mode == "paper-stream":
        runtime = RuntimeConfig(symbol=args.symbol, interval=args.interval, testnet=args.testnet)
        asyncio.run(paper_stream(runtime, StrategyConfig()))
    elif args.mode == "test-order":
        run_test_order_mode(args)


if __name__ == "__main__":
    main()
