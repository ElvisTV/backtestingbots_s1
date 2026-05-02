from __future__ import annotations

import argparse
import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logger = logging.getLogger("crypto_backtester")


# =============================================================================
# Configuration
# =============================================================================


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
    allow_short: bool = False


@dataclass(frozen=True)
class RiskConfig:
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.01
    max_position_fraction: float = 0.35
    stop_atr_multiple: float = 2.0
    take_profit_atr_multiple: float = 3.6
    trailing_stop: bool = True
    trailing_atr_multiple: float = 2.8
    breakeven_after_atr: float = 1.4
    max_bars_in_trade: int = 96
    commission_rate: float = 0.0004
    slippage_rate: float = 0.0002


@dataclass(frozen=True)
class BacktestConfig:
    strategy: StrategyConfig = StrategyConfig()
    risk: RiskConfig = RiskConfig()


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
    highest_price: float
    lowest_price: float
    bars_held: int = 0


# =============================================================================
# Data loading and validation
# =============================================================================


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
BINANCE_DATA_BASE_URL = "https://data.binance.vision/data"


def load_ohlcv_csv(path: Path, symbol: str) -> pd.DataFrame:
    """Load Binance-style OHLCV data from CSV.

    Expected columns:
    timestamp, open, high, low, close, volume

    The timestamp can be milliseconds since epoch, seconds since epoch, or a
    parseable datetime string. This keeps the script friendly to both manual CSV
    exports and later Binance downloader code.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found for {symbol}: {path}")

    df = pd.read_csv(path)
    df.columns = [str(col).strip().lower() for col in df.columns]
    missing = set(REQUIRED_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df[REQUIRED_COLUMNS].copy()
    df["timestamp"] = parse_timestamp_column(df["timestamp"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["symbol"] = symbol.upper()
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    if df.empty:
        raise ValueError(f"{path} produced an empty dataset after cleaning")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{path} contains non-positive prices")
    if (df["volume"] < 0).any():
        raise ValueError(f"{path} contains negative volume")

    return df


def parse_timestamp_column(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.9:
        return parse_numeric_timestamp_column(numeric)
    return pd.to_datetime(series, utc=True, errors="coerce")


def parse_numeric_timestamp_column(series: pd.Series) -> pd.Series:
    """Infer epoch unit defensively.

    Binance archives have used millisecond timestamps historically, while some
    files can be exported with microsecond timestamps. Pandas defaults numeric
    datetimes to nanoseconds if no unit is given, which silently creates 1970
    dates. We test plausible epoch units and select the one whose median date is
    in the modern crypto era.
    """
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")

    # Mixed Binance archives exist: older monthly files may use milliseconds and
    # newer files microseconds. Unit inference by median is therefore not enough.
    microseconds = series >= 100_000_000_000_000
    milliseconds = (series >= 10_000_000_000) & ~microseconds
    seconds = series < 10_000_000_000

    parsed.loc[microseconds] = pd.to_datetime(series.loc[microseconds], unit="us", utc=True, errors="coerce")
    parsed.loc[milliseconds] = pd.to_datetime(series.loc[milliseconds], unit="ms", utc=True, errors="coerce")
    parsed.loc[seconds] = pd.to_datetime(series.loc[seconds], unit="s", utc=True, errors="coerce")
    return parsed


def download_binance_monthly_klines(
    symbol: str,
    interval: str,
    start_month: str,
    end_month: str,
    data_dir: Path,
    market: str = "spot",
) -> Path:
    """Download monthly Binance public kline ZIPs and save one normalized CSV.

    Binance's public archive stores klines as zipped CSV files without the
    exact column names expected by this backtester. This function downloads the
    monthly files, extracts them in memory, keeps only OHLCV, normalizes the
    timestamp column, and writes a clean CSV ready for load_ohlcv_csv().
    """
    symbol = symbol.upper()
    months = pd.period_range(start=start_month, end=end_month, freq="M")
    frames: list[pd.DataFrame] = []

    for month in months:
        url = build_binance_monthly_kline_url(symbol, interval, month, market)
        try:
            logger.info("Downloading %s", url)
            frames.append(download_single_binance_kline_zip(url))
        except HTTPError as exc:
            if exc.code == 404:
                logger.warning("Binance file not available, skipping: %s", url)
                continue
            raise
        except URLError as exc:
            raise RuntimeError(f"Network error downloading {url}: {exc}") from exc

    if not frames:
        raise RuntimeError(f"No Binance kline data downloaded for {symbol} {interval} {start_month}..{end_month}")

    normalized = pd.concat(frames, ignore_index=True)
    normalized = normalized[REQUIRED_COLUMNS]
    normalized["timestamp"] = parse_timestamp_column(normalized["timestamp"])
    normalized = normalized.dropna(subset=["timestamp"])
    normalized = normalized.sort_values("timestamp").drop_duplicates("timestamp")
    normalized["timestamp"] = normalized["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")

    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / f"{symbol}_{interval}.csv"
    normalized.to_csv(output_path, index=False)
    logger.info("Saved %s rows to %s", len(normalized), output_path)
    return output_path


def build_binance_monthly_kline_url(symbol: str, interval: str, month: pd.Period, market: str) -> str:
    return (
        f"{BINANCE_DATA_BASE_URL}/{market}/monthly/klines/"
        f"{symbol}/{interval}/{symbol}-{interval}-{month.strftime('%Y-%m')}.zip"
    )


def download_single_binance_kline_zip(url: str) -> pd.DataFrame:
    request = Request(url, headers={"User-Agent": "crypto-backtester/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file found inside {url}")
        with archive.open(csv_names[0]) as csv_file:
            raw = pd.read_csv(csv_file, header=None)

    if raw.shape[1] < len(BINANCE_KLINE_COLUMNS):
        raise ValueError(f"Unexpected Binance kline format in {url}: {raw.shape[1]} columns")

    raw = raw.iloc[:, : len(BINANCE_KLINE_COLUMNS)].copy()
    raw.columns = BINANCE_KLINE_COLUMNS
    return raw.rename(columns={"open_time": "timestamp"})[REQUIRED_COLUMNS]


def default_end_month() -> str:
    # Monthly archive files are normally complete only after the month closes.
    previous_month = pd.Timestamp.now(tz="UTC").to_period("M") - 1
    return previous_month.strftime("%Y-%m")


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

    # Shifted levels are critical: today's signal may use the previous completed
    # bars, but not the current high/low before the bar has closed.
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
    value = 100 - (100 / (1 + rs))
    return value.fillna(100)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    smoothed_tr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100 * smoothed_plus_dm / smoothed_tr.replace(0, np.nan)
    minus_di = 100 * smoothed_minus_dm / smoothed_tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_signals(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    volume_ok = out["volume_ratio"].between(config.min_volume_ratio, config.max_volume_ratio)
    volatility_ok = out["atr_pct"].between(config.min_atr_pct, config.max_atr_pct)
    trend_strength_ok = out["adx"] >= config.min_adx
    not_overextended = out["close_to_ema_fast_atr"] <= config.max_close_to_ema_fast_atr

    long_trend = (
        (out["ema_fast"] > out["ema_slow"])
        & (out["close"] > out["ema_fast"])
        & (out["ema_slow_slope_pct"] > config.min_ema_slow_slope_pct)
    )
    long_momentum = out["rsi"].between(config.long_rsi_min, config.long_rsi_max)
    long_breakout = out["breakout_distance_atr"] >= config.min_breakout_atr

    short_trend = (
        (out["ema_fast"] < out["ema_slow"])
        & (out["close"] < out["ema_fast"])
        & (out["ema_slow_slope_pct"] < -config.min_ema_slow_slope_pct)
    )
    short_momentum = out["rsi"].between(config.short_rsi_min, config.short_rsi_max)
    short_breakout = out["breakdown_distance_atr"] >= config.min_breakout_atr

    out["raw_signal"] = 0
    long_setup = (
        long_trend
        & long_momentum
        & volume_ok
        & volatility_ok
        & trend_strength_ok
        & not_overextended
        & long_breakout
        & out["recent_pullback_long"]
    )
    out.loc[long_setup, "raw_signal"] = 1
    if config.allow_short:
        short_setup = (
            short_trend
            & short_momentum
            & volume_ok
            & volatility_ok
            & trend_strength_ok
            & not_overextended
            & short_breakout
            & out["recent_pullback_short"]
        )
        out.loc[short_setup, "raw_signal"] = -1

    # Execute on next bar open to avoid using the same close that produced the signal.
    out["entry_signal"] = out["raw_signal"].shift(1).fillna(0).astype(int)
    return out


# =============================================================================
# Backtesting engine
# =============================================================================


def run_backtest(df: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = add_signals(add_indicators(df, config.strategy), config.strategy)
    risk = config.risk

    cash = risk.initial_capital
    position: Optional[Position] = None
    cooldown_remaining = 0
    equity_records: list[dict] = []
    trades: list[dict] = []

    for timestamp, row in data.iterrows():
        if not is_tradeable_row(row):
            equity_records.append(build_equity_record(timestamp, row, cash, position))
            continue

        had_position_at_bar_open = position is not None
        if position is not None:
            position.bars_held += 1
            update_position_extremes(position, row)
            exit_price, exit_reason = evaluate_exit(position, row, risk)
            if exit_price is None:
                exit_price, exit_reason = evaluate_bar_close_exit(position, row, risk)

            if exit_price is not None:
                cash, trade = close_position(position, timestamp, exit_price, exit_reason, cash, risk)
                trades.append(trade)
                position = None
                cooldown_remaining = config.strategy.cooldown_bars
            else:
                update_trailing_stop(position, row, risk)

        can_enter = position is None and not had_position_at_bar_open and cooldown_remaining <= 0
        if can_enter and int(row["entry_signal"]) != 0:
            position = open_position(row, timestamp, cash, risk)
            if position is not None:
                cash -= position.entry_fee
        elif cooldown_remaining > 0:
            cooldown_remaining -= 1

        equity_records.append(build_equity_record(timestamp, row, cash, position))

    if position is not None:
        last_timestamp = data.index[-1]
        last_close = float(data.iloc[-1]["close"])
        cash, trade = close_position(position, last_timestamp, last_close, "end_of_data", cash, risk)
        trades.append(trade)
        equity_records[-1] = build_equity_record(last_timestamp, data.iloc[-1], cash, None)

    equity = pd.DataFrame(equity_records).set_index("timestamp")
    trades_df = pd.DataFrame(trades)
    return equity, trades_df


def is_tradeable_row(row: pd.Series) -> bool:
    required = ["open", "high", "low", "close", "atr", "ema_fast", "rsi", "entry_signal"]
    return bool(pd.notna(row[required]).all() and float(row["atr"]) > 0)


def open_position(row: pd.Series, timestamp: pd.Timestamp, cash: float, risk: RiskConfig) -> Optional[Position]:
    side = int(row["entry_signal"])
    atr_value = float(row["atr"])
    open_price = float(row["open"])
    symbol = str(row["symbol"])

    entry_price = apply_slippage(open_price, side, risk.slippage_rate)
    stop_distance = risk.stop_atr_multiple * atr_value
    if stop_distance <= 0:
        return None

    risk_capital = cash * risk.risk_per_trade
    quantity_by_risk = risk_capital / stop_distance
    quantity_by_cap = (cash * risk.max_position_fraction) / entry_price
    quantity = max(0.0, min(quantity_by_risk, quantity_by_cap))
    if quantity <= 0:
        return None

    if side == 1:
        stop_price = entry_price - stop_distance
        take_profit_price = entry_price + risk.take_profit_atr_multiple * atr_value
    else:
        stop_price = entry_price + stop_distance
        take_profit_price = entry_price - risk.take_profit_atr_multiple * atr_value

    entry_fee = abs(quantity * entry_price) * risk.commission_rate
    return Position(
        symbol=symbol,
        side=side,
        entry_time=timestamp,
        entry_price=entry_price,
        quantity=quantity,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
        atr_at_entry=atr_value,
        entry_fee=entry_fee,
        highest_price=entry_price,
        lowest_price=entry_price,
    )


def update_position_extremes(position: Position, row: pd.Series) -> None:
    position.highest_price = max(position.highest_price, float(row["high"]))
    position.lowest_price = min(position.lowest_price, float(row["low"]))


def update_trailing_stop(position: Position, row: pd.Series, risk: RiskConfig) -> None:
    if not risk.trailing_stop:
        return

    atr_value = float(row["atr"])
    if position.side == 1:
        trail_candidate = float(row["close"]) - risk.trailing_atr_multiple * atr_value
        if position.highest_price >= position.entry_price + risk.breakeven_after_atr * position.atr_at_entry:
            breakeven_candidate = position.entry_price
            trail_candidate = max(trail_candidate, breakeven_candidate)
        position.stop_price = max(position.stop_price, trail_candidate)
    else:
        trail_candidate = float(row["close"]) + risk.trailing_atr_multiple * atr_value
        if position.lowest_price <= position.entry_price - risk.breakeven_after_atr * position.atr_at_entry:
            breakeven_candidate = position.entry_price
            trail_candidate = min(trail_candidate, breakeven_candidate)
        position.stop_price = min(position.stop_price, trail_candidate)


def evaluate_exit(position: Position, row: pd.Series, risk: RiskConfig) -> tuple[Optional[float], Optional[str]]:
    high = float(row["high"])
    low = float(row["low"])

    if position.side == 1:
        stop_hit = low <= position.stop_price
        target_hit = high >= position.take_profit_price
        if stop_hit:
            return apply_slippage(position.stop_price, -1, risk.slippage_rate), "stop_loss"
        if target_hit:
            return apply_slippage(position.take_profit_price, -1, risk.slippage_rate), "take_profit"
    else:
        stop_hit = high >= position.stop_price
        target_hit = low <= position.take_profit_price
        if stop_hit:
            return apply_slippage(position.stop_price, 1, risk.slippage_rate), "stop_loss"
        if target_hit:
            return apply_slippage(position.take_profit_price, 1, risk.slippage_rate), "take_profit"

    return None, None


def evaluate_bar_close_exit(
    position: Position,
    row: pd.Series,
    risk: RiskConfig,
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


def close_position(
    position: Position,
    timestamp: pd.Timestamp,
    exit_price: float,
    reason: Optional[str],
    cash: float,
    risk: RiskConfig,
) -> tuple[float, dict]:
    gross_pnl = (exit_price - position.entry_price) * position.quantity * position.side
    exit_fee = abs(position.quantity * exit_price) * risk.commission_rate
    net_pnl = gross_pnl - position.entry_fee - exit_fee
    cash += net_pnl

    trade = {
        "symbol": position.symbol,
        "side": "long" if position.side == 1 else "short",
        "entry_time": position.entry_time,
        "exit_time": timestamp,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "quantity": position.quantity,
        "gross_pnl": gross_pnl,
        "fees": position.entry_fee + exit_fee,
        "net_pnl": net_pnl,
        "return_pct_on_notional": net_pnl / (position.entry_price * position.quantity),
        "r_multiple": net_pnl / (position.atr_at_entry * position.quantity * risk.stop_atr_multiple),
        "bars_held": position.bars_held,
        "exit_reason": reason or "unknown",
    }
    return cash, trade


def build_equity_record(
    timestamp: pd.Timestamp,
    row: pd.Series,
    cash: float,
    position: Optional[Position],
) -> dict:
    unrealized = 0.0
    if position is not None:
        mark_price = float(row["close"])
        unrealized = (mark_price - position.entry_price) * position.quantity * position.side

    return {
        "timestamp": timestamp,
        "symbol": str(row["symbol"]),
        "cash": cash,
        "unrealized_pnl": unrealized,
        "equity": cash + unrealized,
        "close": float(row["close"]),
        "in_position": int(position is not None),
    }


def apply_slippage(price: float, trade_direction: int, slippage_rate: float) -> float:
    """trade_direction: +1 means buy, -1 means sell."""
    return price * (1 + slippage_rate * trade_direction)


# =============================================================================
# Metrics and reporting
# =============================================================================


def calculate_metrics(equity: pd.DataFrame, trades: pd.DataFrame, initial_capital: float) -> dict[str, float]:
    final_equity = float(equity["equity"].iloc[-1]) if not equity.empty else initial_capital
    total_return = final_equity / initial_capital - 1
    returns = equity["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    max_drawdown = calculate_max_drawdown(equity["equity"])
    sharpe = calculate_sharpe(returns)
    exposure_time_pct = float(equity["in_position"].mean() * 100) if "in_position" in equity else 0.0
    buy_hold_return = float(equity["close"].iloc[-1] / equity["close"].iloc[0] - 1) if len(equity) > 1 else 0.0
    return_over_drawdown = total_return / max_drawdown if max_drawdown > 0 else np.inf
    trade_count = int(len(trades))

    if trades.empty:
        return {
            "final_equity": final_equity,
            "total_return_pct": total_return * 100,
            "buy_hold_return_pct": buy_hold_return * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "return_over_drawdown": return_over_drawdown,
            "sharpe_simplified": sharpe,
            "exposure_time_pct": exposure_time_pct,
            "trade_count": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "avg_trade_pnl": 0.0,
            "win_loss_ratio": 0.0,
            "avg_trade_duration_bars": 0.0,
            "avg_r_multiple": 0.0,
        }

    wins = trades.loc[trades["net_pnl"] > 0, "net_pnl"]
    losses = trades.loc[trades["net_pnl"] < 0, "net_pnl"]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    win_rate = len(wins) / trade_count
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    win_loss_ratio = avg_win / abs(avg_loss) if avg_loss < 0 else np.inf

    return {
        "final_equity": final_equity,
        "total_return_pct": total_return * 100,
        "buy_hold_return_pct": buy_hold_return * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "return_over_drawdown": return_over_drawdown,
        "sharpe_simplified": sharpe,
        "exposure_time_pct": exposure_time_pct,
        "trade_count": trade_count,
        "win_rate_pct": win_rate * 100,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "avg_trade_pnl": float(trades["net_pnl"].mean()),
        "win_loss_ratio": win_loss_ratio,
        "avg_trade_duration_bars": float(trades["bars_held"].mean()),
        "avg_r_multiple": float(trades["r_multiple"].mean()) if "r_multiple" in trades else 0.0,
    }


def calculate_max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return abs(float(drawdown.min()))


def calculate_sharpe(returns: pd.Series) -> float:
    if returns.empty or returns.std(ddof=0) == 0:
        return 0.0
    # Simplified bar-based Sharpe. For professional reporting, annualize using
    # the actual timeframe metadata rather than assuming a fixed calendar.
    return float(returns.mean() / returns.std(ddof=0) * np.sqrt(len(returns)))


def print_metrics(symbol: str, metrics: dict[str, float]) -> None:
    print(f"\n=== {symbol.upper()} BACKTEST SUMMARY ===")
    for key, value in metrics.items():
        if np.isinf(value):
            printable = "inf"
        elif key.endswith("_pct"):
            printable = f"{value:,.2f}%"
        else:
            printable = f"{value:,.4f}"
        print(f"{key:24s}: {printable}")


def print_exit_breakdown(trades: pd.DataFrame) -> None:
    if trades.empty:
        return

    breakdown = (
        trades.groupby("exit_reason")
        .agg(
            trades=("net_pnl", "size"),
            net_pnl=("net_pnl", "sum"),
            avg_pnl=("net_pnl", "mean"),
            win_rate_pct=("net_pnl", lambda values: (values > 0).mean() * 100),
            avg_bars=("bars_held", "mean"),
        )
        .sort_values("net_pnl")
    )
    print("\nExit reason breakdown:")
    print(breakdown.to_string())


# =============================================================================
# Visualization
# =============================================================================


def plot_results(
    symbol: str,
    data: pd.DataFrame,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    strategy_config: StrategyConfig,
    output_dir: Optional[Path],
) -> None:
    enriched = add_signals(add_indicators(data, strategy_config), strategy_config)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    axes[0].plot(equity.index, equity["equity"], label="Equity", color="#1f77b4")
    axes[0].set_title(f"{symbol.upper()} equity curve")
    axes[0].set_ylabel("Equity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(enriched.index, enriched["close"], label="Close", color="#222222", linewidth=1)
    axes[1].plot(enriched.index, enriched["ema_fast"], label="EMA fast", color="#2ca02c", alpha=0.8)
    axes[1].plot(enriched.index, enriched["ema_slow"], label="EMA slow", color="#d62728", alpha=0.8)

    if not trades.empty:
        long_entries = trades[trades["side"] == "long"]
        short_entries = trades[trades["side"] == "short"]
        axes[1].scatter(long_entries["entry_time"], long_entries["entry_price"], marker="^", color="green", label="Long entry")
        axes[1].scatter(short_entries["entry_time"], short_entries["entry_price"], marker="v", color="red", label="Short entry")
        axes[1].scatter(trades["exit_time"], trades["exit_price"], marker="x", color="black", label="Exit")

    axes[1].set_title(f"{symbol.upper()} price and trades")
    axes[1].set_ylabel("Price")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{symbol.lower()}_backtest.png", dpi=150)
    else:
        plt.show()

    plt.close(fig)


# =============================================================================
# Main orchestration
# =============================================================================


def parse_symbol_paths(values: Iterable[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --csv value '{value}'. Use SYMBOL=path/to/file.csv")
        symbol, path = value.split("=", 1)
        mapping[symbol.upper()] = Path(path)
    return mapping


def run_for_symbol(symbol: str, path: Path, config: BacktestConfig, output_dir: Optional[Path] | bool) -> None:
    data = load_ohlcv_csv(path, symbol)
    equity, trades = run_backtest(data, config)
    metrics = calculate_metrics(equity, trades, config.risk.initial_capital)

    print_metrics(symbol, metrics)
    if not trades.empty:
        print_exit_breakdown(trades)
        print("\nLast trades:")
        print(trades.tail(10).to_string(index=False))

    if isinstance(output_dir, Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        equity.to_csv(output_dir / f"{symbol.lower()}_equity.csv")
        trades.to_csv(output_dir / f"{symbol.lower()}_trades.csv", index=False)

    if output_dir is not False:
        plot_results(symbol, data, equity, trades, config.strategy, output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Professional-style crypto OHLCV backtester")
    parser.add_argument(
        "--csv",
        nargs="+",
        default=None,
        help="CSV inputs as SYMBOL=path. Example: --csv BTCUSDT=data/BTCUSDT_1h.csv ETHUSDT=data/ETHUSDT_1h.csv",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download monthly Binance public klines before running the backtest",
    )
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Symbols to download when using --download")
    parser.add_argument("--interval", default="1h", help="Binance kline interval, for example 15m, 1h, 4h, 1d")
    parser.add_argument("--start", default="2024-01", help="Download start month in YYYY-MM format")
    parser.add_argument("--end", default=None, help="Download end month in YYYY-MM format. Defaults to the last complete month")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory where downloaded CSVs are stored")
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position-fraction", type=float, default=0.35)
    parser.add_argument("--commission-rate", type=float, default=0.0004)
    parser.add_argument("--slippage-rate", type=float, default=0.0002)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("backtest_output"))
    parser.add_argument("--show-plots", action="store_true", help="Show plots interactively instead of saving PNG files")
    parser.add_argument("--no-plots", action="store_true", help="Skip chart generation for faster research runs")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = build_arg_parser().parse_args()

    config = BacktestConfig(
        strategy=StrategyConfig(allow_short=args.allow_short),
        risk=RiskConfig(
            initial_capital=args.initial_capital,
            risk_per_trade=args.risk_per_trade,
            max_position_fraction=args.max_position_fraction,
            commission_rate=args.commission_rate,
            slippage_rate=args.slippage_rate,
        ),
    )

    symbol_paths: dict[str, Path] = {}
    if args.download:
        end_month = args.end or default_end_month()
        for symbol in args.symbols:
            path = download_binance_monthly_klines(
                symbol=symbol,
                interval=args.interval,
                start_month=args.start,
                end_month=end_month,
                data_dir=args.data_dir,
            )
            symbol_paths[symbol.upper()] = path

    if args.csv:
        symbol_paths.update(parse_symbol_paths(args.csv))

    if not symbol_paths:
        raise SystemExit("Provide --csv SYMBOL=path or use --download to fetch Binance data automatically.")

    output_dir: Optional[Path] | bool
    if args.no_plots:
        output_dir = False
    else:
        output_dir = None if args.show_plots else args.output_dir

    for symbol, path in symbol_paths.items():
        logger.info("Running backtest for %s using %s", symbol, path)
        run_for_symbol(symbol, path, config, output_dir)


if __name__ == "__main__":
    main()
