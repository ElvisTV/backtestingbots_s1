from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcusdt_usdm_futures_bot import (
    BinanceFuturesClient,
    RuntimeConfig,
    default_end_month,
    download_funding_rates,
    download_usdm_klines,
    load_funding_csv,
    load_ohlcv_csv,
)


SYMBOL = "BTCUSDT"
INTERVAL = "5m"
VARIANTS = ("long_only", "short_only", "both")
logger = logging.getLogger("btcusdt_usdm_5m_research")


@dataclass(frozen=True)
class ResearchStrategyConfig:
    ema_fast: int = 21
    ema_mid: int = 55
    ema_slow: int = 200
    regime_ema: int = 864  # 3 days of 5m bars.
    rsi_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    volume_sma: int = 288
    pullback_lookback: int = 24
    slope_lookback: int = 24
    min_atr_pct: float = 0.00025
    max_atr_pct: float = 0.012
    min_volume_ratio: float = 1.05
    max_volume_ratio: float = 5.0
    min_adx: float = 22.0
    min_slope_pct: float = 0.00018
    long_rsi_min: float = 50.0
    long_rsi_max: float = 62.0
    short_rsi_min: float = 38.0
    short_rsi_max: float = 50.0
    max_distance_from_fast_atr: float = 0.85


@dataclass(frozen=True)
class ResearchRiskConfig:
    initial_equity: float = 2_000.0
    leverage: int = 2
    risk_per_trade: float = 0.0015
    max_margin_fraction: float = 0.20
    taker_fee_rate: float = 0.0004
    slippage_rate: float = 0.00025
    stop_atr_multiple: float = 1.45
    take_profit_atr_multiple: float = 2.00
    trailing_atr_multiple: float = 1.20
    breakeven_after_atr: float = 1.00
    max_bars_in_trade: int = 36
    cooldown_bars: int = 24


@dataclass
class Position:
    side: int
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
    funding_pnl: float = 0.0

    @property
    def notional(self) -> float:
        return self.entry_price * self.quantity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTCUSDT USD-M 5m multi-year research backtester")
    parser.add_argument("--start-month", default=None, help="YYYY-MM. Defaults to 60 completed months ending last month.")
    parser.add_argument("--end-month", default=None, help="YYYY-MM. Defaults to latest completed month.")
    parser.add_argument("--data-dir", type=Path, default=Path("data_usdm_5m_research"))
    parser.add_argument("--output-dir", type=Path, default=Path("usdm_5m_research_output"))
    parser.add_argument("--initial-equity", type=float, default=2_000.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def five_year_month_window(start_month: Optional[str], end_month: Optional[str]) -> tuple[str, str]:
    end = pd.Period(end_month or default_end_month(), freq="M")
    start = pd.Period(start_month, freq="M") if start_month else end - 59
    return start.strftime("%Y-%m"), end.strftime("%Y-%m")


def ensure_data(start_month: str, end_month: str, data_dir: Path, force: bool) -> tuple[Path, Path]:
    interval_dir = data_dir / INTERVAL
    kline_path = interval_dir / f"{SYMBOL}_USDM_{INTERVAL}.csv"
    if force or not kline_path.exists():
        kline_path = download_usdm_klines(SYMBOL, INTERVAL, start_month, end_month, interval_dir)
    data = load_ohlcv_csv(kline_path, SYMBOL)
    if str(data.index[0].to_period("M")) > start_month or str(data.index[-1].to_period("M")) < end_month:
        kline_path = download_usdm_klines(SYMBOL, INTERVAL, start_month, end_month, interval_dir)
        data = load_ohlcv_csv(kline_path, SYMBOL)

    funding_path = data_dir / f"{SYMBOL}_funding.csv"
    should_refresh_funding = force or not funding_path.exists()
    if funding_path.exists() and not should_refresh_funding:
        funding = load_funding_csv(funding_path)
        should_refresh_funding = funding.empty or funding.index[0] > data.index[0] or funding.index[-1] < data.index[-1]
    if should_refresh_funding:
        client = BinanceFuturesClient(RuntimeConfig(testnet=False, dry_run=True))
        funding_path = download_funding_rates(client, SYMBOL, data.index[0], data.index[-1], data_dir)
    return kline_path, funding_path


def add_indicators(df: pd.DataFrame, config: ResearchStrategyConfig) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=config.ema_fast, adjust=False).mean()
    out["ema_mid"] = out["close"].ewm(span=config.ema_mid, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=config.ema_slow, adjust=False).mean()
    out["regime_ema"] = out["close"].ewm(span=config.regime_ema, adjust=False).mean()
    out["ema_slow_slope_pct"] = out["ema_slow"].pct_change(config.slope_lookback)
    out["regime_slope_pct"] = out["regime_ema"].pct_change(config.slope_lookback)
    out["rsi"] = rsi(out["close"], config.rsi_period)
    out["atr"] = atr(out, config.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["adx"] = adx(out, config.adx_period)
    out["volume_sma"] = out["volume"].rolling(config.volume_sma).mean()
    out["volume_ratio"] = out["volume"] / out["volume_sma"]
    out["distance_from_fast_atr"] = (out["close"] - out["ema_fast"]) / out["atr"]
    out["recent_low_touched_fast"] = out["low"].rolling(config.pullback_lookback).min().shift(1) <= out["ema_fast"].shift(1)
    out["recent_high_touched_fast"] = out["high"].rolling(config.pullback_lookback).max().shift(1) >= out["ema_fast"].shift(1)
    out["prior_high_break"] = out["close"] > out["high"].shift(1).rolling(6).max()
    out["prior_low_break"] = out["close"] < out["low"].shift(1).rolling(6).min()
    return out


def add_signals(df: pd.DataFrame, config: ResearchStrategyConfig) -> pd.DataFrame:
    out = df.copy()
    regime_ok = (
        out["volume_ratio"].between(config.min_volume_ratio, config.max_volume_ratio)
        & out["atr_pct"].between(config.min_atr_pct, config.max_atr_pct)
        & (out["adx"] >= config.min_adx)
    )
    long_trend = (
        (out["ema_fast"] > out["ema_mid"])
        & (out["ema_mid"] > out["ema_slow"])
        & (out["close"] > out["regime_ema"])
        & (out["ema_slow_slope_pct"] > config.min_slope_pct)
        & (out["regime_slope_pct"] > 0)
    )
    short_trend = (
        (out["ema_fast"] < out["ema_mid"])
        & (out["ema_mid"] < out["ema_slow"])
        & (out["close"] < out["regime_ema"])
        & (out["ema_slow_slope_pct"] < -config.min_slope_pct)
        & (out["regime_slope_pct"] < 0)
    )
    long_trigger = (
        out["recent_low_touched_fast"]
        & out["prior_high_break"]
        & out["rsi"].between(config.long_rsi_min, config.long_rsi_max)
        & (out["rsi"] > out["rsi"].shift(1))
        & (out["close"] > out["open"])
        & (out["close"] > out["ema_fast"])
        & (out["distance_from_fast_atr"].between(0, config.max_distance_from_fast_atr))
    )
    short_trigger = (
        out["recent_high_touched_fast"]
        & out["prior_low_break"]
        & out["rsi"].between(config.short_rsi_min, config.short_rsi_max)
        & (out["rsi"] < out["rsi"].shift(1))
        & (out["close"] < out["open"])
        & (out["close"] < out["ema_fast"])
        & (out["distance_from_fast_atr"].between(-config.max_distance_from_fast_atr, 0))
    )
    out["long_signal"] = (regime_ok & long_trend & long_trigger).astype(int)
    out["short_signal"] = (regime_ok & short_trend & short_trigger).astype(int) * -1
    out["long_entry"] = out["long_signal"].shift(1).fillna(0).astype(int)
    out["short_entry"] = out["short_signal"].shift(1).fillna(0).astype(int)
    return out


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


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
    atr_smooth = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_smooth
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_smooth
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def run_backtest(data: pd.DataFrame, funding: pd.DataFrame, variant: str, risk: ResearchRiskConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    equity = risk.initial_equity
    position: Optional[Position] = None
    records: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    cooldown = 0
    last_timestamp: Optional[pd.Timestamp] = None

    for timestamp, row in data.iterrows():
        if not row_is_tradeable(row):
            records.append(equity_record(timestamp, row, equity, position, 0.0))
            continue

        funding_pnl = funding_between(funding, last_timestamp, timestamp, position, float(row["close"]))
        if not np.isfinite(funding_pnl):
            funding_pnl = 0.0
        if position is not None:
            position.funding_pnl += funding_pnl
        equity += funding_pnl
        last_timestamp = timestamp

        if position is not None:
            position.bars_held += 1
            position.highest_price = max(position.highest_price, float(row["high"]))
            position.lowest_price = min(position.lowest_price, float(row["low"]))
            exit_price, exit_reason = evaluate_exit(position, row, risk)
            if exit_price is not None:
                equity, trade = close_position(position, timestamp, exit_price, exit_reason, equity, risk)
                trade["variant"] = variant
                trades.append(trade)
                position = None
                cooldown = risk.cooldown_bars
            else:
                update_trailing_stop(position, row, risk)

        if position is None and cooldown <= 0:
            side = entry_side(row, variant)
            if side != 0:
                position = open_position(row, timestamp, side, equity, risk)
                if position is not None:
                    equity -= position.entry_fee
        elif position is None and cooldown > 0:
            cooldown -= 1

        records.append(equity_record(timestamp, row, equity, position, funding_pnl))

    if position is not None:
        final_row = data.iloc[-1]
        equity, trade = close_position(position, data.index[-1], float(final_row["close"]), "end_of_data", equity, risk)
        trade["variant"] = variant
        trades.append(trade)
        records[-1] = equity_record(data.index[-1], final_row, equity, None, 0.0)

    return pd.DataFrame(records).set_index("timestamp"), pd.DataFrame(trades)


def entry_side(row: pd.Series, variant: str) -> int:
    long_signal = int(row["long_entry"]) == 1
    short_signal = int(row["short_entry"]) == -1
    if variant == "long_only":
        return 1 if long_signal else 0
    if variant == "short_only":
        return -1 if short_signal else 0
    if long_signal and not short_signal:
        return 1
    if short_signal and not long_signal:
        return -1
    return 0


def row_is_tradeable(row: pd.Series) -> bool:
    required = ["open", "high", "low", "close", "atr", "ema_fast", "long_entry", "short_entry"]
    return bool(pd.notna(row[required]).all() and float(row["atr"]) > 0)


def open_position(row: pd.Series, timestamp: pd.Timestamp, side: int, equity: float, risk: ResearchRiskConfig) -> Optional[Position]:
    if not np.isfinite(equity) or equity <= 0:
        return None
    atr_value = float(row["atr"])
    entry_price = apply_slippage(float(row["open"]), "buy" if side == 1 else "sell", risk.slippage_rate)
    stop_distance = risk.stop_atr_multiple * atr_value
    if stop_distance <= 0:
        return None
    qty_by_risk = (equity * risk.risk_per_trade) / stop_distance
    qty_by_margin = (equity * risk.max_margin_fraction * risk.leverage) / entry_price
    quantity = min(qty_by_risk, qty_by_margin)
    if not np.isfinite(quantity) or quantity <= 0:
        return None
    notional = quantity * entry_price
    margin_used = notional / risk.leverage
    if margin_used > equity * risk.max_margin_fraction:
        return None
    if side == 1:
        stop_price = entry_price - stop_distance
        take_profit = entry_price + risk.take_profit_atr_multiple * atr_value
    else:
        stop_price = entry_price + stop_distance
        take_profit = entry_price - risk.take_profit_atr_multiple * atr_value
    return Position(
        side=side,
        entry_time=timestamp,
        entry_price=entry_price,
        quantity=quantity,
        stop_price=stop_price,
        take_profit_price=take_profit,
        atr_at_entry=atr_value,
        entry_fee=notional * risk.taker_fee_rate,
        margin_used=margin_used,
        highest_price=entry_price,
        lowest_price=entry_price,
    )


def evaluate_exit(position: Position, row: pd.Series, risk: ResearchRiskConfig) -> tuple[Optional[float], Optional[str]]:
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    if position.side == 1:
        if low <= position.stop_price:
            return apply_slippage(position.stop_price, "sell", risk.slippage_rate), "stop_loss"
        if high >= position.take_profit_price:
            return apply_slippage(position.take_profit_price, "sell", risk.slippage_rate), "take_profit"
        if position.bars_held >= risk.max_bars_in_trade:
            return apply_slippage(close, "sell", risk.slippage_rate), "time_stop"
        if close < float(row["ema_fast"]) and float(row["rsi"]) < 46:
            return apply_slippage(close, "sell", risk.slippage_rate), "trend_invalid"
    else:
        if high >= position.stop_price:
            return apply_slippage(position.stop_price, "buy", risk.slippage_rate), "stop_loss"
        if low <= position.take_profit_price:
            return apply_slippage(position.take_profit_price, "buy", risk.slippage_rate), "take_profit"
        if position.bars_held >= risk.max_bars_in_trade:
            return apply_slippage(close, "buy", risk.slippage_rate), "time_stop"
        if close > float(row["ema_fast"]) and float(row["rsi"]) > 54:
            return apply_slippage(close, "buy", risk.slippage_rate), "trend_invalid"
    return None, None


def update_trailing_stop(position: Position, row: pd.Series, risk: ResearchRiskConfig) -> None:
    atr_value = float(row["atr"])
    if position.side == 1:
        if position.highest_price >= position.entry_price + risk.breakeven_after_atr * position.atr_at_entry:
            position.stop_price = max(position.stop_price, position.entry_price)
        candidate = float(row["close"]) - risk.trailing_atr_multiple * atr_value
        position.stop_price = max(position.stop_price, candidate)
    else:
        if position.lowest_price <= position.entry_price - risk.breakeven_after_atr * position.atr_at_entry:
            position.stop_price = min(position.stop_price, position.entry_price)
        candidate = float(row["close"]) + risk.trailing_atr_multiple * atr_value
        position.stop_price = min(position.stop_price, candidate)


def apply_slippage(price: float, action: str, slippage_rate: float) -> float:
    return price * (1 + slippage_rate) if action == "buy" else price * (1 - slippage_rate)


def close_position(position: Position, timestamp: pd.Timestamp, exit_price: float, reason: str, equity: float, risk: ResearchRiskConfig) -> tuple[float, dict[str, Any]]:
    gross_pnl = (exit_price - position.entry_price) * position.quantity * position.side
    exit_fee = abs(position.quantity * exit_price) * risk.taker_fee_rate
    net_pnl = gross_pnl - position.entry_fee - exit_fee + position.funding_pnl
    equity += gross_pnl - exit_fee
    risk_amount = position.atr_at_entry * risk.stop_atr_multiple * position.quantity
    return equity, {
        "symbol": SYMBOL,
        "side": "long" if position.side == 1 else "short",
        "entry_time": position.entry_time,
        "exit_time": timestamp,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "quantity": position.quantity,
        "notional": position.notional,
        "margin_used": position.margin_used,
        "leverage": risk.leverage,
        "gross_pnl": gross_pnl,
        "fees": position.entry_fee + exit_fee,
        "funding_pnl": position.funding_pnl,
        "net_pnl": net_pnl,
        "r_multiple": net_pnl / risk_amount if risk_amount > 0 else np.nan,
        "bars_held": position.bars_held,
        "exit_reason": reason,
    }


def equity_record(timestamp: pd.Timestamp, row: pd.Series, equity: float, position: Optional[Position], funding_pnl: float) -> dict[str, Any]:
    unrealized = 0.0
    margin_used = 0.0
    side = 0
    if position is not None:
        unrealized = (float(row["close"]) - position.entry_price) * position.quantity * position.side
        margin_used = position.margin_used
        side = position.side
    return {
        "timestamp": timestamp,
        "symbol": SYMBOL,
        "close": float(row["close"]),
        "equity": equity + unrealized,
        "realized_equity": equity,
        "in_position": position is not None,
        "position_side": side,
        "margin_used": margin_used,
        "funding_pnl": funding_pnl,
    }


def funding_between(funding: pd.DataFrame, previous_time: Optional[pd.Timestamp], current_time: pd.Timestamp, position: Optional[Position], mark_price: float) -> float:
    if position is None or funding.empty or previous_time is None:
        return 0.0
    window = funding[(funding.index > previous_time) & (funding.index <= current_time)]
    pnl = 0.0
    for _, row in window.iterrows():
        rate = float(row.get("funding_rate", 0.0))
        if not np.isfinite(rate):
            continue
        row_mark = float(row.get("mark_price", mark_price))
        mark = row_mark if np.isfinite(row_mark) and row_mark > 0 else mark_price
        if not np.isfinite(mark) or mark <= 0:
            continue
        notional = position.quantity * mark
        pnl += -position.side * notional * rate
    return pnl


def metrics(equity: pd.DataFrame, trades: pd.DataFrame, initial_equity: float) -> dict[str, float]:
    equity_series = pd.to_numeric(equity["equity"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if equity_series.empty:
        raise ValueError("Equity series has no finite values")
    final_equity = float(equity_series.iloc[-1])
    returns = equity_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    dd = equity_series / equity_series.cummax() - 1
    finite_trades = trades.copy()
    if not finite_trades.empty:
        finite_trades = finite_trades[pd.to_numeric(finite_trades["net_pnl"], errors="coerce").notna()]
    wins = finite_trades.loc[finite_trades["net_pnl"] > 0, "net_pnl"] if not finite_trades.empty else pd.Series(dtype=float)
    losses = finite_trades.loc[finite_trades["net_pnl"] < 0, "net_pnl"] if not finite_trades.empty else pd.Series(dtype=float)
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses.sum())) if not losses.empty else 0.0
    max_dd = abs(float(dd.min())) if not dd.empty else 0.0
    total_return = final_equity / initial_equity - 1
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    return {
        "final_equity": final_equity,
        "total_return_pct": total_return * 100,
        "cagr_pct": ((final_equity / initial_equity) ** (1 / years) - 1) * 100,
        "max_drawdown_pct": max_dd * 100,
        "return_over_drawdown": total_return / max_dd if max_dd > 0 else np.nan,
        "trade_count": int(len(finite_trades)),
        "trades_per_year": int(len(finite_trades)) / years,
        "win_rate_pct": float((finite_trades["net_pnl"] > 0).mean() * 100) if not finite_trades.empty else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.nan,
        "expectancy": float(finite_trades["net_pnl"].mean()) if not finite_trades.empty else 0.0,
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
        "sharpe_simplified": float((returns.mean() / returns.std()) * np.sqrt(365 * 24 * 12)) if returns.std() > 0 else 0.0,
        "exposure_time_pct": float(equity["in_position"].mean() * 100),
        "avg_trade_duration_bars": float(finite_trades["bars_held"].mean()) if not finite_trades.empty else 0.0,
        "total_fees": float(finite_trades["fees"].sum()) if not finite_trades.empty else 0.0,
    }


def segment_metrics(equity: pd.DataFrame, trades: pd.DataFrame, freq: str, label_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, eq_part in equity.groupby(equity.index.to_period(freq)):
        if eq_part.empty:
            continue
        start = eq_part.index[0]
        end = eq_part.index[-1]
        tr_part = trades[(pd.to_datetime(trades["exit_time"], utc=True) >= start) & (pd.to_datetime(trades["exit_time"], utc=True) <= end)] if not trades.empty else trades
        initial = float(eq_part["equity"].iloc[0])
        row = {label_name: str(label), "start": str(start), "end": str(end), **metrics(eq_part, tr_part, initial)}
        rows.append(row)
    return pd.DataFrame(rows)


def is_oos_metrics(equity: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    cutoff = equity.index[0] + (equity.index[-1] - equity.index[0]) / 2
    rows = []
    for label, mask in [("in_sample", equity.index <= cutoff), ("out_of_sample", equity.index > cutoff)]:
        eq_part = equity.loc[mask]
        if eq_part.empty:
            continue
        tr_part = trades[(pd.to_datetime(trades["exit_time"], utc=True) >= eq_part.index[0]) & (pd.to_datetime(trades["exit_time"], utc=True) <= eq_part.index[-1])] if not trades.empty else trades
        rows.append({"sample": label, "cutoff": str(cutoff), **metrics(eq_part, tr_part, float(eq_part["equity"].iloc[0]))})
    return pd.DataFrame(rows)


def walk_forward_metrics(equity: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window diagnostic: previous years as context, next year as forward block."""
    years = sorted(pd.Index(equity.index.year).unique())
    rows: list[dict[str, Any]] = []
    exit_times = pd.to_datetime(trades["exit_time"], utc=True) if not trades.empty else pd.Series(dtype="datetime64[ns, UTC]")
    for test_year in years[1:]:
        train_eq = equity[equity.index.year < test_year]
        test_eq = equity[equity.index.year == test_year]
        if train_eq.empty or test_eq.empty:
            continue
        train_tr = trades[exit_times.dt.year < test_year] if not trades.empty else trades
        test_tr = trades[exit_times.dt.year == test_year] if not trades.empty else trades
        train_m = metrics(train_eq, train_tr, float(train_eq["equity"].iloc[0]))
        test_m = metrics(test_eq, test_tr, float(test_eq["equity"].iloc[0]))
        rows.append(
            {
                "train_through": int(test_year - 1),
                "test_year": int(test_year),
                "train_trade_count": train_m["trade_count"],
                "train_profit_factor": train_m["profit_factor"],
                "train_expectancy": train_m["expectancy"],
                "train_total_return_pct": train_m["total_return_pct"],
                "test_trade_count": test_m["trade_count"],
                "test_profit_factor": test_m["profit_factor"],
                "test_expectancy": test_m["expectancy"],
                "test_total_return_pct": test_m["total_return_pct"],
                "test_max_drawdown_pct": test_m["max_drawdown_pct"],
            }
        )
    return pd.DataFrame(rows)


def classify_frequency(trades_per_year: float) -> str:
    if trades_per_year < 25:
        return "too_low"
    if trades_per_year > 600:
        return "too_high"
    return "reasonable"


def plot_equity(equity_by_variant: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for variant, equity in equity_by_variant.items():
        equity["equity"].plot(ax=ax, label=variant)
    ax.set_title("BTCUSDT USD-M 5m Research Equity")
    ax.set_ylabel("Equity USDT")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_report(summary: pd.DataFrame, yearly: pd.DataFrame, is_oos: pd.DataFrame, walk_forward: pd.DataFrame, output_dir: Path, start_month: str, end_month: str) -> None:
    lines = [
        "# BTCUSDT USD-M 5m Research Backtest",
        "",
        f"Period: {start_month} through {end_month}.",
        "Execution timeframe: 5m only. Variants: long_only, short_only, both.",
        "Costs: taker fees and slippage modeled. Capital: 2000 USDT. Leverage: 2x.",
        "",
        "## Summary",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "## Yearly",
        "```",
        yearly.to_string(index=False),
        "```",
        "",
        "## IS/OOS",
        "```",
        is_oos.to_string(index=False),
        "```",
        "",
        "## Walk-Forward",
        "```",
        walk_forward.to_string(index=False),
        "```",
        "",
        "Research output only. Not live-trading approval.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = parse_args()
    start_month, end_month = five_year_month_window(args.start_month, args.end_month)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kline_path, funding_path = ensure_data(start_month, end_month, args.data_dir, args.force_download)
    raw = load_ohlcv_csv(kline_path, SYMBOL)
    funding = load_funding_csv(funding_path)
    strategy = ResearchStrategyConfig()
    risk = ResearchRiskConfig(initial_equity=args.initial_equity)
    enriched = add_signals(add_indicators(raw, strategy), strategy)

    summary_rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    is_oos_frames: list[pd.DataFrame] = []
    walk_forward_frames: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    equity_by_variant: dict[str, pd.DataFrame] = {}

    for variant in VARIANTS:
        logger.info("Running variant=%s", variant)
        equity, trades = run_backtest(enriched, funding, variant, risk)
        equity_by_variant[variant] = equity
        if not trades.empty:
            trades = trades.copy()
            trades["variant"] = variant
            all_trades.append(trades)
        m = metrics(equity, trades, risk.initial_equity)
        summary_rows.append(
            {
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "variant": variant,
                "start": str(raw.index[0]),
                "end": str(raw.index[-1]),
                "rows": len(raw),
                "frequency_class": classify_frequency(m["trades_per_year"]),
                **m,
            }
        )
        yearly = segment_metrics(equity, trades, "Y", "year")
        yearly.insert(0, "variant", variant)
        yearly_frames.append(yearly)
        sample = is_oos_metrics(equity, trades)
        sample.insert(0, "variant", variant)
        is_oos_frames.append(sample)
        forward = walk_forward_metrics(equity, trades)
        forward.insert(0, "variant", variant)
        walk_forward_frames.append(forward)
        equity.to_csv(args.output_dir / f"{variant}_equity.csv")
        trades.to_csv(args.output_dir / f"{variant}_trades.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    yearly_all = pd.concat(yearly_frames, ignore_index=True)
    is_oos_all = pd.concat(is_oos_frames, ignore_index=True)
    walk_forward_all = pd.concat(walk_forward_frames, ignore_index=True)
    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    wide_equity = pd.DataFrame({variant: eq["equity"] for variant, eq in equity_by_variant.items()})

    summary.to_csv(args.output_dir / "summary.csv", index=False)
    yearly_all.to_csv(args.output_dir / "yearly.csv", index=False)
    is_oos_all.to_csv(args.output_dir / "is_oos.csv", index=False)
    walk_forward_all.to_csv(args.output_dir / "walk_forward.csv", index=False)
    trades_all.to_csv(args.output_dir / "trades.csv", index=False)
    wide_equity.to_csv(args.output_dir / "equity.csv")
    (args.output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps({"strategy": asdict(strategy), "risk": asdict(risk)}, indent=2), encoding="utf-8")
    if args.plot:
        plot_equity(equity_by_variant, args.output_dir / "equity_comparison.png")
    write_report(summary, yearly_all, is_oos_all, walk_forward_all, args.output_dir, start_month, end_month)
    print(summary.to_string(index=False))
    print("\nYearly:")
    print(yearly_all[["variant", "year", "total_return_pct", "max_drawdown_pct", "trade_count", "profit_factor", "expectancy"]].to_string(index=False))
    print("\nIS/OOS:")
    print(is_oos_all[["variant", "sample", "total_return_pct", "max_drawdown_pct", "trade_count", "profit_factor", "expectancy"]].to_string(index=False))
    print("\nWalk-forward:")
    print(walk_forward_all[["variant", "train_through", "test_year", "test_trade_count", "test_profit_factor", "test_expectancy", "test_total_return_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
