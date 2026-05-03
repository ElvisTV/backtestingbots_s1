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
INTERVAL = "1m"
VARIANTS = ("long_only", "short_only", "both")
logger = logging.getLogger("btcusdt_usdm_1m_research")


@dataclass(frozen=True)
class Strategy1mConfig:
    ema_fast: int = 20
    ema_slow: int = 50
    vwap_session: str = "UTC"
    atr_period: int = 14
    rsi_period: int = 14
    volume_sma: int = 60
    pullback_lookback: int = 10
    confirmation_lookback: int = 3
    min_atr_pct: float = 0.00008
    max_atr_pct: float = 0.0040
    min_volume_ratio: float = 1.10
    max_volume_ratio: float = 8.0
    max_distance_fast_atr: float = 0.85
    long_rsi_min: float = 50.0
    long_rsi_max: float = 66.0
    short_rsi_min: float = 34.0
    short_rsi_max: float = 50.0


@dataclass(frozen=True)
class Risk1mConfig:
    initial_equity: float = 2_000.0
    leverage: int = 2
    risk_per_trade: float = 0.0010
    max_margin_fraction: float = 0.15
    taker_fee_rate: float = 0.0004
    slippage_rate: float = 0.00025
    stop_atr_multiple: float = 1.20
    take_profit_atr_multiple: float = 1.70
    trailing_atr_multiple: float = 0.95
    breakeven_after_atr: float = 0.80
    max_bars_in_trade: int = 30
    cooldown_bars: int = 20


@dataclass
class Position:
    side: int
    entry_i: int
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
    parser = argparse.ArgumentParser(description="BTCUSDT USD-M 1m 5-year research backtester")
    parser.add_argument("--start-month", default=None)
    parser.add_argument("--end-month", default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data_usdm_1m_research"))
    parser.add_argument("--output-dir", type=Path, default=Path("usdm_1m_research_output"))
    parser.add_argument("--initial-equity", type=float, default=2_000.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def five_year_month_window(start_month: Optional[str], end_month: Optional[str]) -> tuple[str, str]:
    end = pd.Period(end_month or default_end_month(), freq="M")
    start = pd.Period(start_month, freq="M") if start_month else end - 59
    if start > end:
        raise ValueError("start_month must be <= end_month")
    return start.strftime("%Y-%m"), end.strftime("%Y-%m")


def ensure_data(start_month: str, end_month: str, data_dir: Path, force: bool) -> tuple[Path, Path]:
    interval_dir = data_dir / INTERVAL
    kline_path = interval_dir / f"{SYMBOL}_USDM_{INTERVAL}.csv"
    if force or not kline_path.exists():
        kline_path = download_usdm_klines(SYMBOL, INTERVAL, start_month, end_month, interval_dir)

    data = load_ohlcv_csv(kline_path, SYMBOL)
    first_month = str(data.index[0].to_period("M"))
    last_month = str(data.index[-1].to_period("M"))
    if first_month > start_month or last_month < end_month:
        logger.info("Cached 1m file has %s..%s, refreshing for %s..%s", first_month, last_month, start_month, end_month)
        kline_path = download_usdm_klines(SYMBOL, INTERVAL, start_month, end_month, interval_dir)
        data = load_ohlcv_csv(kline_path, SYMBOL)

    funding_path = data_dir / f"{SYMBOL}_funding.csv"
    should_refresh_funding = force or not funding_path.exists()
    if funding_path.exists() and not should_refresh_funding:
        funding = load_funding_csv(funding_path)
        should_refresh_funding = funding.empty or funding.index[0] > data.index[0] or funding.index[-1] < data.index[-1]
    if should_refresh_funding:
        client = BinanceFuturesClient(RuntimeConfig(symbol=SYMBOL, interval=INTERVAL, testnet=False, dry_run=True))
        funding_path = download_funding_rates(client, SYMBOL, data.index[0], data.index[-1], data_dir)
    return kline_path, funding_path


def add_indicators(df: pd.DataFrame, cfg: Strategy1mConfig) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    out["rsi"] = rsi(out["close"], cfg.rsi_period)
    out["atr"] = atr(out, cfg.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["volume_sma"] = out["volume"].rolling(cfg.volume_sma).mean()
    out["volume_ratio"] = out["volume"] / out["volume_sma"]
    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    session = out.index.floor("D")
    cum_pv = (typical * out["volume"]).groupby(session).cumsum()
    cum_volume = out["volume"].groupby(session).cumsum()
    out["session_vwap"] = cum_pv / cum_volume.replace(0, np.nan)
    out["distance_fast_atr"] = (out["close"] - out["ema_fast"]) / out["atr"]
    out["recent_low_touched_fast"] = out["low"].rolling(cfg.pullback_lookback).min().shift(1) <= out["ema_fast"].shift(1)
    out["recent_high_touched_fast"] = out["high"].rolling(cfg.pullback_lookback).max().shift(1) >= out["ema_fast"].shift(1)
    out["prior_high_break"] = out["close"] > out["high"].shift(1).rolling(cfg.confirmation_lookback).max()
    out["prior_low_break"] = out["close"] < out["low"].shift(1).rolling(cfg.confirmation_lookback).min()
    return out


def add_signals(df: pd.DataFrame, cfg: Strategy1mConfig) -> pd.DataFrame:
    out = df.copy()
    market_ok = (
        out["atr_pct"].between(cfg.min_atr_pct, cfg.max_atr_pct)
        & out["volume_ratio"].between(cfg.min_volume_ratio, cfg.max_volume_ratio)
        & out["session_vwap"].notna()
    )
    long_setup = (
        market_ok
        & (out["ema_fast"] > out["ema_slow"])
        & (out["close"] > out["session_vwap"])
        & out["recent_low_touched_fast"]
        & out["prior_high_break"]
        & (out["close"] > out["ema_fast"])
        & (out["close"] > out["open"])
        & out["rsi"].between(cfg.long_rsi_min, cfg.long_rsi_max)
        & out["distance_fast_atr"].between(0, cfg.max_distance_fast_atr)
    )
    short_setup = (
        market_ok
        & (out["ema_fast"] < out["ema_slow"])
        & (out["close"] < out["session_vwap"])
        & out["recent_high_touched_fast"]
        & out["prior_low_break"]
        & (out["close"] < out["ema_fast"])
        & (out["close"] < out["open"])
        & out["rsi"].between(cfg.short_rsi_min, cfg.short_rsi_max)
        & out["distance_fast_atr"].between(-cfg.max_distance_fast_atr, 0)
    )
    out["long_signal"] = long_setup.astype(np.int8)
    out["short_signal"] = short_setup.astype(np.int8) * -1
    out["long_entry"] = out["long_signal"].shift(1).fillna(0).astype(np.int8)
    out["short_entry"] = out["short_signal"].shift(1).fillna(0).astype(np.int8)
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
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - previous_close).abs(), (df["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def entry_side(long_signal: int, short_signal: int, variant: str) -> int:
    if variant == "long_only":
        return 1 if long_signal == 1 else 0
    if variant == "short_only":
        return -1 if short_signal == -1 else 0
    if long_signal == 1 and short_signal != -1:
        return 1
    if short_signal == -1 and long_signal != 1:
        return -1
    return 0


def apply_slippage(price: float, action: str, slippage_rate: float) -> float:
    return price * (1 + slippage_rate) if action == "buy" else price * (1 - slippage_rate)


def run_backtest(data: pd.DataFrame, funding: pd.DataFrame, variant: str, risk: Risk1mConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = data.index
    n = len(data)
    open_arr = data["open"].to_numpy(dtype=float)
    high_arr = data["high"].to_numpy(dtype=float)
    low_arr = data["low"].to_numpy(dtype=float)
    close_arr = data["close"].to_numpy(dtype=float)
    atr_arr = data["atr"].to_numpy(dtype=float)
    ema_arr = data["ema_fast"].to_numpy(dtype=float)
    rsi_arr = data["rsi"].to_numpy(dtype=float)
    long_arr = data["long_entry"].to_numpy(dtype=np.int8)
    short_arr = data["short_entry"].to_numpy(dtype=np.int8)

    ts_ns = index.view("int64")
    funding_times = funding.index.view("int64") if not funding.empty else np.array([], dtype=np.int64)
    funding_rates = funding["funding_rate"].to_numpy(dtype=float) if not funding.empty else np.array([], dtype=float)
    funding_marks = funding["mark_price"].to_numpy(dtype=float) if not funding.empty and "mark_price" in funding.columns else np.array([], dtype=float)
    funding_i = 0

    equity = risk.initial_equity
    position: Optional[Position] = None
    cooldown = 0
    trades: list[dict[str, Any]] = []

    equity_values = np.empty(n, dtype=float)
    realized_values = np.empty(n, dtype=float)
    in_position_values = np.zeros(n, dtype=bool)
    side_values = np.zeros(n, dtype=np.int8)
    margin_values = np.zeros(n, dtype=float)
    funding_values = np.zeros(n, dtype=float)

    for i in range(n):
        while funding_i < len(funding_times) and funding_times[funding_i] <= ts_ns[i]:
            if position is not None:
                rate = funding_rates[funding_i]
                mark = funding_marks[funding_i] if funding_i < len(funding_marks) else close_arr[i]
                if np.isfinite(rate) and np.isfinite(mark) and mark > 0:
                    pnl = -position.side * position.quantity * mark * rate
                    equity += pnl
                    position.funding_pnl += pnl
                    funding_values[i] += pnl
            funding_i += 1

        if position is not None:
            position.bars_held += 1
            position.highest_price = max(position.highest_price, high_arr[i])
            position.lowest_price = min(position.lowest_price, low_arr[i])
            exit_price, exit_reason = evaluate_exit(position, i, open_arr, high_arr, low_arr, close_arr, atr_arr, ema_arr, rsi_arr, risk)
            if exit_price is not None:
                equity, trade = close_position(position, i, index[i], exit_price, exit_reason, equity, risk)
                trade["variant"] = variant
                trades.append(trade)
                position = None
                cooldown = risk.cooldown_bars
            else:
                update_trailing_stop(position, i, close_arr, atr_arr, risk)

        if position is None and cooldown <= 0 and np.isfinite(atr_arr[i]) and atr_arr[i] > 0:
            side = entry_side(int(long_arr[i]), int(short_arr[i]), variant)
            if side != 0 and equity > 0:
                position = open_position(i, index[i], side, open_arr[i], atr_arr[i], equity, risk)
                if position is not None:
                    equity -= position.entry_fee
        elif position is None and cooldown > 0:
            cooldown -= 1

        unrealized = 0.0
        if position is not None:
            unrealized = (close_arr[i] - position.entry_price) * position.quantity * position.side
            in_position_values[i] = True
            side_values[i] = position.side
            margin_values[i] = position.margin_used
        equity_values[i] = equity + unrealized
        realized_values[i] = equity

    if position is not None:
        equity, trade = close_position(position, n - 1, index[-1], close_arr[-1], "end_of_data", equity, risk)
        trade["variant"] = variant
        trades.append(trade)
        equity_values[-1] = equity
        realized_values[-1] = equity
        in_position_values[-1] = False
        side_values[-1] = 0
        margin_values[-1] = 0.0

    equity_df = pd.DataFrame(
        {
            "symbol": SYMBOL,
            "close": close_arr,
            "equity": equity_values,
            "realized_equity": realized_values,
            "in_position": in_position_values,
            "position_side": side_values,
            "margin_used": margin_values,
            "funding_pnl": funding_values,
        },
        index=index,
    )
    equity_df.index.name = "timestamp"
    return equity_df, pd.DataFrame(trades)


def open_position(i: int, timestamp: pd.Timestamp, side: int, raw_open: float, atr_value: float, equity: float, risk: Risk1mConfig) -> Optional[Position]:
    entry_price = apply_slippage(raw_open, "buy" if side == 1 else "sell", risk.slippage_rate)
    stop_distance = risk.stop_atr_multiple * atr_value
    if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(stop_distance) or stop_distance <= 0:
        return None
    qty_by_risk = (equity * risk.risk_per_trade) / stop_distance
    qty_by_margin = (equity * risk.max_margin_fraction * risk.leverage) / entry_price
    quantity = min(qty_by_risk, qty_by_margin)
    if not np.isfinite(quantity) or quantity <= 0:
        return None
    margin_used = quantity * entry_price / risk.leverage
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
        entry_i=i,
        entry_time=timestamp,
        entry_price=entry_price,
        quantity=quantity,
        stop_price=stop_price,
        take_profit_price=take_profit,
        atr_at_entry=atr_value,
        entry_fee=quantity * entry_price * risk.taker_fee_rate,
        margin_used=margin_used,
        highest_price=entry_price,
        lowest_price=entry_price,
    )


def evaluate_exit(
    position: Position,
    i: int,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    atr_arr: np.ndarray,
    ema_arr: np.ndarray,
    rsi_arr: np.ndarray,
    risk: Risk1mConfig,
) -> tuple[Optional[float], Optional[str]]:
    high = high_arr[i]
    low = low_arr[i]
    close = close_arr[i]
    if position.side == 1:
        if low <= position.stop_price:
            return apply_slippage(position.stop_price, "sell", risk.slippage_rate), "stop_loss"
        if high >= position.take_profit_price:
            return apply_slippage(position.take_profit_price, "sell", risk.slippage_rate), "take_profit"
        if position.bars_held >= risk.max_bars_in_trade:
            return apply_slippage(close, "sell", risk.slippage_rate), "time_stop"
        if close < ema_arr[i] and rsi_arr[i] < 48:
            return apply_slippage(close, "sell", risk.slippage_rate), "trend_invalid"
    else:
        if high >= position.stop_price:
            return apply_slippage(position.stop_price, "buy", risk.slippage_rate), "stop_loss"
        if low <= position.take_profit_price:
            return apply_slippage(position.take_profit_price, "buy", risk.slippage_rate), "take_profit"
        if position.bars_held >= risk.max_bars_in_trade:
            return apply_slippage(close, "buy", risk.slippage_rate), "time_stop"
        if close > ema_arr[i] and rsi_arr[i] > 52:
            return apply_slippage(close, "buy", risk.slippage_rate), "trend_invalid"
    return None, None


def update_trailing_stop(position: Position, i: int, close_arr: np.ndarray, atr_arr: np.ndarray, risk: Risk1mConfig) -> None:
    atr_value = atr_arr[i]
    if not np.isfinite(atr_value) or atr_value <= 0:
        return
    if position.side == 1:
        if position.highest_price >= position.entry_price + risk.breakeven_after_atr * position.atr_at_entry:
            position.stop_price = max(position.stop_price, position.entry_price)
        position.stop_price = max(position.stop_price, close_arr[i] - risk.trailing_atr_multiple * atr_value)
    else:
        if position.lowest_price <= position.entry_price - risk.breakeven_after_atr * position.atr_at_entry:
            position.stop_price = min(position.stop_price, position.entry_price)
        position.stop_price = min(position.stop_price, close_arr[i] + risk.trailing_atr_multiple * atr_value)


def close_position(position: Position, exit_i: int, exit_time: pd.Timestamp, exit_price: float, reason: str, equity: float, risk: Risk1mConfig) -> tuple[float, dict[str, Any]]:
    gross_pnl = (exit_price - position.entry_price) * position.quantity * position.side
    exit_fee = abs(position.quantity * exit_price) * risk.taker_fee_rate
    equity += gross_pnl - exit_fee
    net_pnl = gross_pnl - position.entry_fee - exit_fee + position.funding_pnl
    risk_amount = position.atr_at_entry * risk.stop_atr_multiple * position.quantity
    return equity, {
        "symbol": SYMBOL,
        "side": "long" if position.side == 1 else "short",
        "entry_time": position.entry_time,
        "exit_time": exit_time,
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
        "bars_held": exit_i - position.entry_i,
        "exit_reason": reason,
    }


def metrics(equity: pd.DataFrame, trades: pd.DataFrame, initial_equity: float) -> dict[str, float]:
    equity_series = pd.to_numeric(equity["equity"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    final_equity = float(equity_series.iloc[-1])
    returns = equity_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    dd = equity_series / equity_series.cummax() - 1
    finite_trades = trades[pd.to_numeric(trades["net_pnl"], errors="coerce").notna()].copy() if not trades.empty else trades
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
        "cagr_pct": ((final_equity / initial_equity) ** (1 / years) - 1) * 100 if final_equity > 0 else -100.0,
        "max_drawdown_pct": max_dd * 100,
        "return_over_drawdown": total_return / max_dd if max_dd > 0 else np.nan,
        "trade_count": int(len(finite_trades)),
        "trades_per_year": int(len(finite_trades)) / years,
        "win_rate_pct": float((finite_trades["net_pnl"] > 0).mean() * 100) if not finite_trades.empty else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.nan,
        "expectancy": float(finite_trades["net_pnl"].mean()) if not finite_trades.empty else 0.0,
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
        "sharpe_simplified": float((returns.mean() / returns.std()) * np.sqrt(365 * 24 * 60)) if returns.std() > 0 else 0.0,
        "exposure_time_pct": float(equity["in_position"].mean() * 100),
        "avg_trade_duration_bars": float(finite_trades["bars_held"].mean()) if not finite_trades.empty else 0.0,
        "total_fees": float(finite_trades["fees"].sum()) if not finite_trades.empty else 0.0,
        "funding_pnl": float(finite_trades["funding_pnl"].sum()) if not finite_trades.empty else 0.0,
    }


def segment_metrics(equity: pd.DataFrame, trades: pd.DataFrame, freq: str, label_name: str) -> pd.DataFrame:
    rows = []
    exit_times = pd.to_datetime(trades["exit_time"], utc=True) if not trades.empty else pd.Series(dtype="datetime64[ns, UTC]")
    for label, eq_part in equity.groupby(equity.index.to_period(freq)):
        if eq_part.empty:
            continue
        start, end = eq_part.index[0], eq_part.index[-1]
        tr_part = trades[(exit_times >= start) & (exit_times <= end)] if not trades.empty else trades
        rows.append({label_name: str(label), "start": str(start), "end": str(end), **metrics(eq_part, tr_part, float(eq_part["equity"].iloc[0]))})
    return pd.DataFrame(rows)


def is_oos_metrics(equity: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    cutoff = equity.index[0] + (equity.index[-1] - equity.index[0]) / 2
    rows = []
    exit_times = pd.to_datetime(trades["exit_time"], utc=True) if not trades.empty else pd.Series(dtype="datetime64[ns, UTC]")
    for label, mask in [("in_sample", equity.index <= cutoff), ("out_of_sample", equity.index > cutoff)]:
        eq_part = equity.loc[mask]
        if eq_part.empty:
            continue
        tr_part = trades[(exit_times >= eq_part.index[0]) & (exit_times <= eq_part.index[-1])] if not trades.empty else trades
        rows.append({"sample": label, "cutoff": str(cutoff), **metrics(eq_part, tr_part, float(eq_part["equity"].iloc[0]))})
    return pd.DataFrame(rows)


def walk_forward_metrics(equity: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    years = sorted(pd.Index(equity.index.year).unique())
    exit_times = pd.to_datetime(trades["exit_time"], utc=True) if not trades.empty else pd.Series(dtype="datetime64[ns, UTC]")
    rows = []
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
    if trades_per_year < 50:
        return "too_low"
    if trades_per_year > 750:
        return "too_high"
    return "reasonable"


def demo_gate(summary: pd.DataFrame, is_oos: pd.DataFrame) -> dict[str, Any]:
    decisions = []
    for _, row in summary.iterrows():
        variant = row["variant"]
        oos = is_oos[(is_oos["variant"] == variant) & (is_oos["sample"] == "out_of_sample")]
        oos_pf = float(oos["profit_factor"].iloc[0]) if not oos.empty else np.nan
        oos_expectancy = float(oos["expectancy"].iloc[0]) if not oos.empty else np.nan
        passes = (
            float(row["profit_factor"]) > 1.10
            and float(row["expectancy"]) > 0
            and oos_pf > 1.0
            and oos_expectancy >= 0
            and float(row["max_drawdown_pct"]) <= 20
            and row["frequency_class"] == "reasonable"
        )
        decisions.append(
            {
                "variant": variant,
                "passes_demo_gate": bool(passes),
                "profit_factor": float(row["profit_factor"]),
                "expectancy": float(row["expectancy"]),
                "oos_profit_factor": oos_pf,
                "oos_expectancy": oos_expectancy,
                "max_drawdown_pct": float(row["max_drawdown_pct"]),
                "frequency_class": row["frequency_class"],
            }
        )
    return {"allowed": any(item["passes_demo_gate"] for item in decisions), "variants": decisions}


def plot_equity(equity_by_variant: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for variant, equity in equity_by_variant.items():
        equity["equity"].plot(ax=ax, label=variant)
    ax.set_title("BTCUSDT USD-M 1m Research Equity")
    ax.set_ylabel("Equity USDT")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_report(summary: pd.DataFrame, yearly: pd.DataFrame, is_oos: pd.DataFrame, walk_forward: pd.DataFrame, gate: dict[str, Any], output_dir: Path, start_month: str, end_month: str) -> None:
    lines = [
        "# BTCUSDT USD-M 1m Research Backtest",
        "",
        f"Period: {start_month} through {end_month}.",
        "Execution timeframe: 1m only. Variants: long_only, short_only, both.",
        "Indicators: EMA20/EMA50, UTC session VWAP, ATR14, RSI14, relative volume.",
        "Costs: taker fees, slippage and funding modeled. Capital: 2000 USDT. Leverage: 2x.",
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
        "## Demo Gate",
        "```json",
        json.dumps(gate, indent=2),
        "```",
        "",
        "Research output only. Demo trading is blocked unless Demo Gate allowed=true.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_month, end_month = five_year_month_window(args.start_month, args.end_month)
    kline_path, funding_path = ensure_data(start_month, end_month, args.data_dir, args.force_download)
    raw = load_ohlcv_csv(kline_path, SYMBOL)
    funding = load_funding_csv(funding_path)
    strategy = Strategy1mConfig()
    risk = Risk1mConfig(initial_equity=args.initial_equity)
    data = add_signals(add_indicators(raw, strategy), strategy)

    summary_rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    is_oos_frames: list[pd.DataFrame] = []
    walk_forward_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    equity_by_variant: dict[str, pd.DataFrame] = {}

    for variant in VARIANTS:
        logger.info("Running variant=%s", variant)
        equity, trades = run_backtest(data, funding, variant, risk)
        equity_by_variant[variant] = equity
        if not trades.empty:
            trades = trades.copy()
            trades["variant"] = variant
            trade_frames.append(trades)
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

    summary = pd.DataFrame(summary_rows)
    yearly_all = pd.concat(yearly_frames, ignore_index=True)
    is_oos_all = pd.concat(is_oos_frames, ignore_index=True)
    walk_forward_all = pd.concat(walk_forward_frames, ignore_index=True)
    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    gate = demo_gate(summary, is_oos_all)

    equity_wide = pd.DataFrame({"close": raw["close"]})
    for variant, equity in equity_by_variant.items():
        equity_wide[variant] = equity["equity"]
        equity_wide[f"{variant}_in_position"] = equity["in_position"].astype(int)

    summary.to_csv(args.output_dir / "summary.csv", index=False)
    yearly_all.to_csv(args.output_dir / "yearly.csv", index=False)
    is_oos_all.to_csv(args.output_dir / "is_oos.csv", index=False)
    walk_forward_all.to_csv(args.output_dir / "walk_forward.csv", index=False)
    trades_all.to_csv(args.output_dir / "trades.csv", index=False)
    equity_wide.to_csv(args.output_dir / "equity.csv")
    (args.output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps({"strategy": asdict(strategy), "risk": asdict(risk), "demo_gate": gate}, indent=2), encoding="utf-8")
    plot_equity(equity_by_variant, args.output_dir / "equity.png")
    if args.plot:
        plot_equity(equity_by_variant, args.output_dir / "equity_plot_requested.png")
    write_report(summary, yearly_all, is_oos_all, walk_forward_all, gate, args.output_dir, start_month, end_month)

    print(summary.to_string(index=False))
    print("\nYearly:")
    print(yearly_all[["variant", "year", "total_return_pct", "max_drawdown_pct", "trade_count", "profit_factor", "expectancy"]].to_string(index=False))
    print("\nIS/OOS:")
    print(is_oos_all[["variant", "sample", "total_return_pct", "max_drawdown_pct", "trade_count", "profit_factor", "expectancy"]].to_string(index=False))
    print("\nWalk-forward:")
    print(walk_forward_all[["variant", "train_through", "test_year", "test_trade_count", "test_profit_factor", "test_expectancy", "test_total_return_pct"]].to_string(index=False))
    print("\nDemo gate:")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
