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
SUPPORTED_INTERVALS = {"5m", "1m"}
logger = logging.getLogger("btcusdt_usdm_scalping_backtester")


@dataclass(frozen=True)
class ScalpStrategyConfig:
    ema_fast: int = 21
    ema_slow: int = 55
    trend_ema: int = 200
    rsi_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    volume_sma: int = 48
    pullback_lookback: int = 6
    breakdown_lookback: int = 12
    min_volume_ratio: float = 0.90
    max_volume_ratio: float = 6.0
    min_atr_pct: float = 0.00015
    max_atr_pct: float = 0.018
    min_adx: float = 16.0
    rsi_max: float = 50.0
    min_ema_slow_slope_pct: float = 0.00004
    min_breakdown_atr: float = 0.03


@dataclass(frozen=True)
class ScalpRiskConfig:
    initial_equity: float = 1_000.0
    leverage: int = 2
    risk_per_trade: float = 0.0025
    max_margin_fraction: float = 0.50
    taker_fee_rate: float = 0.0004
    slippage_rate: float = 0.00025
    stop_atr_multiple: float = 1.20
    take_profit_atr_multiple: float = 1.45
    trailing_atr_multiple: float = 1.00
    breakeven_after_atr: float = 0.90
    max_bars_in_trade: int = 18
    cooldown_bars: int = 4


@dataclass
class ScalpPosition:
    entry_time: pd.Timestamp
    entry_price: float
    quantity: float
    stop_price: float
    take_profit_price: float
    atr_at_entry: float
    entry_fee: float
    margin_used: float
    lowest_price: float
    bars_held: int = 0

    @property
    def notional(self) -> float:
        return self.entry_price * self.quantity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTCUSDT USD-M futures short-only scalping backtester")
    parser.add_argument("--intervals", nargs="+", default=["5m", "1m"], choices=sorted(SUPPORTED_INTERVALS))
    parser.add_argument("--start-month", default=None, help="YYYY-MM. Defaults to last 12 completed monthly archives.")
    parser.add_argument("--end-month", default=None, help="YYYY-MM. Defaults to latest completed monthly archive.")
    parser.add_argument("--data-dir", type=Path, default=Path("data_usdm_scalping"))
    parser.add_argument("--output-dir", type=Path, default=Path("scalping_backtest_output"))
    parser.add_argument("--initial-equity", type=float, default=1_000.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def completed_month_window(start_month: Optional[str], end_month: Optional[str]) -> tuple[str, str]:
    end = pd.Period(end_month or default_end_month(), freq="M")
    start = pd.Period(start_month, freq="M") if start_month else end - 11
    return start.strftime("%Y-%m"), end.strftime("%Y-%m")


def ensure_data(symbol: str, interval: str, start_month: str, end_month: str, data_dir: Path, force: bool) -> tuple[Path, Path]:
    interval_dir = data_dir / interval
    kline_path = interval_dir / f"{symbol}_USDM_{interval}.csv"
    if force or not kline_path.exists():
        kline_path = download_usdm_klines(symbol, interval, start_month, end_month, interval_dir)

    data = load_ohlcv_csv(kline_path, symbol)
    funding_path = data_dir / f"{symbol}_funding.csv"
    if force or not funding_path.exists():
        client = BinanceFuturesClient(RuntimeConfig(testnet=False, dry_run=True))
        funding_path = download_funding_rates(client, symbol, data.index[0], data.index[-1], data_dir)
    return kline_path, funding_path


def add_indicators(df: pd.DataFrame, config: ScalpStrategyConfig) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=config.ema_fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=config.ema_slow, adjust=False).mean()
    out["trend_ema"] = out["close"].ewm(span=config.trend_ema, adjust=False).mean()
    out["ema_slow_slope_pct"] = out["ema_slow"].pct_change(config.pullback_lookback)
    out["rsi"] = rsi(out["close"], config.rsi_period)
    out["atr"] = atr(out, config.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["adx"] = adx(out, config.adx_period)
    out["volume_sma"] = out["volume"].rolling(config.volume_sma).mean()
    out["volume_ratio"] = out["volume"] / out["volume_sma"]
    out["prev_breakdown_low"] = out["low"].rolling(config.breakdown_lookback).min().shift(1)
    out["breakdown_distance_atr"] = (out["prev_breakdown_low"] - out["close"]) / out["atr"]
    out["recent_pullback_to_fast"] = out["high"].rolling(config.pullback_lookback).max().shift(1) >= out["ema_fast"].shift(1)
    return out


def add_short_signals(df: pd.DataFrame, config: ScalpStrategyConfig) -> pd.DataFrame:
    out = df.copy()
    downtrend = (
        (out["ema_fast"] < out["ema_slow"])
        & (out["ema_slow"] < out["trend_ema"])
        & (out["close"] < out["ema_fast"])
        & (out["ema_slow_slope_pct"] < -config.min_ema_slow_slope_pct)
    )
    regime_ok = (
        out["volume_ratio"].between(config.min_volume_ratio, config.max_volume_ratio)
        & out["atr_pct"].between(config.min_atr_pct, config.max_atr_pct)
        & (out["adx"] >= config.min_adx)
    )
    trigger = (
        out["recent_pullback_to_fast"]
        & (out["rsi"] <= config.rsi_max)
        & (out["rsi"] < out["rsi"].shift(1))
        & (out["breakdown_distance_atr"] >= config.min_breakdown_atr)
    )
    out["raw_short_signal"] = (downtrend & regime_ok & trigger).astype(int) * -1
    out["entry_signal"] = out["raw_short_signal"].shift(1).fillna(0).astype(int)
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


def run_backtest(df: pd.DataFrame, funding: pd.DataFrame, strategy: ScalpStrategyConfig, risk: ScalpRiskConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = add_short_signals(add_indicators(df, strategy), strategy)
    equity = risk.initial_equity
    position: Optional[ScalpPosition] = None
    records: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    cooldown = 0
    last_timestamp: Optional[pd.Timestamp] = None

    for timestamp, row in data.iterrows():
        if not row_is_tradeable(row):
            records.append(equity_record(timestamp, row, equity, position, 0.0))
            continue

        funding_pnl = funding_between(funding, last_timestamp, timestamp, position, float(row["close"]))
        equity += funding_pnl
        last_timestamp = timestamp

        if position is not None:
            position.bars_held += 1
            position.lowest_price = min(position.lowest_price, float(row["low"]))
            exit_price, exit_reason = evaluate_exit(position, row, risk)
            if exit_price is not None:
                equity, trade = close_position(position, timestamp, exit_price, exit_reason, equity, risk, funding_pnl)
                trades.append(trade)
                position = None
                cooldown = risk.cooldown_bars
            else:
                update_trailing_stop(position, row, risk)

        if position is None and cooldown <= 0 and int(row["entry_signal"]) == -1:
            position = open_short(row, timestamp, equity, risk)
            if position is not None:
                equity -= position.entry_fee
        elif position is None and cooldown > 0:
            cooldown -= 1

        records.append(equity_record(timestamp, row, equity, position, funding_pnl))

    if position is not None:
        final_row = data.iloc[-1]
        equity, trade = close_position(position, data.index[-1], float(final_row["close"]), "end_of_data", equity, risk, 0.0)
        trades.append(trade)
        records[-1] = equity_record(data.index[-1], final_row, equity, None, 0.0)

    return pd.DataFrame(records).set_index("timestamp"), pd.DataFrame(trades)


def row_is_tradeable(row: pd.Series) -> bool:
    return bool(pd.notna(row[["open", "high", "low", "close", "atr", "ema_fast", "entry_signal"]]).all() and row["atr"] > 0)


def open_short(row: pd.Series, timestamp: pd.Timestamp, equity: float, risk: ScalpRiskConfig) -> Optional[ScalpPosition]:
    atr_value = float(row["atr"])
    entry_price = apply_slippage(float(row["open"]), "sell", risk.slippage_rate)
    stop_distance = risk.stop_atr_multiple * atr_value
    if stop_distance <= 0:
        return None

    qty_by_risk = (equity * risk.risk_per_trade) / stop_distance
    qty_by_margin = (equity * risk.max_margin_fraction * risk.leverage) / entry_price
    quantity = min(qty_by_risk, qty_by_margin)
    if quantity <= 0:
        return None

    notional = quantity * entry_price
    margin_used = notional / risk.leverage
    if margin_used > equity * risk.max_margin_fraction:
        return None

    return ScalpPosition(
        entry_time=timestamp,
        entry_price=entry_price,
        quantity=quantity,
        stop_price=entry_price + stop_distance,
        take_profit_price=entry_price - risk.take_profit_atr_multiple * atr_value,
        atr_at_entry=atr_value,
        entry_fee=notional * risk.taker_fee_rate,
        margin_used=margin_used,
        lowest_price=entry_price,
    )


def evaluate_exit(position: ScalpPosition, row: pd.Series, risk: ScalpRiskConfig) -> tuple[Optional[float], Optional[str]]:
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    if high >= position.stop_price:
        return apply_slippage(position.stop_price, "buy", risk.slippage_rate), "stop_loss"
    if low <= position.take_profit_price:
        return apply_slippage(position.take_profit_price, "buy", risk.slippage_rate), "take_profit"
    if position.bars_held >= risk.max_bars_in_trade:
        return apply_slippage(close, "buy", risk.slippage_rate), "time_stop"
    if close > float(row["ema_fast"]) and float(row["rsi"]) > 54:
        return apply_slippage(close, "buy", risk.slippage_rate), "trend_invalid"
    return None, None


def update_trailing_stop(position: ScalpPosition, row: pd.Series, risk: ScalpRiskConfig) -> None:
    atr_value = float(row["atr"])
    if position.lowest_price <= position.entry_price - risk.breakeven_after_atr * position.atr_at_entry:
        position.stop_price = min(position.stop_price, position.entry_price)
    candidate = float(row["close"]) + risk.trailing_atr_multiple * atr_value
    position.stop_price = min(position.stop_price, candidate)


def apply_slippage(price: float, action: str, slippage_rate: float) -> float:
    return price * (1 + slippage_rate) if action == "buy" else price * (1 - slippage_rate)


def close_position(
    position: ScalpPosition,
    timestamp: pd.Timestamp,
    exit_price: float,
    reason: str,
    equity: float,
    risk: ScalpRiskConfig,
    funding_pnl: float,
) -> tuple[float, dict[str, Any]]:
    gross_pnl = (position.entry_price - exit_price) * position.quantity
    exit_fee = abs(position.quantity * exit_price) * risk.taker_fee_rate
    net_pnl = gross_pnl - position.entry_fee - exit_fee
    equity += net_pnl
    risk_amount = position.atr_at_entry * risk.stop_atr_multiple * position.quantity
    return equity, {
        "symbol": SYMBOL,
        "side": "short",
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
        "last_bar_funding_pnl": funding_pnl,
        "net_pnl": net_pnl,
        "r_multiple": net_pnl / risk_amount if risk_amount > 0 else np.nan,
        "bars_held": position.bars_held,
        "exit_reason": reason,
    }


def equity_record(timestamp: pd.Timestamp, row: pd.Series, equity: float, position: Optional[ScalpPosition], funding_pnl: float) -> dict[str, Any]:
    unrealized = 0.0
    margin_used = 0.0
    if position is not None:
        unrealized = (position.entry_price - float(row["close"])) * position.quantity
        margin_used = position.margin_used
    return {
        "timestamp": timestamp,
        "symbol": SYMBOL,
        "close": float(row["close"]),
        "equity": equity + unrealized,
        "realized_equity": equity,
        "in_position": position is not None,
        "margin_used": margin_used,
        "funding_pnl": funding_pnl,
    }


def funding_between(
    funding: pd.DataFrame,
    previous_time: Optional[pd.Timestamp],
    current_time: pd.Timestamp,
    position: Optional[ScalpPosition],
    mark_price: float,
) -> float:
    if position is None or funding.empty or previous_time is None:
        return 0.0
    window = funding[(funding.index > previous_time) & (funding.index <= current_time)]
    pnl = 0.0
    for _, row in window.iterrows():
        notional = position.quantity * float(row.get("mark_price", mark_price) or mark_price)
        pnl += notional * float(row["funding_rate"])
    return pnl


def calculate_metrics(equity: pd.DataFrame, trades: pd.DataFrame, initial_equity: float, bars_per_year: int) -> dict[str, float]:
    final_equity = float(equity["equity"].iloc[-1])
    returns = equity["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    dd = equity["equity"] / equity["equity"].cummax() - 1
    wins = trades.loc[trades["net_pnl"] > 0, "net_pnl"] if not trades.empty else pd.Series(dtype=float)
    losses = trades.loc[trades["net_pnl"] < 0, "net_pnl"] if not trades.empty else pd.Series(dtype=float)
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses.sum())) if not losses.empty else 0.0
    total_return = final_equity / initial_equity - 1
    max_dd = abs(float(dd.min())) if not dd.empty else 0.0
    return {
        "final_equity": final_equity,
        "total_return_pct": total_return * 100,
        "max_drawdown_pct": max_dd * 100,
        "return_over_drawdown": total_return / max_dd if max_dd > 0 else np.nan,
        "trade_count": int(len(trades)),
        "win_rate_pct": float((trades["net_pnl"] > 0).mean() * 100) if not trades.empty else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.nan,
        "expectancy": float(trades["net_pnl"].mean()) if not trades.empty else 0.0,
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
        "win_loss_ratio": abs(float(wins.mean() / losses.mean())) if not wins.empty and not losses.empty else np.nan,
        "sharpe_simplified": float((returns.mean() / returns.std()) * np.sqrt(bars_per_year)) if returns.std() > 0 else 0.0,
        "exposure_time_pct": float(equity["in_position"].mean() * 100),
        "avg_trade_duration_bars": float(trades["bars_held"].mean()) if not trades.empty else 0.0,
        "total_fees": float(trades["fees"].sum()) if not trades.empty else 0.0,
    }


def bars_per_year(interval: str) -> int:
    return {"1m": 365 * 24 * 60, "5m": 365 * 24 * 12}[interval]


def interval_risk(interval: str, initial_equity: float) -> ScalpRiskConfig:
    if interval == "1m":
        return ScalpRiskConfig(initial_equity=initial_equity, max_bars_in_trade=45, cooldown_bars=8, slippage_rate=0.00030)
    return ScalpRiskConfig(initial_equity=initial_equity, max_bars_in_trade=18, cooldown_bars=4, slippage_rate=0.00025)


def export_plot(equity: pd.DataFrame, trades: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    equity["equity"].plot(ax=ax, title=path.stem)
    ax.set_ylabel("Equity USDT")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(summary: pd.DataFrame, output_dir: Path, start_month: str, end_month: str) -> None:
    table = summary.to_string(index=False)
    lines = [
        "# BTCUSDT USD-M Scalping Backtest",
        "",
        f"Period: monthly archives {start_month} through {end_month}.",
        "Scope: BTCUSDT perpetual futures, short-only, 2x leverage, taker fees and slippage modeled.",
        "",
        "```",
        table,
        "```",
        "",
        "Note: this is research output, not live-trading approval.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = parse_args()
    start_month, end_month = completed_month_window(args.start_month, args.end_month)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    all_trades: list[pd.DataFrame] = []
    strategy = ScalpStrategyConfig()

    for interval in args.intervals:
        logger.info("Running %s %s scalping backtest from %s to %s", SYMBOL, interval, start_month, end_month)
        kline_path, funding_path = ensure_data(SYMBOL, interval, start_month, end_month, args.data_dir, args.force_download)
        data = load_ohlcv_csv(kline_path, SYMBOL)
        funding = load_funding_csv(funding_path)
        risk = interval_risk(interval, args.initial_equity)
        equity, trades = run_backtest(data, funding, strategy, risk)
        metrics = calculate_metrics(equity, trades, risk.initial_equity, bars_per_year(interval))
        row = {
            "symbol": SYMBOL,
            "interval": interval,
            "start": str(data.index[0]),
            "end": str(data.index[-1]),
            "rows": len(data),
            **metrics,
        }
        rows.append(row)
        equity.to_csv(args.output_dir / f"{SYMBOL.lower()}_{interval}_equity.csv")
        trades.to_csv(args.output_dir / f"{SYMBOL.lower()}_{interval}_trades.csv", index=False)
        if not trades.empty:
            t = trades.copy()
            t["interval"] = interval
            all_trades.append(t)
        if args.plot:
            export_plot(equity, trades, args.output_dir / f"{SYMBOL.lower()}_{interval}_equity.png")

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(args.output_dir / "all_trades.csv", index=False)
    else:
        pd.DataFrame().to_csv(args.output_dir / "all_trades.csv", index=False)
    (args.output_dir / "config.json").write_text(
        json.dumps({"strategy": asdict(strategy), "risk_5m": asdict(interval_risk("5m", args.initial_equity)), "risk_1m": asdict(interval_risk("1m", args.initial_equity))}, indent=2),
        encoding="utf-8",
    )
    write_report(summary, args.output_dir, start_month, end_month)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
