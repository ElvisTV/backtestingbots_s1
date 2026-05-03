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

import btcusdt_usdm_scalping_backtester as base


SYMBOL = "BTCUSDT"
logger = logging.getLogger("btcusdt_usdm_scalping_backtester_v2")


@dataclass(frozen=True)
class V2StrategyConfig:
    ema_fast: int = 21
    ema_mid: int = 55
    ema_slow: int = 200
    rsi_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    volume_sma: int = 72
    pullback_lookback: int = 9
    breakdown_lookback: int = 24
    min_volume_ratio: float = 0.97
    max_volume_ratio: float = 4.5
    min_atr_pct: float = 0.00025
    max_atr_pct: float = 0.010
    min_adx: float = 19.0
    rsi_min: float = 31.0
    rsi_max: float = 42.8
    min_breakdown_atr: float = 0.10
    min_5m_slow_slope_pct: float = 0.00008
    min_15m_slope_pct: float = 0.00012
    min_1h_slope_pct: float = 0.00020
    max_distance_below_fast_atr: float = 1.15
    min_distance_below_fast_atr: float = 0.70
    strong_adx: float = 28.8
    strong_1h_slope_pct: float = 0.00226


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTCUSDT USD-M 5m short-only scalping backtester v2")
    parser.add_argument("--interval", default="5m", choices=["5m"])
    parser.add_argument("--start-month", default=None)
    parser.add_argument("--end-month", default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data_usdm_scalping"))
    parser.add_argument("--output-dir", type=Path, default=Path("scalping_backtest_output_v2"))
    parser.add_argument("--initial-equity", type=float, default=1_000.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def add_v2_indicators(df: pd.DataFrame, config: V2StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=config.ema_fast, adjust=False).mean()
    out["ema_mid"] = out["close"].ewm(span=config.ema_mid, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=config.ema_slow, adjust=False).mean()
    out["ema_slow_slope_pct"] = out["ema_slow"].pct_change(config.pullback_lookback)
    out["rsi"] = base.rsi(out["close"], config.rsi_period)
    out["atr"] = base.atr(out, config.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["adx"] = base.adx(out, config.adx_period)
    out["volume_sma"] = out["volume"].rolling(config.volume_sma).mean()
    out["volume_ratio"] = out["volume"] / out["volume_sma"]
    out["distance_below_fast_atr"] = (out["ema_fast"] - out["close"]) / out["atr"]
    out["prev_breakdown_low"] = out["low"].rolling(config.breakdown_lookback).min().shift(1)
    out["breakdown_distance_atr"] = (out["prev_breakdown_low"] - out["close"]) / out["atr"]
    out["pullback_touched_fast"] = out["high"].rolling(config.pullback_lookback).max().shift(1) >= out["ema_fast"].shift(1)
    out["lower_high_after_pullback"] = out["high"] < out["high"].rolling(config.pullback_lookback).max().shift(1)
    return add_context_features(out, config)


def add_context_features(df: pd.DataFrame, config: V2StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    ohlc = out[["open", "high", "low", "close", "volume"]]
    ctx_15m = build_context(ohlc, "15min", fast=20, slow=80, slope_window=8, prefix="ctx15")
    ctx_1h = build_context(ohlc, "1h", fast=12, slow=48, slope_window=6, prefix="ctx1h")
    out = pd.merge_asof(out.sort_index(), ctx_15m.sort_index(), left_index=True, right_index=True, direction="backward")
    out = pd.merge_asof(out.sort_index(), ctx_1h.sort_index(), left_index=True, right_index=True, direction="backward")
    out["ctx15_bearish"] = (
        (out["ctx15_close"] < out["ctx15_ema_fast"])
        & (out["ctx15_ema_fast"] < out["ctx15_ema_slow"])
        & (out["ctx15_slope_pct"] < -config.min_15m_slope_pct)
    )
    out["ctx1h_bearish"] = (
        (out["ctx1h_close"] < out["ctx1h_ema_fast"])
        & (out["ctx1h_ema_fast"] < out["ctx1h_ema_slow"])
        & (out["ctx1h_slope_pct"] < -config.min_1h_slope_pct)
    )
    return out


def build_context(df: pd.DataFrame, rule: str, fast: int, slow: int, slope_window: int, prefix: str) -> pd.DataFrame:
    resampled = pd.DataFrame(
        {
            "open": df["open"].resample(rule).first(),
            "high": df["high"].resample(rule).max(),
            "low": df["low"].resample(rule).min(),
            "close": df["close"].resample(rule).last(),
            "volume": df["volume"].resample(rule).sum(),
        }
    ).dropna()
    resampled[f"{prefix}_close"] = resampled["close"]
    resampled[f"{prefix}_ema_fast"] = resampled["close"].ewm(span=fast, adjust=False).mean()
    resampled[f"{prefix}_ema_slow"] = resampled["close"].ewm(span=slow, adjust=False).mean()
    resampled[f"{prefix}_slope_pct"] = resampled[f"{prefix}_ema_slow"].pct_change(slope_window)
    return resampled[[f"{prefix}_close", f"{prefix}_ema_fast", f"{prefix}_ema_slow", f"{prefix}_slope_pct"]]


def add_v2_signals(df: pd.DataFrame, config: V2StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    local_trend = (
        (out["ema_fast"] < out["ema_mid"])
        & (out["ema_mid"] < out["ema_slow"])
        & (out["ema_slow_slope_pct"] < -config.min_5m_slow_slope_pct)
    )
    regime = (
        out["ctx15_bearish"].fillna(False)
        & out["ctx1h_bearish"].fillna(False)
        & out["volume_ratio"].between(config.min_volume_ratio, config.max_volume_ratio)
        & out["atr_pct"].between(config.min_atr_pct, config.max_atr_pct)
        & (out["adx"] >= config.min_adx)
    )
    trigger = (
        out["pullback_touched_fast"]
        & out["lower_high_after_pullback"]
        & out["rsi"].between(config.rsi_min, config.rsi_max)
        & (out["rsi"] < out["rsi"].shift(1))
        & (out["close"] < out["open"])
        & (out["close"] < out["ema_fast"])
        & (out["distance_below_fast_atr"] >= config.min_distance_below_fast_atr)
        & (out["distance_below_fast_atr"] <= config.max_distance_below_fast_atr)
    )
    momentum_quality = (out["adx"] >= config.strong_adx) & (out["ctx1h_slope_pct"] <= -config.strong_1h_slope_pct)
    out["raw_short_signal"] = (local_trend & regime & trigger & momentum_quality).astype(int) * -1
    out["entry_signal"] = out["raw_short_signal"].shift(1).fillna(0).astype(int)
    return out


def run_v2_backtest(
    df: pd.DataFrame,
    funding: pd.DataFrame,
    strategy: V2StrategyConfig,
    risk: base.ScalpRiskConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = add_v2_signals(add_v2_indicators(df, strategy), strategy)
    return run_from_enriched_data(data, funding, risk)


def run_from_enriched_data(
    data: pd.DataFrame,
    funding: pd.DataFrame,
    risk: base.ScalpRiskConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    equity = risk.initial_equity
    position: Optional[base.ScalpPosition] = None
    records: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    cooldown = 0
    last_timestamp: Optional[pd.Timestamp] = None

    for timestamp, row in data.iterrows():
        if not base.row_is_tradeable(row):
            records.append(base.equity_record(timestamp, row, equity, position, 0.0))
            continue

        funding_pnl = base.funding_between(funding, last_timestamp, timestamp, position, float(row["close"]))
        equity += funding_pnl
        last_timestamp = timestamp

        if position is not None:
            position.bars_held += 1
            position.lowest_price = min(position.lowest_price, float(row["low"]))
            exit_price, exit_reason = base.evaluate_exit(position, row, risk)
            if exit_price is not None:
                equity, trade = base.close_position(position, timestamp, exit_price, exit_reason, equity, risk, funding_pnl)
                trades.append(trade)
                position = None
                cooldown = risk.cooldown_bars
            else:
                base.update_trailing_stop(position, row, risk)

        if position is None and cooldown <= 0 and int(row["entry_signal"]) == -1:
            position = base.open_short(row, timestamp, equity, risk)
            if position is not None:
                equity -= position.entry_fee
        elif position is None and cooldown > 0:
            cooldown -= 1

        records.append(base.equity_record(timestamp, row, equity, position, funding_pnl))

    if position is not None:
        final_row = data.iloc[-1]
        equity, trade = base.close_position(position, data.index[-1], float(final_row["close"]), "end_of_data", equity, risk, 0.0)
        trades.append(trade)
        records[-1] = base.equity_record(data.index[-1], final_row, equity, None, 0.0)

    return pd.DataFrame(records).set_index("timestamp"), pd.DataFrame(trades)


def v2_risk(initial_equity: float) -> base.ScalpRiskConfig:
    return base.ScalpRiskConfig(
        initial_equity=initial_equity,
        leverage=2,
        risk_per_trade=0.0015,
        max_margin_fraction=0.30,
        taker_fee_rate=0.0004,
        slippage_rate=0.00025,
        stop_atr_multiple=1.55,
        take_profit_atr_multiple=2.10,
        trailing_atr_multiple=1.25,
        breakeven_after_atr=1.10,
        max_bars_in_trade=24,
        cooldown_bars=18,
    )


def export_plot(equity: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    equity["equity"].plot(ax=ax, title=path.stem)
    ax.set_ylabel("Equity USDT")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(summary: pd.DataFrame, output_dir: Path, start_month: str, end_month: str) -> None:
    lines = [
        "# BTCUSDT USD-M Scalping Backtest V2",
        "",
        f"Period: monthly archives {start_month} through {end_month}.",
        "Scope: BTCUSDT 5m, short-only, 2x leverage, 15m/1h context filters, taker fees and slippage modeled.",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "Research output only. Not live-trading approval.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = parse_args()
    start_month, end_month = base.completed_month_window(args.start_month, args.end_month)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kline_path, funding_path = base.ensure_data(SYMBOL, "5m", start_month, end_month, args.data_dir, args.force_download)
    data = base.load_ohlcv_csv(kline_path, SYMBOL)
    funding = base.load_funding_csv(funding_path)
    strategy = V2StrategyConfig()
    risk = v2_risk(args.initial_equity)
    equity, trades = run_v2_backtest(data, funding, strategy, risk)
    metrics = base.calculate_metrics(equity, trades, risk.initial_equity, base.bars_per_year("5m"))
    row = {
        "symbol": SYMBOL,
        "interval": "5m",
        "version": "v2_context_pullback",
        "start": str(data.index[0]),
        "end": str(data.index[-1]),
        "rows": len(data),
        **metrics,
    }
    summary = pd.DataFrame([row])
    equity.to_csv(args.output_dir / "btcusdt_5m_v2_equity.csv")
    trades.to_csv(args.output_dir / "btcusdt_5m_v2_trades.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps([row], indent=2, default=str), encoding="utf-8")
    (args.output_dir / "config.json").write_text(
        json.dumps({"strategy": asdict(strategy), "risk": asdict(risk)}, indent=2),
        encoding="utf-8",
    )
    if args.plot:
        export_plot(equity, args.output_dir / "btcusdt_5m_v2_equity.png")
    write_report(summary, args.output_dir, start_month, end_month)
    print(summary.to_string(index=False))
    if not trades.empty:
        print("\nExit breakdown:")
        print(trades.groupby("exit_reason")["net_pnl"].agg(["count", "sum", "mean"]).to_string())


if __name__ == "__main__":
    main()
