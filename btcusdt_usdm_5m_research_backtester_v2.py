from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import btcusdt_usdm_5m_research_backtester as core


SYMBOL = core.SYMBOL
INTERVAL = core.INTERVAL
VARIANTS = core.VARIANTS
logger = logging.getLogger("btcusdt_usdm_5m_research_v2")


@dataclass(frozen=True)
class LiquiditySweepConfig:
    """5m-only failed-breakout hypothesis.

    The strategy looks for a stop-run beyond a recent 5m range that fails by the
    candle close. This is intentionally a different family from EMA pullback
    continuation: it is a controlled mean-reversion/rejection setup.
    """

    range_lookback: int = 72  # 6 hours of 5m bars.
    volume_sma: int = 288  # 24 hours of 5m bars.
    atr_period: int = 14
    rsi_period: int = 14
    adx_period: int = 14
    ema_bias: int = 200
    min_atr_pct: float = 0.00025
    max_atr_pct: float = 0.012
    min_volume_ratio: float = 1.20
    max_volume_ratio: float = 8.0
    max_adx: float = 34.0
    min_sweep_atr: float = 0.12
    min_reclaim_atr: float = 0.04
    max_close_distance_ema_atr: float = 4.0
    long_rsi_min: float = 28.0
    long_rsi_max: float = 52.0
    short_rsi_min: float = 48.0
    short_rsi_max: float = 72.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTCUSDT USD-M 5m liquidity-sweep research backtester")
    parser.add_argument("--start-month", default=None)
    parser.add_argument("--end-month", default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data_usdm_5m_research"))
    parser.add_argument("--output-dir", type=Path, default=Path("usdm_5m_research_output_v2"))
    parser.add_argument("--initial-equity", type=float, default=2_000.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def add_indicators(df: pd.DataFrame, config: LiquiditySweepConfig) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = core.atr(out, config.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["rsi"] = core.rsi(out["close"], config.rsi_period)
    out["adx"] = core.adx(out, config.adx_period)
    out["ema_bias"] = out["close"].ewm(span=config.ema_bias, adjust=False).mean()
    out["ema_fast"] = out["ema_bias"]
    out["volume_sma"] = out["volume"].rolling(config.volume_sma).mean()
    out["volume_ratio"] = out["volume"] / out["volume_sma"]
    out["prior_range_high"] = out["high"].shift(1).rolling(config.range_lookback).max()
    out["prior_range_low"] = out["low"].shift(1).rolling(config.range_lookback).min()
    out["sweep_high_atr"] = (out["high"] - out["prior_range_high"]) / out["atr"]
    out["sweep_low_atr"] = (out["prior_range_low"] - out["low"]) / out["atr"]
    out["high_reclaim_atr"] = (out["prior_range_high"] - out["close"]) / out["atr"]
    out["low_reclaim_atr"] = (out["close"] - out["prior_range_low"]) / out["atr"]
    out["close_distance_ema_atr"] = (out["close"] - out["ema_bias"]).abs() / out["atr"]
    out["body_pct"] = (out["close"] - out["open"]).abs() / (out["high"] - out["low"]).replace(0, np.nan)
    return out


def add_signals(df: pd.DataFrame, config: LiquiditySweepConfig) -> pd.DataFrame:
    out = df.copy()
    regime_ok = (
        out["atr_pct"].between(config.min_atr_pct, config.max_atr_pct)
        & out["volume_ratio"].between(config.min_volume_ratio, config.max_volume_ratio)
        & (out["adx"] <= config.max_adx)
        & (out["close_distance_ema_atr"] <= config.max_close_distance_ema_atr)
        & (out["body_pct"] >= 0.25)
    )
    failed_breakdown_long = (
        (out["low"] < out["prior_range_low"])
        & (out["close"] > out["prior_range_low"])
        & (out["sweep_low_atr"] >= config.min_sweep_atr)
        & (out["low_reclaim_atr"] >= config.min_reclaim_atr)
        & (out["close"] > out["open"])
        & out["rsi"].between(config.long_rsi_min, config.long_rsi_max)
    )
    failed_breakout_short = (
        (out["high"] > out["prior_range_high"])
        & (out["close"] < out["prior_range_high"])
        & (out["sweep_high_atr"] >= config.min_sweep_atr)
        & (out["high_reclaim_atr"] >= config.min_reclaim_atr)
        & (out["close"] < out["open"])
        & out["rsi"].between(config.short_rsi_min, config.short_rsi_max)
    )
    out["long_signal"] = (regime_ok & failed_breakdown_long).astype(int)
    out["short_signal"] = (regime_ok & failed_breakout_short).astype(int) * -1
    out["long_entry"] = out["long_signal"].shift(1).fillna(0).astype(int)
    out["short_entry"] = out["short_signal"].shift(1).fillna(0).astype(int)
    return out


def write_report(
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    is_oos: pd.DataFrame,
    walk_forward: pd.DataFrame,
    output_dir: Path,
    start_month: str,
    end_month: str,
) -> None:
    lines = [
        "# BTCUSDT USD-M 5m Liquidity Sweep Research",
        "",
        f"Period: {start_month} through {end_month}.",
        "Execution timeframe: 5m only. Hypothesis: failed breakout / liquidity sweep.",
        "Variants: long_only, short_only, both. Costs include taker fees, slippage and funding.",
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


def run_research(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start_month, end_month = core.five_year_month_window(args.start_month, args.end_month)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    kline_path, funding_path = core.ensure_data(start_month, end_month, args.data_dir, args.force_download)
    raw = core.load_ohlcv_csv(kline_path, SYMBOL)
    funding = core.load_funding_csv(funding_path)

    strategy = LiquiditySweepConfig()
    risk = core.ResearchRiskConfig(
        initial_equity=args.initial_equity,
        leverage=2,
        risk_per_trade=0.0015,
        max_margin_fraction=0.20,
        stop_atr_multiple=1.10,
        take_profit_atr_multiple=1.55,
        trailing_atr_multiple=0.95,
        breakeven_after_atr=0.80,
        max_bars_in_trade=24,
        cooldown_bars=18,
    )
    enriched = add_signals(add_indicators(raw, strategy), strategy)

    summary_rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    is_oos_frames: list[pd.DataFrame] = []
    walk_forward_frames: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    equity_by_variant: dict[str, pd.DataFrame] = {}

    for variant in VARIANTS:
        logger.info("Running variant=%s", variant)
        equity, trades = core.run_backtest(enriched, funding, variant, risk)
        equity_by_variant[variant] = equity
        if not trades.empty:
            trades = trades.copy()
            trades["variant"] = variant
            all_trades.append(trades)

        m = core.metrics(equity, trades, risk.initial_equity)
        summary_rows.append(
            {
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "variant": variant,
                "hypothesis": "liquidity_sweep_failed_breakout",
                "start": str(raw.index[0]),
                "end": str(raw.index[-1]),
                "rows": len(raw),
                "frequency_class": core.classify_frequency(m["trades_per_year"]),
                **m,
            }
        )

        yearly = core.segment_metrics(equity, trades, "Y", "year")
        yearly.insert(0, "variant", variant)
        yearly_frames.append(yearly)
        sample = core.is_oos_metrics(equity, trades)
        sample.insert(0, "variant", variant)
        is_oos_frames.append(sample)
        forward = core.walk_forward_metrics(equity, trades)
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
        core.plot_equity(equity_by_variant, args.output_dir / "equity_comparison.png")
    write_report(summary, yearly_all, is_oos_all, walk_forward_all, args.output_dir, start_month, end_month)
    return summary, yearly_all, is_oos_all, walk_forward_all


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = parse_args()
    summary, yearly, is_oos, walk_forward = run_research(args)
    print(summary.to_string(index=False))
    print("\nYearly:")
    print(yearly[["variant", "year", "total_return_pct", "max_drawdown_pct", "trade_count", "profit_factor", "expectancy"]].to_string(index=False))
    print("\nIS/OOS:")
    print(is_oos[["variant", "sample", "total_return_pct", "max_drawdown_pct", "trade_count", "profit_factor", "expectancy"]].to_string(index=False))
    print("\nWalk-forward:")
    print(walk_forward[["variant", "train_through", "test_year", "test_trade_count", "test_profit_factor", "test_expectancy", "test_total_return_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
