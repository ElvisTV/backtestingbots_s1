from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import btcusdt_usdm_1m_research_backtester as core


SYMBOL = "BTCUSDT"
INTERVAL = "1m"
VARIANTS = ("long_only", "short_only", "both")
FAMILIES = ("mean_reversion", "liquidity_sweep", "selective_continuation")
FEE_MODELS = ("taker", "maker_like")
logger = logging.getLogger("btcusdt_usdm_1m_hypothesis_lab")


@dataclass(frozen=True)
class LabConfig:
    volume_sma: int = 60
    atr_period: int = 14
    rsi_period: int = 14
    ema_fast: int = 20
    ema_slow: int = 50
    vwap_z_window: int = 120
    range_lookback: int = 60
    continuation_lookback: int = 30
    atr_sma: int = 120
    min_atr_pct: float = 0.00006
    max_atr_pct: float = 0.0045
    liquid_start_hour_utc: int = 7
    liquid_end_hour_utc: int = 22
    no_trade_funding_minutes: int = 6
    max_signals_per_day: int = 6

    mr_min_abs_vwap_atr: float = 1.8
    mr_min_abs_vwap_z: float = 1.8
    mr_min_volume_ratio: float = 1.25
    mr_long_rsi_max: float = 32.0
    mr_short_rsi_min: float = 68.0

    sweep_min_atr: float = 0.16
    sweep_min_reclaim_atr: float = 0.05
    sweep_min_wick_ratio: float = 0.45
    sweep_min_volume_ratio: float = 1.20

    cont_min_atr_ratio: float = 1.20
    cont_min_break_atr: float = 0.16
    cont_min_volume_ratio: float = 1.35
    cont_max_distance_fast_atr: float = 1.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTCUSDT USD-M 1m hypothesis lab")
    parser.add_argument("--start-month", default=None)
    parser.add_argument("--end-month", default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data_usdm_1m_research"))
    parser.add_argument("--output-dir", type=Path, default=Path("usdm_1m_hypothesis_lab_output"))
    parser.add_argument("--initial-equity", type=float, default=2_000.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def add_common_indicators(df: pd.DataFrame, cfg: LabConfig) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    out["rsi"] = core.rsi(out["close"], cfg.rsi_period)
    out["atr"] = core.atr(out, cfg.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["atr_ratio"] = out["atr"] / out["atr"].rolling(cfg.atr_sma).mean()
    out["volume_sma"] = out["volume"].rolling(cfg.volume_sma).mean()
    out["volume_ratio"] = out["volume"] / out["volume_sma"]

    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    session = out.index.floor("D")
    cumulative_pv = (typical * out["volume"]).groupby(session).cumsum()
    cumulative_volume = out["volume"].groupby(session).cumsum()
    out["session_vwap"] = cumulative_pv / cumulative_volume.replace(0, np.nan)
    out["vwap_distance"] = out["close"] - out["session_vwap"]
    out["vwap_distance_atr"] = out["vwap_distance"] / out["atr"]
    out["vwap_distance_z"] = out["vwap_distance"] / out["vwap_distance"].rolling(cfg.vwap_z_window).std().replace(0, np.nan)

    out["prior_range_high"] = out["high"].shift(1).rolling(cfg.range_lookback).max()
    out["prior_range_low"] = out["low"].shift(1).rolling(cfg.range_lookback).min()
    out["prior_break_high"] = out["high"].shift(1).rolling(cfg.continuation_lookback).max()
    out["prior_break_low"] = out["low"].shift(1).rolling(cfg.continuation_lookback).min()
    out["distance_fast_atr"] = (out["close"] - out["ema_fast"]) / out["atr"]
    out["bar_range"] = (out["high"] - out["low"]).replace(0, np.nan)
    out["body"] = (out["close"] - out["open"]).abs()
    out["upper_wick_ratio"] = (out["high"] - out[["open", "close"]].max(axis=1)) / out["bar_range"]
    out["lower_wick_ratio"] = (out[["open", "close"]].min(axis=1) - out["low"]) / out["bar_range"]
    out["body_ratio"] = out["body"] / out["bar_range"]
    out["recent_low_touched_fast"] = out["low"].rolling(10).min().shift(1) <= out["ema_fast"].shift(1)
    out["recent_high_touched_fast"] = out["high"].rolling(10).max().shift(1) >= out["ema_fast"].shift(1)

    minutes = out.index.hour * 60 + out.index.minute
    hour = out.index.hour
    funding_minute = ((hour % 8) == 0) & ((out.index.minute < cfg.no_trade_funding_minutes) | (out.index.minute >= 60 - cfg.no_trade_funding_minutes))
    liquid_hours = (hour >= cfg.liquid_start_hour_utc) & (hour <= cfg.liquid_end_hour_utc)
    out["session_ok"] = liquid_hours & ~funding_minute
    out["minute_of_day"] = minutes
    return out


def add_family_signals(df: pd.DataFrame, family: str, cfg: LabConfig) -> pd.DataFrame:
    out = df.copy()
    market_ok = (
        out["session_ok"]
        & out["atr_pct"].between(cfg.min_atr_pct, cfg.max_atr_pct)
        & out["session_vwap"].notna()
        & out["volume_ratio"].notna()
    )

    if family == "mean_reversion":
        long_setup = (
            market_ok
            & (out["vwap_distance_atr"] <= -cfg.mr_min_abs_vwap_atr)
            & (out["vwap_distance_z"] <= -cfg.mr_min_abs_vwap_z)
            & (out["rsi"] <= cfg.mr_long_rsi_max)
            & (out["volume_ratio"] >= cfg.mr_min_volume_ratio)
            & (out["lower_wick_ratio"] >= 0.35)
            & (out["close"] > out["open"])
            & (out["close"] > out["close"].shift(1))
        )
        short_setup = (
            market_ok
            & (out["vwap_distance_atr"] >= cfg.mr_min_abs_vwap_atr)
            & (out["vwap_distance_z"] >= cfg.mr_min_abs_vwap_z)
            & (out["rsi"] >= cfg.mr_short_rsi_min)
            & (out["volume_ratio"] >= cfg.mr_min_volume_ratio)
            & (out["upper_wick_ratio"] >= 0.35)
            & (out["close"] < out["open"])
            & (out["close"] < out["close"].shift(1))
        )
    elif family == "liquidity_sweep":
        sweep_high_atr = (out["high"] - out["prior_range_high"]) / out["atr"]
        sweep_low_atr = (out["prior_range_low"] - out["low"]) / out["atr"]
        high_reclaim_atr = (out["prior_range_high"] - out["close"]) / out["atr"]
        low_reclaim_atr = (out["close"] - out["prior_range_low"]) / out["atr"]
        long_setup = (
            market_ok
            & (out["low"] < out["prior_range_low"])
            & (out["close"] > out["prior_range_low"])
            & (sweep_low_atr >= cfg.sweep_min_atr)
            & (low_reclaim_atr >= cfg.sweep_min_reclaim_atr)
            & (out["lower_wick_ratio"] >= cfg.sweep_min_wick_ratio)
            & (out["volume_ratio"] >= cfg.sweep_min_volume_ratio)
            & (out["rsi"] <= 46)
        )
        short_setup = (
            market_ok
            & (out["high"] > out["prior_range_high"])
            & (out["close"] < out["prior_range_high"])
            & (sweep_high_atr >= cfg.sweep_min_atr)
            & (high_reclaim_atr >= cfg.sweep_min_reclaim_atr)
            & (out["upper_wick_ratio"] >= cfg.sweep_min_wick_ratio)
            & (out["volume_ratio"] >= cfg.sweep_min_volume_ratio)
            & (out["rsi"] >= 54)
        )
    elif family == "selective_continuation":
        break_high_atr = (out["close"] - out["prior_break_high"]) / out["atr"]
        break_low_atr = (out["prior_break_low"] - out["close"]) / out["atr"]
        long_setup = (
            market_ok
            & (out["atr_ratio"] >= cfg.cont_min_atr_ratio)
            & (out["volume_ratio"] >= cfg.cont_min_volume_ratio)
            & (out["ema_fast"] > out["ema_slow"])
            & (out["close"] > out["session_vwap"])
            & (break_high_atr >= cfg.cont_min_break_atr)
            & out["rsi"].between(55, 70)
            & out["distance_fast_atr"].between(0, cfg.cont_max_distance_fast_atr)
            & (out["close"] > out["open"])
        )
        short_setup = (
            market_ok
            & (out["atr_ratio"] >= cfg.cont_min_atr_ratio)
            & (out["volume_ratio"] >= cfg.cont_min_volume_ratio)
            & (out["ema_fast"] < out["ema_slow"])
            & (out["close"] < out["session_vwap"])
            & (break_low_atr >= cfg.cont_min_break_atr)
            & out["rsi"].between(30, 45)
            & out["distance_fast_atr"].between(-cfg.cont_max_distance_fast_atr, 0)
            & (out["close"] < out["open"])
        )
    else:
        raise ValueError(f"Unknown family: {family}")

    out["long_signal"] = long_setup.astype(np.int8)
    out["short_signal"] = short_setup.astype(np.int8) * -1
    out = apply_daily_signal_cap(out, cfg.max_signals_per_day)
    out["long_entry"] = out["long_signal"].shift(1).fillna(0).astype(np.int8)
    out["short_entry"] = out["short_signal"].shift(1).fillna(0).astype(np.int8)
    return out


def apply_daily_signal_cap(df: pd.DataFrame, max_signals_per_day: int) -> pd.DataFrame:
    out = df.copy()
    candidate = (out["long_signal"] == 1) | (out["short_signal"] == -1)
    session = out.index.floor("D")
    rank = candidate.groupby(session).cumsum()
    blocked = candidate & (rank > max_signals_per_day)
    out.loc[blocked, ["long_signal", "short_signal"]] = 0
    return out


def risk_for_fee_model(initial_equity: float, fee_model: str) -> core.Risk1mConfig:
    if fee_model == "taker":
        return core.Risk1mConfig(initial_equity=initial_equity, taker_fee_rate=0.0004, slippage_rate=0.00025)
    if fee_model == "maker_like":
        return core.Risk1mConfig(initial_equity=initial_equity, taker_fee_rate=0.0002, slippage_rate=0.00005)
    raise ValueError(f"Unknown fee model: {fee_model}")


def combo_key(family: str, fee_model: str, variant: str) -> str:
    return f"{family}__{fee_model}__{variant}"


def demo_gate(summary: pd.DataFrame, is_oos: pd.DataFrame, walk_forward: pd.DataFrame) -> dict[str, Any]:
    decisions = []
    for _, row in summary.iterrows():
        mask = (
            (is_oos["family"] == row["family"])
            & (is_oos["fee_model"] == row["fee_model"])
            & (is_oos["variant"] == row["variant"])
            & (is_oos["sample"] == "out_of_sample")
        )
        oos = is_oos[mask]
        wf = walk_forward[
            (walk_forward["family"] == row["family"])
            & (walk_forward["fee_model"] == row["fee_model"])
            & (walk_forward["variant"] == row["variant"])
        ]
        oos_pf = float(oos["profit_factor"].iloc[0]) if not oos.empty else np.nan
        oos_expectancy = float(oos["expectancy"].iloc[0]) if not oos.empty else np.nan
        positive_wf_years = int((wf["test_expectancy"] > 0).sum()) if not wf.empty else 0
        min_wf_pf = float(wf["test_profit_factor"].min()) if not wf.empty else np.nan
        passes = (
            float(row["profit_factor"]) > 1.10
            and float(row["expectancy"]) > 0
            and oos_pf > 1.0
            and oos_expectancy >= 0
            and float(row["max_drawdown_pct"]) <= 20
            and row["frequency_class"] == "reasonable"
            and positive_wf_years >= 3
            and min_wf_pf > 0.85
        )
        decisions.append(
            {
                "family": row["family"],
                "fee_model": row["fee_model"],
                "variant": row["variant"],
                "passes_demo_gate": bool(passes),
                "profit_factor": float(row["profit_factor"]),
                "expectancy": float(row["expectancy"]),
                "oos_profit_factor": oos_pf,
                "oos_expectancy": oos_expectancy,
                "max_drawdown_pct": float(row["max_drawdown_pct"]),
                "frequency_class": row["frequency_class"],
                "positive_walk_forward_years": positive_wf_years,
                "min_walk_forward_pf": min_wf_pf,
            }
        )
    return {"allowed": any(item["passes_demo_gate"] for item in decisions), "candidates": decisions}


def plot_equity(equity_wide: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for col in equity_wide.columns:
        if col != "close":
            equity_wide[col].plot(ax=ax, alpha=0.65, linewidth=1.0, label=col)
    ax.set_title("BTCUSDT USD-M 1m Hypothesis Lab Equity")
    ax.set_ylabel("Equity USDT")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    is_oos: pd.DataFrame,
    walk_forward: pd.DataFrame,
    gate: dict[str, Any],
    output_dir: Path,
    start_month: str,
    end_month: str,
) -> None:
    lines = [
        "# BTCUSDT USD-M 1m Hypothesis Lab",
        "",
        f"Period: {start_month} through {end_month}.",
        "Execution timeframe: 1m only.",
        "Families: mean_reversion, liquidity_sweep, selective_continuation.",
        "Fee assumptions: taker and maker_like. maker_like is optimistic and not a fill guarantee.",
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
        "Research output only. Demo 1m remains blocked unless Demo Gate allowed=true.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_month, end_month = core.five_year_month_window(args.start_month, args.end_month)
    kline_path, funding_path = core.ensure_data(start_month, end_month, args.data_dir, args.force_download)
    raw = core.load_ohlcv_csv(kline_path, SYMBOL)
    funding = core.load_funding_csv(funding_path)

    lab_cfg = LabConfig()
    common = add_common_indicators(raw, lab_cfg)

    summary_rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    is_oos_frames: list[pd.DataFrame] = []
    walk_forward_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    equity_wide = pd.DataFrame({"close": raw["close"]})

    for family in FAMILIES:
        logger.info("Building signals for family=%s", family)
        family_data = add_family_signals(common, family, lab_cfg)
        for fee_model in FEE_MODELS:
            risk = risk_for_fee_model(args.initial_equity, fee_model)
            for variant in VARIANTS:
                logger.info("Running family=%s fee_model=%s variant=%s", family, fee_model, variant)
                equity, trades = core.run_backtest(family_data, funding, variant, risk)
                key = combo_key(family, fee_model, variant)
                equity_wide[key] = equity["equity"].astype("float32")
                if not trades.empty:
                    trades = trades.copy()
                    trades.insert(0, "family", family)
                    trades.insert(1, "fee_model", fee_model)
                    trades["variant"] = variant
                    trade_frames.append(trades)
                m = core.metrics(equity, trades, risk.initial_equity)
                summary_rows.append(
                    {
                        "symbol": SYMBOL,
                        "interval": INTERVAL,
                        "family": family,
                        "fee_model": fee_model,
                        "variant": variant,
                        "start": str(raw.index[0]),
                        "end": str(raw.index[-1]),
                        "rows": len(raw),
                        "frequency_class": core.classify_frequency(m["trades_per_year"]),
                        **m,
                    }
                )
                yearly = core.segment_metrics(equity, trades, "Y", "year")
                yearly.insert(0, "variant", variant)
                yearly.insert(0, "fee_model", fee_model)
                yearly.insert(0, "family", family)
                yearly_frames.append(yearly)
                sample = core.is_oos_metrics(equity, trades)
                sample.insert(0, "variant", variant)
                sample.insert(0, "fee_model", fee_model)
                sample.insert(0, "family", family)
                is_oos_frames.append(sample)
                forward = core.walk_forward_metrics(equity, trades)
                forward.insert(0, "variant", variant)
                forward.insert(0, "fee_model", fee_model)
                forward.insert(0, "family", family)
                walk_forward_frames.append(forward)

    summary = pd.DataFrame(summary_rows)
    yearly_all = pd.concat(yearly_frames, ignore_index=True)
    is_oos_all = pd.concat(is_oos_frames, ignore_index=True)
    walk_forward_all = pd.concat(walk_forward_frames, ignore_index=True)
    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    gate = demo_gate(summary, is_oos_all, walk_forward_all)

    summary.to_csv(args.output_dir / "summary.csv", index=False)
    yearly_all.to_csv(args.output_dir / "yearly.csv", index=False)
    is_oos_all.to_csv(args.output_dir / "is_oos.csv", index=False)
    walk_forward_all.to_csv(args.output_dir / "walk_forward.csv", index=False)
    trades_all.to_csv(args.output_dir / "trades.csv", index=False)
    equity_wide.to_csv(args.output_dir / "equity.csv")
    (args.output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps({"lab": asdict(lab_cfg), "demo_gate": gate}, indent=2), encoding="utf-8")
    plot_equity(equity_wide.drop(columns=["close"]), args.output_dir / "equity.png")
    if args.plot:
        plot_equity(equity_wide.drop(columns=["close"]), args.output_dir / "equity_plot_requested.png")
    write_report(summary, yearly_all, is_oos_all, walk_forward_all, gate, args.output_dir, start_month, end_month)

    print(summary.to_string(index=False))
    print("\nTop by profit factor:")
    print(summary.sort_values("profit_factor", ascending=False)[["family", "fee_model", "variant", "total_return_pct", "max_drawdown_pct", "trade_count", "trades_per_year", "profit_factor", "expectancy", "frequency_class"]].head(12).to_string(index=False))
    print("\nDemo gate:")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
