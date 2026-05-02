from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from btcusdt_usdm_futures_bot import (
    BacktestConfig,
    BinanceFuturesClient,
    FuturesRiskConfig,
    RuntimeConfig,
    StrategyConfig,
    calculate_metrics,
    default_end_month,
    download_funding_rates,
    download_usdm_klines,
    load_funding_csv,
    load_ohlcv_csv,
    run_futures_backtest,
)


logger = logging.getLogger("run_usdm_experiments")


@dataclass(frozen=True)
class ExperimentSpec:
    timeframe: str
    sample: str
    variant: str
    start: Optional[str]
    end: Optional[str]


VARIANTS = {
    "both": StrategyConfig(allow_long=True, allow_short=True),
    "long_only": StrategyConfig(allow_long=True, allow_short=False),
    "short_only": StrategyConfig(allow_long=False, allow_short=True),
}


def ensure_data(symbol: str, interval: str, start_month: str, end_month: str, data_dir: Path) -> tuple[Path, Path]:
    runtime = RuntimeConfig(symbol=symbol, interval=interval, testnet=False)
    client = BinanceFuturesClient(runtime)

    kline_path = data_dir / interval / f"{symbol}_USDM_{interval}.csv"
    if not kline_path.exists():
        kline_path = download_usdm_klines(symbol, interval, start_month, end_month, data_dir / interval)

    data = load_ohlcv_csv(kline_path, symbol)
    funding_path = data_dir / f"{symbol}_funding.csv"
    if not funding_path.exists():
        funding_path = download_funding_rates(client, symbol, data.index[0], data.index[-1], data_dir)
    return kline_path, funding_path


def filter_sample(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    out = df
    if start:
        out = out[out.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        out = out[out.index < pd.Timestamp(end, tz="UTC")]
    return out.copy()


def run_experiment(
    symbol: str,
    data: pd.DataFrame,
    funding: pd.DataFrame,
    spec: ExperimentSpec,
    risk: FuturesRiskConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    sample_data = filter_sample(data, spec.start, spec.end)
    if sample_data.empty:
        raise ValueError(f"Empty sample for {spec}")

    config = BacktestConfig(strategy=VARIANTS[spec.variant], risk=risk)
    equity, trades = run_futures_backtest(sample_data, funding, config)
    metrics = calculate_metrics(equity, trades, risk.initial_equity)
    breakdown = build_exit_breakdown(trades)
    side_breakdown = build_side_breakdown(trades)

    row: dict[str, object] = {
        "symbol": symbol,
        "timeframe": spec.timeframe,
        "sample": spec.sample,
        "variant": spec.variant,
        "start": str(sample_data.index[0]),
        "end": str(sample_data.index[-1]),
        **metrics,
        "exit_breakdown": json.dumps(breakdown, sort_keys=True),
        "side_breakdown": json.dumps(side_breakdown, sort_keys=True),
    }
    return row, trades


def build_exit_breakdown(trades: pd.DataFrame) -> dict[str, dict[str, float]]:
    if trades.empty:
        return {}
    grouped = trades.groupby("exit_reason")["net_pnl"].agg(["count", "sum", "mean"])
    return {
        str(index): {
            "count": float(row["count"]),
            "sum": float(row["sum"]),
            "mean": float(row["mean"]),
        }
        for index, row in grouped.iterrows()
    }


def build_side_breakdown(trades: pd.DataFrame) -> dict[str, dict[str, float]]:
    if trades.empty:
        return {}
    grouped = trades.groupby("side")["net_pnl"].agg(["count", "sum", "mean"])
    return {
        str(index): {
            "count": float(row["count"]),
            "sum": float(row["sum"]),
            "mean": float(row["mean"]),
        }
        for index, row in grouped.iterrows()
    }


def format_metric(value: object) -> str:
    if isinstance(value, (int, float, np.floating)):
        if np.isinf(value):
            return "inf"
        return f"{float(value):.4f}"
    return str(value)


def write_markdown_report(results: pd.DataFrame, output_path: Path) -> None:
    metric_cols = [
        "timeframe",
        "sample",
        "variant",
        "total_return_pct",
        "max_drawdown_pct",
        "trade_count",
        "profit_factor",
        "expectancy",
        "win_rate_pct",
        "avg_win",
        "avg_loss",
        "sharpe_simplified",
        "exposure_time_pct",
        "avg_trade_duration_bars",
    ]
    lines = ["# USD-M Futures Experiment Report", ""]
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| " + " | ".join(metric_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(metric_cols)) + " |")
    for _, row in results[metric_cols].iterrows():
        lines.append("| " + " | ".join(format_metric(row[col]) for col in metric_cols) + " |")
    lines.append("")
    lines.append("## Exit Breakdowns")
    lines.append("")
    for _, row in results.iterrows():
        lines.append(f"### {row['timeframe']} {row['sample']} {row['variant']}")
        lines.append("")
        lines.append(f"- Exit breakdown: `{row['exit_breakdown']}`")
        lines.append(f"- Side breakdown: `{row['side_breakdown']}`")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run structured USD-M futures validation experiments")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2024-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data_usdm_experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("usdm_experiment_output"))
    parser.add_argument("--is-end", default="2025-07-01", help="Exclusive end timestamp for in-sample; OOS starts here")
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--leverage", type=int, default=2)
    parser.add_argument("--risk-per-trade", type=float, default=0.005)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = build_arg_parser().parse_args()
    end_month = args.end or default_end_month()
    risk = FuturesRiskConfig(
        initial_equity=args.initial_equity,
        leverage=args.leverage,
        risk_per_trade=args.risk_per_trade,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    all_trade_frames: list[pd.DataFrame] = []

    for timeframe in ["1h", "4h", "30m", "15m"]:
        logger.info("Preparing data for %s", timeframe)
        kline_path, funding_path = ensure_data(args.symbol, timeframe, args.start, end_month, args.data_dir)
        data = load_ohlcv_csv(kline_path, args.symbol)
        funding = load_funding_csv(funding_path)

        specs = []
        for sample, start, end in [
            ("full", None, None),
            ("in_sample", None, args.is_end),
            ("out_of_sample", args.is_end, None),
        ]:
            for variant in ["both", "long_only", "short_only"]:
                specs.append(ExperimentSpec(timeframe, sample, variant, start, end))

        for spec in specs:
            logger.info("Running %s %s %s", spec.timeframe, spec.sample, spec.variant)
            row, trades = run_experiment(args.symbol, data, funding, spec, risk)
            all_rows.append(row)
            if not trades.empty:
                trades = trades.copy()
                trades["timeframe"] = spec.timeframe
                trades["sample"] = spec.sample
                trades["variant"] = spec.variant
                all_trade_frames.append(trades)

    results = pd.DataFrame(all_rows)
    results_path = args.output_dir / "summary.csv"
    results.to_csv(results_path, index=False)
    (args.output_dir / "summary.json").write_text(results.to_json(orient="records", indent=2), encoding="utf-8")
    write_markdown_report(results, args.output_dir / "report.md")

    if all_trade_frames:
        pd.concat(all_trade_frames, ignore_index=True).to_csv(args.output_dir / "all_trades.csv", index=False)

    print(results[
        [
            "timeframe",
            "sample",
            "variant",
            "total_return_pct",
            "max_drawdown_pct",
            "trade_count",
            "profit_factor",
            "expectancy",
            "win_rate_pct",
            "sharpe_simplified",
            "exposure_time_pct",
        ]
    ].to_string(index=False))
    print(f"\nSaved experiment report to {args.output_dir}")


if __name__ == "__main__":
    main()
