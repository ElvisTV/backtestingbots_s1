from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

import btcusdt_usdm_1m_hypothesis_lab as lab
import btcusdt_usdm_1m_research_backtester as core
from btcusdt_usdm_futures_bot import (
    BINANCE_DATA_BASE_URL,
    BINANCE_KLINE_COLUMNS,
    REQUIRED_COLUMNS,
    BinanceFuturesClient,
    RuntimeConfig,
    default_end_month,
    download_funding_rates,
    download_kline_zip,
    load_funding_csv,
    load_ohlcv_csv,
    parse_timestamp_column,
)


SYMBOL = "BTCUSDT"
INTERVAL = "1m"
API_BASE_URL = "https://fapi.binance.com"
MONTHLY_ARCHIVE_START = "2020-01"
DEFAULT_EXISTING_TAIL = Path("data_usdm_1m_research") / INTERVAL / f"{SYMBOL}_USDM_{INTERVAL}.csv"
logger = logging.getLogger("btcusdt_usdm_full_history_research")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BTCUSDT USD-M perpetual full-history 1m research backtester"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data_usdm_full_history"))
    parser.add_argument("--output-dir", type=Path, default=Path("usdm_1m_full_history_output"))
    parser.add_argument("--initial-equity", type=float, default=2_000.0)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--ignore-existing-tail", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def request_json(url: str, timeout: int = 30) -> Any:
    request = Request(url, headers={"User-Agent": "btcusdt-usdm-full-history-research/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def discover_first_kline(symbol: str, interval: str) -> tuple[pd.Timestamp, dict[str, Any]]:
    query = urlencode({"symbol": symbol, "interval": interval, "startTime": 0, "limit": 1})
    url = f"{API_BASE_URL}/fapi/v1/klines?{query}"
    payload = request_json(url)
    if not payload:
        raise RuntimeError(f"No first kline returned by {url}")
    first = payload[0]
    timestamp = pd.to_datetime(int(first[0]), unit="ms", utc=True)
    return timestamp, {"endpoint": url, "first_kline": first}


def completed_month_end(end_month: str) -> pd.Timestamp:
    return pd.Timestamp(pd.Period(end_month, freq="M").end_time).tz_localize("UTC").floor("min")


def read_last_data_line(path: Path) -> str:
    with path.open("rb") as file:
        file.seek(0, 2)
        position = file.tell()
        chunk = b""
        while position > 0:
            step = min(4096, position)
            position -= step
            file.seek(position)
            chunk = file.read(step) + chunk
            lines = chunk.splitlines()
            if len(lines) > 1:
                return lines[-1].decode("utf-8", errors="replace")
    return ""


def cached_coverage(path: Path) -> Optional[tuple[pd.Timestamp, pd.Timestamp, int]]:
    if not path.exists():
        return None
    try:
        first_row = pd.read_csv(path, usecols=["timestamp"], nrows=1)
        if first_row.empty:
            return None
        first_ts = parse_timestamp_column(first_row["timestamp"]).iloc[0]
        last_line = read_last_data_line(path)
        if not last_line:
            return None
        last_value = last_line.split(",", 1)[0]
        last_ts = parse_timestamp_column(pd.Series([last_value])).iloc[0]
        row_count = sum(1 for _ in path.open("rb")) - 1
        return first_ts, last_ts, row_count
    except Exception as exc:  # noqa: BLE001 - cache validation must fail closed.
        logger.warning("Could not validate cache %s: %s", path, exc)
        return None


def normalize_kline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[REQUIRED_COLUMNS].copy()
    if pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    else:
        out["timestamp"] = parse_timestamp_column(out["timestamp"])
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=REQUIRED_COLUMNS)


def api_klines_to_frame(rows: list[list[Any]]) -> pd.DataFrame:
    raw = pd.DataFrame(rows, columns=BINANCE_KLINE_COLUMNS)
    return normalize_kline_frame(raw.rename(columns={"open_time": "timestamp"})[REQUIRED_COLUMNS])


def download_klines_from_api(
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[list[Any]] = []

    while start_ms <= end_ms:
        query = urlencode(
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1500,
            }
        )
        url = f"{API_BASE_URL}/fapi/v1/klines?{query}"
        batch = request_json(url, timeout=30)
        if not batch:
            break
        rows.extend(batch)
        next_start = int(batch[-1][0]) + 60_000
        if next_start <= start_ms:
            break
        start_ms = next_start
        time.sleep(0.03)

    if not rows:
        raise RuntimeError(f"No API klines downloaded for {symbol} {interval} {start}..{end}")
    frame = api_klines_to_frame(rows)
    logger.info("Downloaded %s pre-archive rows from Binance Futures API", len(frame))
    return frame


def download_monthly_archives(
    symbol: str,
    interval: str,
    start_month: str,
    end_month: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for month in pd.period_range(start=start_month, end=end_month, freq="M"):
        url = (
            f"{BINANCE_DATA_BASE_URL}/futures/um/monthly/klines/"
            f"{symbol}/{interval}/{symbol}-{interval}-{month.strftime('%Y-%m')}.zip"
        )
        try:
            logger.info("Downloading monthly archive %s", url)
            frames.append(normalize_kline_frame(download_kline_zip(url)))
        except HTTPError as exc:
            if exc.code == 404:
                logger.warning("Missing monthly archive, skipping: %s", url)
                continue
            raise
    if not frames:
        raise RuntimeError(f"No monthly archives downloaded for {symbol} {interval} {start_month}..{end_month}")
    return pd.concat(frames, ignore_index=True)


def load_existing_tail(end_month: str, ignore_tail: bool) -> Optional[pd.DataFrame]:
    if ignore_tail or not DEFAULT_EXISTING_TAIL.exists():
        return None
    data = load_ohlcv_csv(DEFAULT_EXISTING_TAIL, SYMBOL)
    first_month = str(data.index[0].to_period("M"))
    last_month = str(data.index[-1].to_period("M"))
    if first_month <= "2021-05" and last_month >= end_month:
        logger.info("Reusing existing cached tail %s..%s from %s", first_month, last_month, DEFAULT_EXISTING_TAIL)
        return data.reset_index()[REQUIRED_COLUMNS]
    logger.info("Existing tail coverage %s..%s is not enough for requested end month %s", first_month, last_month, end_month)
    return None


def write_full_cache(
    frames: list[pd.DataFrame],
    path: Path,
    first_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict[str, Any]:
    combined = pd.concat(frames, ignore_index=True)
    combined = normalize_kline_frame(combined)
    combined = combined[(combined["timestamp"] >= first_ts) & (combined["timestamp"] <= end_ts)]
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp")

    expected_rows = int(((end_ts - first_ts).total_seconds() // 60) + 1)
    missing_rows = max(expected_rows - len(combined), 0)
    max_gap = combined["timestamp"].diff().max()
    max_gap_minutes = float(max_gap.total_seconds() / 60) if pd.notna(max_gap) else 0.0

    path.parent.mkdir(parents=True, exist_ok=True)
    out = combined.copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    out.to_csv(path, index=False)

    return {
        "path": str(path),
        "rows": int(len(combined)),
        "first_timestamp": str(combined["timestamp"].iloc[0]),
        "last_timestamp": str(combined["timestamp"].iloc[-1]),
        "expected_minute_rows": expected_rows,
        "missing_minute_rows": int(missing_rows),
        "max_gap_minutes": max_gap_minutes,
    }


def build_or_load_full_history(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    args.data_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.data_dir / INTERVAL / f"{SYMBOL}_USDM_{INTERVAL}_full.csv"
    first_ts, first_meta = discover_first_kline(SYMBOL, INTERVAL)
    end_month = default_end_month()
    end_ts = completed_month_end(end_month)

    cache = cached_coverage(full_path)
    if cache and not args.force_rebuild:
        cache_first, cache_last, rows = cache
        if cache_first <= first_ts and cache_last >= end_ts:
            logger.info("Full history cache is valid: %s rows %s..%s", rows, cache_first, cache_last)
            metadata = {
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "first_kline_api": first_meta,
                "range_start": str(first_ts),
                "range_end": str(end_ts),
                "latest_completed_month": end_month,
                "cache": {"path": str(full_path), "rows": rows, "first_timestamp": str(cache_first), "last_timestamp": str(cache_last)},
                "sources": [
                    "Binance Futures API /fapi/v1/klines for 2019-09-08 through 2019-12-31",
                    "Binance public data monthly USD-M futures archives from 2020-01 onward",
                ],
                "spot_proxy_used": False,
            }
            funding_path = ensure_funding(args.data_dir, first_ts, end_ts, args.force_rebuild)
            return full_path, funding_path, metadata

    frames: list[pd.DataFrame] = []
    pre_archive_end = completed_month_end("2019-12")
    frames.append(download_klines_from_api(SYMBOL, INTERVAL, first_ts, pre_archive_end))

    tail = load_existing_tail(end_month, args.ignore_existing_tail)
    archive_end_month = end_month
    tail_start_month = None
    if tail is not None:
        tail_ts = pd.to_datetime(tail["timestamp"], utc=True, errors="coerce")
        tail_start_month = str(tail_ts.min().to_period("M"))
        archive_end_month = str((pd.Period(tail_start_month, freq="M") - 1).strftime("%Y-%m"))

    if pd.Period(MONTHLY_ARCHIVE_START, freq="M") <= pd.Period(archive_end_month, freq="M"):
        frames.append(download_monthly_archives(SYMBOL, INTERVAL, MONTHLY_ARCHIVE_START, archive_end_month))
    if tail is not None:
        frames.append(normalize_kline_frame(tail))

    cache_meta = write_full_cache(frames, full_path, first_ts, end_ts)
    metadata = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "first_kline_api": first_meta,
        "range_start": str(first_ts),
        "range_end": str(end_ts),
        "latest_completed_month": end_month,
        "monthly_archive_start": MONTHLY_ARCHIVE_START,
        "monthly_archive_end": archive_end_month,
        "existing_tail_reused_from": str(DEFAULT_EXISTING_TAIL) if tail is not None else None,
        "existing_tail_start_month": tail_start_month,
        "cache": cache_meta,
        "sources": [
            "Binance Futures API /fapi/v1/klines for 2019-09-08 through 2019-12-31",
            "Binance public data monthly USD-M futures archives from 2020-01 onward",
            "Local cached official 1m futures tail reused when coverage was sufficient",
        ],
        "spot_proxy_used": False,
    }
    (args.data_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    funding_path = ensure_funding(args.data_dir, first_ts, end_ts, args.force_rebuild)
    return full_path, funding_path, metadata


def ensure_funding(data_dir: Path, start: pd.Timestamp, end: pd.Timestamp, force: bool) -> Path:
    funding_path = data_dir / f"{SYMBOL}_funding.csv"
    should_refresh = force or not funding_path.exists()
    if funding_path.exists() and not should_refresh:
        funding = load_funding_csv(funding_path)
        should_refresh = funding.empty or funding.index[0] > start or funding.index[-1] < end
    if should_refresh:
        client = BinanceFuturesClient(RuntimeConfig(symbol=SYMBOL, interval=INTERVAL, testnet=False, dry_run=True))
        return download_funding_rates(client, SYMBOL, start, end, data_dir)
    return funding_path


def run_full_history_lab(
    raw: pd.DataFrame,
    funding: pd.DataFrame,
    initial_equity: float,
    output_dir: Path,
    metadata: dict[str, Any],
    plot_requested: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lab_cfg = lab.LabConfig()
    common = lab.add_common_indicators(raw, lab_cfg)

    summary_rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    is_oos_frames: list[pd.DataFrame] = []
    walk_forward_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    equity_wide = pd.DataFrame({"close": raw["close"]})

    for family in lab.FAMILIES:
        logger.info("Building signals for family=%s", family)
        family_data = lab.add_family_signals(common, family, lab_cfg)
        for fee_model in lab.FEE_MODELS:
            risk = lab.risk_for_fee_model(initial_equity, fee_model)
            for variant in lab.VARIANTS:
                logger.info("Running family=%s fee_model=%s variant=%s", family, fee_model, variant)
                equity, trades = core.run_backtest(family_data, funding, variant, risk)
                key = lab.combo_key(family, fee_model, variant)
                equity_wide[key] = equity["equity"].astype("float32")
                if not trades.empty:
                    trades = trades.copy()
                    trades.insert(0, "family", family)
                    trades.insert(1, "fee_model", fee_model)
                    trades["variant"] = variant
                    trade_frames.append(trades)
                metrics = core.metrics(equity, trades, risk.initial_equity)
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
                        "frequency_class": core.classify_frequency(metrics["trades_per_year"]),
                        **metrics,
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
    gate = lab.demo_gate(summary, is_oos_all, walk_forward_all)

    summary.to_csv(output_dir / "summary.csv", index=False)
    yearly_all.to_csv(output_dir / "yearly.csv", index=False)
    is_oos_all.to_csv(output_dir / "is_oos.csv", index=False)
    walk_forward_all.to_csv(output_dir / "walk_forward.csv", index=False)
    trades_all.to_csv(output_dir / "trades.csv", index=False)
    equity_wide.to_csv(output_dir / "equity.csv")
    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, default=str), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (output_dir / "config.json").write_text(
        json.dumps({"lab": asdict(lab_cfg), "demo_gate": gate, "metadata": metadata}, indent=2, default=str),
        encoding="utf-8",
    )
    lab.plot_equity(equity_wide.drop(columns=["close"]), output_dir / "equity.png")
    if plot_requested:
        lab.plot_equity(equity_wide.drop(columns=["close"]), output_dir / "equity_plot_requested.png")
    lab.write_report(
        summary,
        yearly_all,
        is_oos_all,
        walk_forward_all,
        gate,
        output_dir,
        str(raw.index[0]),
        str(raw.index[-1]),
    )

    print(summary.to_string(index=False))
    print("\nTop by profit factor:")
    columns = [
        "family",
        "fee_model",
        "variant",
        "total_return_pct",
        "max_drawdown_pct",
        "trade_count",
        "trades_per_year",
        "profit_factor",
        "expectancy",
        "frequency_class",
    ]
    print(summary.sort_values("profit_factor", ascending=False)[columns].head(12).to_string(index=False))
    print("\nDemo gate:")
    print(json.dumps(gate, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = parse_args()
    kline_path, funding_path, metadata = build_or_load_full_history(args)
    raw = load_ohlcv_csv(kline_path, SYMBOL)
    funding = load_funding_csv(funding_path)
    logger.info("Running full-history research on %s rows: %s..%s", len(raw), raw.index[0], raw.index[-1])
    run_full_history_lab(raw, funding, args.initial_equity, args.output_dir, metadata, args.plot)


if __name__ == "__main__":
    main()
