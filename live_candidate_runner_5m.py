from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

import btcusdt_usdm_5m_research_backtester as research5m
import live_candidate_runner as base


SYMBOL = "BTCUSDT"
INTERVAL = "5m"
REQUIRED_SIGNAL = -1
BOOTSTRAP_CANDLE_LIMIT = 1200

base.SYMBOL = SYMBOL
base.INTERVAL = INTERVAL
base.REQUIRED_SIGNAL = REQUIRED_SIGNAL
base.BOOTSTRAP_CANDLE_LIMIT = BOOTSTRAP_CANDLE_LIMIT

logger = logging.getLogger("live_candidate_5m")


class SessionComplete(Exception):
    pass


def evaluate_5m_short_signal(candles: list[dict[str, Any]]) -> int:
    """Evaluate the current best 5m research baseline: short-only.

    This is not a profitability approval. It lets the demo runner exercise real
    signal -> decision -> order -> reconciliation flow using the same 5m candle
    logic as the multi-year research baseline.
    """
    if len(candles) < 900:
        return 0
    df = pd.DataFrame(candles)
    if df.empty:
        return 0
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).set_index("timestamp")
    if len(df) < 900:
        return 0
    config = research5m.ResearchStrategyConfig()
    enriched = research5m.add_signals(research5m.add_indicators(df, config), config)
    signal = int(enriched["short_entry"].iloc[-1])
    return REQUIRED_SIGNAL if signal == REQUIRED_SIGNAL else 0


class LiveCandidateRunner5m(base.LiveCandidateRunner):
    max_runtime_seconds: int = 0
    max_closed_candles: int = 0
    closed_candles_processed: int = 0

    async def run(self) -> None:
        self.validate_static_guardrails()
        self.private_api_preflight_or_abort()
        await self.reconcile_or_abort()
        self.prepare_account_settings_or_abort()
        self.bootstrap_historical_candles()

        if self.config.mode == "test-order":
            self.submit_test_order_or_abort()
            return

        await self.start_user_data_stream()
        tasks = [
            asyncio.create_task(self.market_data_loop()),
            asyncio.create_task(self.user_data_loop()),
            asyncio.create_task(self.listen_key_keepalive_loop()),
            asyncio.create_task(self.stream_health_loop()),
        ]
        if self.max_runtime_seconds > 0:
            tasks.append(asyncio.create_task(self.runtime_limit_loop()))
        try:
            await asyncio.gather(*tasks)
        except SessionComplete as exc:
            self.event_logger.emit(
                "session_complete_exit",
                reason=str(exc),
                closed_candles_processed=self.closed_candles_processed,
                max_closed_candles=self.max_closed_candles,
                max_runtime_seconds=self.max_runtime_seconds,
            )
        finally:
            for task in tasks:
                task.cancel()
            await self.close_listen_key()

    async def runtime_limit_loop(self) -> None:
        await asyncio.sleep(self.max_runtime_seconds)
        raise SessionComplete("max_runtime_seconds_reached")

    def validate_static_guardrails(self) -> None:
        if self.runtime.symbol != SYMBOL or self.runtime.interval != INTERVAL:
            raise base.GuardrailViolation("Runner is hard-restricted to BTCUSDT 5m")
        if self.config.mode not in {"observe", "test-order", "demo-supervised"}:
            raise base.GuardrailViolation("5m runner only supports observe, test-order and demo-supervised")
        if not self.config.testnet:
            raise base.GuardrailViolation("5m runner is demo/testnet-only; mainnet is blocked")
        if self.config.allocated_capital_usdt != Decimal("2000"):
            raise base.GuardrailViolation("5m demo runner requires allocated capital = 2000 USDT")
        if self.config.leverage != 2:
            raise base.GuardrailViolation("5m demo runner requires LEVERAGE=2")
        if self.config.max_notional_usdt <= 0:
            raise base.GuardrailViolation("MAX_NOTIONAL_USDT must be positive")
        if self.config.max_notional_usdt > Decimal("100"):
            raise base.GuardrailViolation("5m demo runner caps MAX_NOTIONAL_USDT at 100 USDT")
        if self.config.max_capital_pct_per_trade <= 0 or self.config.max_capital_pct_per_trade > Decimal("5"):
            raise base.GuardrailViolation("MAX_CAPITAL_PCT_PER_TRADE must be > 0 and <= 5 for 5m demo")
        if self.config.max_positions != 1:
            raise base.GuardrailViolation("5m demo runner is restricted to MAX_POSITIONS=1")
        if self.config.max_daily_loss_pct <= 0 or self.config.max_cumulative_loss_pct <= 0:
            raise base.GuardrailViolation("Loss limits must be positive percentages")

        if self.config.mode == "demo-supervised":
            if self.config.dry_run:
                self.event_logger.emit("demo_supervised_dry_run", reason="demo orders require --allow-demo-orders and DEMO_TRADING=true")
            elif os.getenv(self.config.allow_demo_env_var, "").lower() != "true":
                raise base.GuardrailViolation(f"Set {self.config.allow_demo_env_var}=true to permit demo/testnet orders")

        if self.config.mode in {"test-order", "demo-supervised"}:
            auth_type = os.getenv("BINANCE_FUTURES_AUTH_TYPE", "HMAC").strip().upper()
            if not os.getenv("BINANCE_FUTURES_API_KEY"):
                raise base.GuardrailViolation("Missing BINANCE_FUTURES_API_KEY")
            if auth_type == "HMAC" and not os.getenv("BINANCE_FUTURES_API_SECRET"):
                raise base.GuardrailViolation("Missing BINANCE_FUTURES_API_SECRET for HMAC authentication")
            if auth_type == "ED25519" and not os.getenv("BINANCE_FUTURES_PRIVATE_KEY_PATH"):
                raise base.GuardrailViolation("Missing BINANCE_FUTURES_PRIVATE_KEY_PATH for Ed25519 authentication")

        self.event_logger.emit(
            "guardrails_static_passed",
            mode=self.config.mode,
            dry_run=self.config.dry_run,
            testnet=self.config.testnet,
            symbol=SYMBOL,
            interval=INTERVAL,
            direction="short_only",
            research_baseline="btcusdt_usdm_5m_research_backtester.short_only",
            profitability_approval=False,
            auth_type=os.getenv("BINANCE_FUTURES_AUTH_TYPE", "HMAC").strip().upper(),
            allocated_capital=str(self.config.allocated_capital_usdt),
            max_notional=str(self.config.max_notional_usdt),
            max_capital_pct_per_trade=str(self.config.max_capital_pct_per_trade),
            max_positions=self.config.max_positions,
            max_daily_loss_pct=str(self.config.max_daily_loss_pct),
            max_cumulative_loss_pct=str(self.config.max_cumulative_loss_pct),
            leverage=self.config.leverage,
            margin_type=self.config.margin_type,
        )

    def bootstrap_historical_candles(self) -> None:
        klines = self.client.public_get(
            "/fapi/v1/klines",
            {"symbol": SYMBOL, "interval": INTERVAL, "limit": BOOTSTRAP_CANDLE_LIMIT},
        )
        now = pd.Timestamp.now(tz="UTC")
        candles: list[dict[str, Any]] = []
        for kline in klines:
            close_time = pd.to_datetime(int(kline[6]), unit="ms", utc=True)
            if close_time > now:
                continue
            candles.append(
                {
                    "timestamp": close_time,
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                    "is_closed": True,
                }
            )
        self.candles = candles[-BOOTSTRAP_CANDLE_LIMIT:]
        signal = evaluate_5m_short_signal(self.candles)
        latest = self.candles[-1] if self.candles else None
        self.event_logger.emit(
            "historical_candles_bootstrapped",
            candles=len(self.candles),
            latest_candle_time=str(latest["timestamp"]) if latest else None,
            latest_close=str(latest["close"]) if latest else None,
        )
        self.event_logger.emit(
            "bootstrap_signal_evaluated",
            signal=signal,
            candles=len(self.candles),
            candle_time=str(latest["timestamp"]) if latest else None,
            traded=False,
            reason="startup_signal_is_context_only_waiting_for_next_closed_stream_candle",
        )

    async def market_data_loop(self) -> None:
        while True:
            try:
                async with base.websockets.connect(self.runtime.market_ws_url, ping_interval=180, ping_timeout=600) as websocket:
                    self.event_logger.emit("market_stream_connected", url=self.runtime.market_ws_url)
                    async for raw in websocket:
                        payload = json.loads(raw)
                        data = payload.get("data", payload)
                        event_type = data.get("e")
                        if event_type == "markPriceUpdate":
                            self.latest_mark_price = Decimal(str(data["p"]))
                            continue
                        candle = base.parse_kline_stream(data)
                        if candle is None or not candle["is_closed"]:
                            continue
                        self.closed_candles_processed += 1
                        self.candles.append(candle)
                        self.candles = self.candles[-BOOTSTRAP_CANDLE_LIMIT:]
                        signal = evaluate_5m_short_signal(self.candles)
                        self.event_logger.emit(
                            "signal_evaluated",
                            candle_time=str(candle.get("timestamp")),
                            close=str(candle["close"]),
                            mark_price=str(self.latest_mark_price) if self.latest_mark_price else None,
                            signal=signal,
                            candles=len(self.candles),
                            direction="short_only",
                            timeframe=INTERVAL,
                        )
                        await self.handle_signal(signal, str(candle.get("timestamp")))
                        if self.max_closed_candles > 0 and self.closed_candles_processed >= self.max_closed_candles:
                            raise SessionComplete("max_closed_candles_reached")
            except asyncio.CancelledError:
                raise
            except SessionComplete:
                raise
            except Exception as exc:
                self.event_logger.emit("market_stream_error", error=str(exc))
                if self.config.mode != "observe":
                    raise
                await asyncio.sleep(5)


def parse_bool(value: str, default: bool = False) -> bool:
    return base.parse_bool(value, default)


def decimal_from_5m_env(name: str, fallback: str) -> Decimal:
    return Decimal(os.getenv(name, fallback))


def int_from_5m_env(name: str, fallback: str) -> int:
    return int(os.getenv(name, fallback))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo-only BTCUSDT USD-M 5m short-only runner")
    parser.add_argument("--mode", choices=["observe", "test-order", "demo-supervised"], default="observe")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Optional .env file. Existing environment variables win.")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--allow-demo-orders", action="store_true", help="Only meaningful with demo-supervised plus DEMO_TRADING=true.")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--interval", default=INTERVAL)
    parser.add_argument("--allocated-capital-usdt", default=None)
    parser.add_argument("--max-notional-usdt", default=None)
    parser.add_argument("--max-capital-pct-per-trade", default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--max-daily-loss-pct", default=None)
    parser.add_argument("--max-cumulative-loss-pct", default=None)
    parser.add_argument("--test-order-notional-usdt", default=None)
    parser.add_argument("--leverage", type=int, default=None)
    parser.add_argument("--margin-type", default=None, choices=["ISOLATED", "CROSSED", "isolated", "crossed"])
    parser.add_argument("--log-file", type=Path, default=Path("logs/live_candidate_5m_events.jsonl"))
    parser.add_argument("--max-runtime-seconds", type=int, default=0, help="Stop cleanly after this many seconds. 0 means no limit.")
    parser.add_argument("--max-closed-candles", type=int, default=0, help="Stop cleanly after this many closed 5m candles. 0 means no limit.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = build_arg_parser().parse_args()
    base.load_env_file(args.env_file)

    os.environ["FUTURES_SYMBOL"] = SYMBOL
    os.environ["FUTURES_INTERVAL"] = INTERVAL
    testnet = True
    base.apply_demo_key_aliases(testnet)

    if args.symbol != SYMBOL or args.interval != INTERVAL:
        raise base.GuardrailViolation("This runner is intentionally locked to FUTURES_SYMBOL=BTCUSDT and FUTURES_INTERVAL=5m")

    dry_run = True
    if args.mode == "demo-supervised" and args.allow_demo_orders and parse_bool(os.getenv("DEMO_TRADING", "")):
        dry_run = False

    config = base.LiveCandidateConfig(
        mode=args.mode,
        testnet=testnet,
        dry_run=dry_run,
        allocated_capital_usdt=Decimal(str(args.allocated_capital_usdt)) if args.allocated_capital_usdt else decimal_from_5m_env("FIVE_MIN_ALLOCATED_CAPITAL_USDT", "2000"),
        max_notional_usdt=Decimal(str(args.max_notional_usdt)) if args.max_notional_usdt else decimal_from_5m_env("FIVE_MIN_MAX_NOTIONAL_USDT", "60"),
        max_capital_pct_per_trade=Decimal(str(args.max_capital_pct_per_trade)) if args.max_capital_pct_per_trade else decimal_from_5m_env("FIVE_MIN_MAX_CAPITAL_PCT_PER_TRADE", "3"),
        max_positions=args.max_positions if args.max_positions is not None else int_from_5m_env("FIVE_MIN_MAX_POSITIONS", "1"),
        max_daily_loss_pct=Decimal(str(args.max_daily_loss_pct)) if args.max_daily_loss_pct else decimal_from_5m_env("FIVE_MIN_MAX_DAILY_LOSS_PCT", "2"),
        max_cumulative_loss_pct=Decimal(str(args.max_cumulative_loss_pct)) if args.max_cumulative_loss_pct else decimal_from_5m_env("FIVE_MIN_MAX_CUMULATIVE_LOSS_PCT", "6"),
        test_order_notional_usdt=Decimal(str(args.test_order_notional_usdt)) if args.test_order_notional_usdt else decimal_from_5m_env("FIVE_MIN_TEST_ORDER_NOTIONAL_USDT", "60"),
        leverage=args.leverage if args.leverage is not None else int_from_5m_env("FIVE_MIN_LEVERAGE", "2"),
        margin_type=(args.margin_type or os.getenv("FIVE_MIN_MARGIN_TYPE", "ISOLATED")).upper(),
        log_file=args.log_file,
    )
    runner = LiveCandidateRunner5m(config)
    runner.max_runtime_seconds = max(args.max_runtime_seconds, 0)
    runner.max_closed_candles = max(args.max_closed_candles, 0)
    try:
        asyncio.run(runner.run())
    except base.GuardrailViolation as exc:
        logger.error("Guardrail violation: %s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
